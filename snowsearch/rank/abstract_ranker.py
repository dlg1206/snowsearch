import json
import math
from collections import deque
from json import JSONDecodeError
from typing import List

from ai.model import ModelClient
from openalex.dto import PaperDTO
from rank.config import MIN_ABSTRACT_PER_COMPARISON, AVG_TOKEN_PER_WORD, RANK_CONTEXT_FILE, TOKEN_BUFFER_MODIFIER, \
    MAX_RETRIES
from rank.dto import AbstractDTO
from rank.exception import ExceedMaxRankingGenerationAttemptsError
from util.logger import logger
from util.timer import Timer

"""
File: abstract_ranker.py

Description: Rank papers based on the relevance of their abstracts

@author Derek Garcia
"""


class AbstractRanker:

    def __init__(self,
                 model_client: ModelClient,
                 context_window: int,
                 abstracts_per_comparison: int = MIN_ABSTRACT_PER_COMPARISON,
                 tokens_per_word: float = AVG_TOKEN_PER_WORD):
        """
        Create new Abstract ranker

        :param model_client: Client to use for ranking abstracts
        :param context_window: Context window of the model
        :param abstracts_per_comparison: Number of abstracts per ranking (Default: 2)
            If using a larger model, this can be increased to reduce the number of API calls made or kept low to keep a
            more detailed analysis of each abstract. For smaller models, keeping this number low keeps the abstracts
            within the smaller context window
        :param tokens_per_word: Average tokens per word to use for window estimation
        """
        self._model_client = model_client
        self._abstracts_per_comparison = abstracts_per_comparison
        self._token_per_word = tokens_per_word

        # load content for one-shot
        with open(RANK_CONTEXT_FILE, 'r') as f:
            self._rank_context = f.read()

        # reserve tokens for one-shot
        self._context_window_budget = context_window - math.ceil(
            self._estimate_tokens(self._rank_context) * TOKEN_BUFFER_MODIFIER)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens the abstract will take

        :param text: Text content of abstract
        :return: Estimated number of tokens
        """
        return math.ceil(len(text.split()) / self._token_per_word)

    def _bin_pack_min_bins(self, current_abstracts: List[AbstractDTO]) -> List[List[AbstractDTO]]:
        """
        Pack bins of abstracts to limit the number
        of comparison api calls to the llm

        :param current_abstracts: Current round of abstracts
        :return: Optimal packing of the batch of abstracts
        """
        # Sort by token size ascending
        indexed = sorted(current_abstracts, key=lambda x: x.tokens)
        abstracts = deque(indexed)

        # shrink bins until within min range
        bins = []
        while abstracts:
            current_bin = []
            total_abstracts = 0
            total_tokens = 0
            # greedy pack bin
            while (
                    abstracts
                    and total_abstracts < self._abstracts_per_comparison
                    and total_tokens + abstracts[0].tokens <= self._context_window_budget
            ):
                abstract = abstracts.popleft()
                current_bin.append(abstract)
                total_abstracts += 1
                total_tokens += abstract.tokens

            # Enforce at least min papers per bin
            if total_abstracts < MIN_ABSTRACT_PER_COMPARISON and bins:
                bins[-1].extend(current_bin)
            else:
                bins.append(current_bin)

        return bins

    def _rank_abstracts(self, prompt: str, abstracts: List[AbstractDTO]) -> List[AbstractDTO]:
        """
        Use an LLM to rank abstracts from most to least relevant based on the provided prompt

        :param prompt: Natural language query for papers
        :raises ExceedMaxRankingGenerationAttemptsError: If fail to extract ranking from model reply
        :return: Ordered list of most relevant abstracts
        """
        # format prompt
        final_prompt = "\n"
        abstract_lookup = {}
        for a in abstracts:
            final_prompt += f"Title:\n{a.id}\nAbstract:\n{a.text}\n\n"
            abstract_lookup[a.id] = a

        final_prompt += f"Search:\n{prompt.strip()}"

        context = self._rank_context.replace("{total_abstracts}", str(len(abstracts)))

        # warn if exceed budget
        total_tokens = self._estimate_tokens(final_prompt) + self._estimate_tokens(context)
        if total_tokens > self._context_window_budget:
            logger.warn(f"Exceeded context budget by {total_tokens} tokens, ranking may be impacted")

        # error if exceed retries
        for attempt in range(0, MAX_RETRIES):
            logger.debug_msg(f"Ranking {len(abstracts)} abstracts ({attempt + 1}/{MAX_RETRIES})")
            completion, timer = self._model_client.prompt(
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0
            )
            '''
            Attempt to extract the ranking from the response. 
            This is to safeguard against wordy and descriptive replies 
            '''
            try:
                results = json.loads(completion.choices[0].message.content.strip())
                # convert back to abstracts
                return [abstract_lookup[aid] for _, aid in sorted(results.items(), key=lambda x: int(x[0]))]
            except JSONDecodeError:
                # else retry
                if attempt + 1 < MAX_RETRIES:
                    logger.warn("Failed to generate ranking, retrying. . .")
                    continue
        # error if exceed retries
        raise ExceedMaxRankingGenerationAttemptsError(self._model_client.model)

    def _filter_abstracts(self, prompt: str, abstracts: List[AbstractDTO], min_abstracts: int) -> List[AbstractDTO]:
        """
        Filter abstracts tournament style to get the top most relevant papers

        :param prompt: Search prompt to match
        :param abstracts: List of abstracts to filter
        :param min_abstracts: Minimum abstracts that must be returned
        :return: List of top abstracts
        """

        round_num = 0
        current_abstracts = abstracts
        logger.info(f"Filtering {len(current_abstracts)} papers")
        timer = Timer()
        while True:

            # Step 1: Bin packing
            bins = self._bin_pack_min_bins(current_abstracts)

            # break if bins will eliminate enough to not meet top n quota
            if len(bins) <= min_abstracts:
                break

            logger.info(f"Round {round_num + 1}: {len(current_abstracts)} papers remain")
            logger.debug_msg(f"Processing {len(bins)} matches, comparing {min(bins, key=len)} to {max(bins, key=len)} abstracts per round")

            # Step 2: Select one from each bin
            selected = []
            for b in logger.get_data_queue(bins, f"Round {round_num + 1} | Ranking with LLM", "ranking"):
                selected.append(self._rank_abstracts(prompt, b)[0])  # get the best ranked abstract

            current_abstracts = selected
            round_num += 1

        logger.info(f"Completed filtering in {timer.format_time()}s | {len(current_abstracts)} papers remain")
        return current_abstracts

    def rank_papers(self, prompt: str, papers: List[PaperDTO], top_n: int) -> List[PaperDTO]:
        """
        Rank a list of papers using an llm

        :param prompt: Search prompt to match
        :param papers: List of papers to rank
        :param top_n: Number of top papers to find
        :return: Ordered list of the most relevant to the prompt
        """
        # guard against pointless prompting
        match len(papers):
            case 1:
                logger.warn("Only one paper provided, skipping ranking")
                return papers
            case 0:
                logger.error_msg("No papers provided, skipping ranking")
                return papers

        # Convert to AbstractDTOs with token estimates
        abstracts = [
            AbstractDTO(p.id, p.abstract_text, self._estimate_tokens(p.abstract_text))
            for p in papers
        ]
        # filter papers if not enough to do final ranking
        top_abstracts = self._filter_abstracts(prompt, abstracts, top_n) if len(abstracts) > top_n else abstracts

        logger.info(f"Performing final ranking")
        timer = Timer()
        ranked_abstracts = self._rank_abstracts(prompt, top_abstracts)
        logger.info(f"Final ranking determined in {timer.format_time()}s")

        # convert back to papers
        return [PaperDTO(a.id, abstract_text=a.text) for a in ranked_abstracts]
