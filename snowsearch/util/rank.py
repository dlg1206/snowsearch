import math
from collections import deque
from dataclasses import dataclass
from typing import List

from ai.model import ModelClient
from openalex.dto import PaperDTO
from util.logger import logger
from util.timer import Timer

"""
File: rank.py

Description: Rank papers based on the relevance of their abstracts

@author Derek Garcia
"""

AVG_TOKEN_PER_WORD = 1.2
MIN_ABSTRACT_PER_COMPARISON = 2
RESERVED_TOKENS = 1000


@dataclass
class AbstractDTO:
    id: str
    abstract: str
    tokens: int


class AbstractRanker:

    def __init__(self,
                 model_client: ModelClient,
                 context_window: int,
                 abstracts_per_comparison: int = MIN_ABSTRACT_PER_COMPARISON,
                 token_per_word: int = AVG_TOKEN_PER_WORD):
        """
        Create new Abstract ranker

        :param model_client: Client to use for ranking abstracts
        :param context_window: Context window of the model
        :param abstracts_per_comparison: Number of abstracts per ranking (Default: 2)
            If using a larger model, this can be increased to reduce the number of API calls made or kept low to keep a
            more detailed analysis of each abstract. For smaller models, keeping this number low keeps the abstracts
            within the smaller context window
        :param token_per_word: Average tokens per word to use for window estimation
        """
        self._model_client = model_client
        self._context_window_budget = context_window - RESERVED_TOKENS  # reserve tokens for one-shot
        self._abstracts_per_comparison = abstracts_per_comparison
        self._token_per_word = token_per_word

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

    def _filter_abstracts(self, abstracts: List[AbstractDTO], min_abstracts: int) -> List[AbstractDTO]:
        """
        Filter abstracts tournament style to get the top most relevant papers

        :param abstracts: List of abstracts to filter
        :param min_abstracts: Minimum abstracts that must be returned
        :return: List of top abstracts
        """

        round_num = 0
        current_abstracts = abstracts
        logger.info(f"Filtering {len(current_abstracts)} papers")
        timer = Timer()
        while True:
            logger.info(f"Round {round_num + 1}: {len(current_abstracts)} papers remain")

            # Step 1: Bin packing
            bins = self._bin_pack_min_bins(current_abstracts)

            # break if bins will eliminate enough to not meet top n quota
            if len(bins) <= min_abstracts:
                break

            # Step 2: Select one from each bin
            selected = []
            for b in bins:
                # TODO: Replace with LLM-based selection
                winner = min(b, key=lambda x: x.tokens)  # Placeholder for best-in-bin
                selected.append(winner)

            current_abstracts = selected
            round_num += 1

        logger.info(f"Completed filtering in {timer.format_time()}s | {len(current_abstracts)} papers remain")
        return current_abstracts

    def rank_papers(self, prompt: str, papers: List[PaperDTO], top_n: int) -> List[AbstractDTO]:
        """
        Rank a list of papers using an llm

        :param prompt: Search prompt to match
        :param papers: List of papers to rank
        :param top_n: Number of top papers to find
        :return: Ordered list of the most relevant to the prompt
        """
        # Convert to AbstractDTOs with token estimates
        abstracts = [
            AbstractDTO(p.id, p.abstract_text, self._estimate_tokens(p.abstract_text))
            for p in papers
        ]
        top_abstracts = self._filter_abstracts(abstracts, top_n)
        # TODO: Replace with LLM-based selection
        return sorted(top_abstracts, key=lambda x: x.tokens)[:top_n]
