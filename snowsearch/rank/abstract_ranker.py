import json
import math
from json import JSONDecodeError
from typing import List, Tuple, Dict

from ai.model import ModelClient
from rank.config import AVG_TOKEN_PER_WORD, RANK_CONTEXT_FILE, TOKEN_BUFFER_MODIFIER, \
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
                 tokens_per_word: float = AVG_TOKEN_PER_WORD):
        """
        Create new Abstract ranker

        :param model_client: Client to use for ranking abstracts
        :param context_window: Context window of the model
        :param tokens_per_word: Average tokens per word to use for window estimation
        """
        self._model_client = model_client
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

    def _format_context_and_prompt(self, nl_query: str, abstracts: List[AbstractDTO]) -> Tuple[str, str]:
        """
        Format the prompt to send to the LLM to rank the abstracts

        :param nl_query: Natural language search query to best match papers to
        :param abstracts: List of abstracts to embed into the prompt
        :return: Formated prompt and context
        """
        # built prompt
        final_prompt = "\n"
        for a in abstracts:
            final_prompt += f"id: {a.id}\nAbstract:\n{a.text}\n\n"
        final_prompt += f"Search:\n{nl_query.strip()}"

        # return context and prompt
        return self._rank_context.replace("{total_abstracts}", str(len(abstracts))), final_prompt

    def _rank_with_llm(self, context: str, prompt: str, abstract_lookup: Dict[str, AbstractDTO]) -> List[
        AbstractDTO]:
        """
        Submit prompt to an LLM to rank abstracts

        :param context: LLM context with examples
        :param prompt: Prompt to submit to LLM
        :param abstract_lookup: Lookup dict for mapping the temp ID to abstract object
        :raises ExceedMaxRankingGenerationAttemptsError: If fail to extract ranking from model reply
        :return: Ordered list of abstracts based on LLM ranking
        """
        # warn if exceed budget
        total_tokens = self._estimate_tokens(context) + self._estimate_tokens(prompt)
        if total_tokens > self._context_window_budget:
            logger.warn(f"Exceeded context budget by {total_tokens} tokens, ranking may be impacted")

        # error if exceed retries
        for attempt in range(0, MAX_RETRIES):
            logger.debug_msg(f"Ranking attempt {attempt + 1}/{MAX_RETRIES}")
            completion, timer = self._model_client.prompt(
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            '''
            Attempt to extract the ranking from the response. 
            This is to safeguard against wordy and descriptive replies 
            '''
            try:
                results = json.loads(completion.choices[0].message.content.strip())
                # convert back to dtos in order
                return [abstract_lookup[results[key]] for key in sorted(results.keys(), key=int)]
            except JSONDecodeError:
                # else retry
                if attempt + 1 < MAX_RETRIES:
                    logger.warn("Failed to generate ranking, retrying. . .")
                    continue
        # error if exceed retries
        raise ExceedMaxRankingGenerationAttemptsError(self._model_client.model)

    def rank_abstracts(self, nl_query: str, abstracts: List[AbstractDTO]) -> List[AbstractDTO]:
        """
        Rank a list of abstracts using an LLM

        :param nl_query: Natural language search query to best match papers to
        :param abstracts: List of paper abstracts to rank
        :return: Ordered list of the most relevant abstracts to the search query
        """
        # guard against pointless prompting
        match len(abstracts):
            case 1:
                logger.warn("Only one abstract provided, skipping ranking")
                return abstracts
            case 0:
                logger.error_msg("No abstracts provided, skipping ranking")
                return abstracts

        # format prompt
        context, prompt = self._format_context_and_prompt(nl_query, abstracts)
        # rank abstracts
        logger.info(f"Ranking {len(abstracts)} abstracts")
        timer = Timer()
        ranked_abstracts = self._rank_with_llm(context, prompt, abstract_lookup={a.id: a for a in abstracts})
        logger.info(f"Final ranking determined in {timer.format_time()}s")
        # return results
        return ranked_abstracts
