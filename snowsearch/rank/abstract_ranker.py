import asyncio
import contextlib
import json
import math
from json import JSONDecodeError
from typing import List, Tuple, Dict

from ai.model import ModelClient
from dto.paper_dto import PaperDTO
from rank.config import AVG_TOKEN_PER_WORD, RANK_CONTEXT_FILE, TOKEN_BUFFER_MODIFIER, \
    MAX_RETRIES
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

    def _format_context_and_prompt(self, nl_query: str, papers: List[PaperDTO]) -> Tuple[str, str]:
        """
        Format the prompt to send to the LLM to rank the abstracts

        :param nl_query: Natural language search query to best match papers to
        :param papers: List of papers to embed into the prompt
        :return: Formated prompt and context
        """
        # built prompt
        final_prompt = "\n"
        for p in papers:
            final_prompt += f"id: {p.generate_short_uid()}\nAbstract:\n{p.abstract_text}\n\n"
        final_prompt += f"Search:\n{nl_query.strip()}"

        # return context and prompt
        return self._rank_context.replace("{total_abstracts}", str(len(papers))), final_prompt

    async def _rank_with_llm(self, context: str, prompt: str, paper_lookup: Dict[str, PaperDTO]) -> List[PaperDTO]:
        """
        Submit prompt to an LLM to rank abstracts

        :param context: LLM context with examples
        :param prompt: Prompt to submit to LLM
        :param paper_lookup: Lookup dict for mapping the temp ID to paper object
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
            completion, timer = await self._model_client.prompt(
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
                return [paper_lookup[results[key]] for key in sorted(results.keys(), key=int)]
            except JSONDecodeError:
                # else retry
                if attempt + 1 < MAX_RETRIES:
                    logger.warn("Failed to generate ranking, retrying. . .")
                    continue
        # error if exceed retries
        raise ExceedMaxRankingGenerationAttemptsError(self._model_client.model)

    async def rank_paper_abstracts(self, nl_query: str, papers: List[PaperDTO]) -> List[PaperDTO]:
        """
        Rank a list of abstracts using an LLM

        :param nl_query: Natural language search query to best match papers to
        :param papers: List of papers to rank
        :return: Ordered list of the most relevant abstracts to the search query
        """
        # guard against pointless prompting
        match len(papers):
            case 1:
                logger.warn("Only one abstract provided, skipping ranking")
                return papers
            case 0:
                logger.error_msg("No abstracts provided, skipping ranking")
                return papers

        # format prompt
        context, prompt = self._format_context_and_prompt(nl_query, papers)
        # rank abstracts
        logger.info(f"Ranking {len(papers)} abstracts, this may take a while")
        timer = Timer()

        async def __heartbeat():
            """
            Heartbeat for long running llm ranking
            """
            try:
                while True:
                    await asyncio.sleep(5)
                    logger.info(f"{timer.format_time()} elapsed")
            except asyncio.CancelledError:
                pass

        hb = asyncio.create_task(__heartbeat())
        try:
            ranked_abstracts = await self._rank_with_llm(
                context,
                prompt,
                paper_lookup={p.generate_short_uid(): p for p in papers},
            )
        finally:
            # Always stop timer and cancel heartbeat even on exceptions
            timer.stop()
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

        logger.info(f"Final ranking determined in {timer.format_time()}s")
        # return results
        return ranked_abstracts
