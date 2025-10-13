import asyncio
import dataclasses
import os
import re
from datetime import datetime
from typing import List, Dict, Tuple

from aiohttp import ClientSession

from client.ai.model import ModelClient
from db.paper_database import PaperDatabase, DOI_PREFIX
from util.logger import logger

"""
File: openalex.py
Description: Client for interacting with the OpenAlex Database

https://docs.openalex.org/

@author Derek Garcia
"""

# https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication
# 10 requests per second
RATE_LIMIT_SLEEP = 0.1
# https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging?q=per_page#basic-paging
MAX_PER_PAGE = 200

NL_TO_QUERY_CONTEXT_FILE = "snowsearch/prompts/nl_to_elasticsearch_query.prompt"
QUERY_JSON_RE = re.compile(r'\{\n.*"query": "(.*?)"')
# attempts to generate OpenAlex query
MAX_RETRIES = 3

OPENALEX_ENDPOINT = "https://api.openalex.org"
OPENALEX_PREFIX = "https://openalex.org/"


class ExceedMaxQueryGenerationAttemptsError(Exception):
    """Failed to generate valid query string"""

    def __init__(self, model: str):
        super().__init__(f"Exceeded OpenAlex query generation using '{model}'")
        self._model = model

    @property
    def model(self) -> str:
        return self._model


@dataclasses.dataclass
class OpenAlexDTO:
    id: str
    title: str
    doi: str
    is_open_access: bool
    pdf_url: str

    def to_properties(self) -> Dict[str, str | bool | datetime]:
        return {
            'id': self.title,
            'openalex_id': self.id.removeprefix(OPENALEX_PREFIX) if self.id else None,
            'doi': self.doi.removeprefix(DOI_PREFIX) if self.doi else None,
            'is_open_access': self.is_open_access,
            'pdf_url': self.pdf_url,
            'time_added': datetime.now()
        }


class OpenAlexClient:
    def __init__(self, model_client: ModelClient, email: str = None):
        """
        Create new OpenAlex client

        https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication#the-polite-pool

        :param model_client: Client to use for generating OpenAlex queries
        :param email: Optional email to add to the polite pool
        """
        self._model_client = model_client
        self._email = email
        self._api_key_available = False  # default false
        # check for OpenAlex API key
        if os.getenv('OPENALEX_API_KEY'):
            self._api_key_available = True
            logger.info("Found OpenAlex API key")
        # load content for few-shot
        with open(NL_TO_QUERY_CONTEXT_FILE, 'r') as f:
            self._nl_to_query_context = f.read()

    def _add_auth(self, params_obj: Dict[str, str]) -> None:
        """
        Add email and / or api key to params if present

        :param params_obj: Params to update
        """
        # add email if provided
        if self._email:
            params_obj['mailto'] = self._email
        # add api key if available
        if self._api_key_available:
            params_obj['api_key'] = os.getenv('OPENALEX_API_KEY')

    async def _fetch_paper_count(self, session: ClientSession, query: str) -> int:
        """
        Get the number of papers that match the given query

        :param session: HTTP session to use
        :param query: Query string to use
        :return: Number of papers the query matches
        """
        params = {'filter': f"title_and_abstract.search:{query}", 'per_page': 1}
        self._add_auth(params)
        await asyncio.sleep(RATE_LIMIT_SLEEP)
        # make the request
        async with session.get(f"{OPENALEX_ENDPOINT}/works", params=params) as response:
            logger.debug_msg(f"Querying '{response.url}'")
            response.raise_for_status()
            result = await response.json()
        return int(result['meta']['count'])

    async def _fetch_papers(self, session: ClientSession, query: str, cursor: str = '*') -> Tuple[
        str | None, List[OpenAlexDTO]]:
        """
        Search for exact title match from OpenAlex

        :param session: HTTP session to use
        :param query: Query string to use
        :param cursor: Cursor to use for pagination (Default: starting cursor)
        https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging?q=per_page#cursor-paging
        :return: Next cursor and list of papers
        """
        params = {'filter': f"title_and_abstract.search:{query}", 'cursor': cursor, 'per_page': MAX_PER_PAGE}
        self._add_auth(params)
        # block to respect rate limit
        await asyncio.sleep(RATE_LIMIT_SLEEP)
        # make the request
        async with session.get(f"{OPENALEX_ENDPOINT}/works", params=params) as response:
            logger.debug_msg(f"Querying '{response.url}'")
            response.raise_for_status()
            result = await response.json()

        # parse findings
        return result['meta']['next_cursor'], [
            OpenAlexDTO(p['id'], p['title'], p['doi'], bool(p['open_access']['is_oa']), p['open_access']['oa_url'])
            for p in result.get('results', [])
        ]

    def prompt_to_query(self, prompt: str) -> str:
        """
        Use an LLM to convert a natural language search query
        to an OpenAlex style search query based on Elasticsearch query

        https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities#boolean-searches

        :param prompt: Natural language query for papers
        :raises ExceedMaxQueryGenerationAttemptsError: If fail to extract query from model reply
        :return: OpenAlex query string
        """
        # error if exceed retries
        for attempt in range(0, MAX_RETRIES):
            logger.info(f"Generating OpenAlex query ({attempt + 1}/{MAX_RETRIES})")
            completion, timer = self._model_client.prompt(
                messages=[
                    {"role": "system", "content": self._nl_to_query_context},
                    {"role": "user", "content": f"\nNatural language prompt:\n{prompt.strip()}"}
                ],
                temperature=0
            )
            '''
            Attempt to extract the query from the response. 
            This is to safeguard against wordy and descriptive replies 
            '''
            query_match = QUERY_JSON_RE.findall(completion.choices[0].message.content.strip())
            if query_match:
                query = query_match[0].strip().replace("'", '"')
                # report success
                logger.info(f"Generated OpenAlex query in {timer.format_time()}s")
                logger.debug_msg(f"Generated query: {query}")
                return query
            # else retry
            if attempt + 1 < MAX_RETRIES:
                logger.warn("Failed to generate OpenAlex query, retrying. . .")
        # error if exceed retries
        raise ExceedMaxQueryGenerationAttemptsError(self._model_client.model)

    async def save_papers(self, run_id: int, paper_db: PaperDatabase, query: str) -> None:
        """
        Fetch papers from OpenAlex and save to database

        :param run_id: ID of current run
        :param paper_db: Database to save papers to
        :param query: Query string to use
        """
        async with ClientSession() as session:
            hits = await self._fetch_paper_count(session, query)
            logger.debug_msg(f"Found {hits} papers in OpenAlex")
            progress = logger.get_data_queue(hits, "Querying OpenAlex Database", "paper")
            # fetch all papers
            next_cursor = "*"
            while True:
                try:
                    # save results
                    next_cursor, papers = await self._fetch_papers(session, query, next_cursor)
                    paper_db.insert_paper_batch(run_id, [p.to_properties() for p in papers])
                    # no pages left
                    if not next_cursor:
                        break
                except Exception as e:
                    # todo - handle exceed requests per day
                    logger.error_exp(e)
                finally:
                    # update progress
                    if not isinstance(progress, int):
                        progress.update(MAX_PER_PAGE)

    @property
    def model(self) -> str:
        return self._model_client.model
