import asyncio
import os
from asyncio import Semaphore
from typing import List, Dict, Tuple, Any

from aiohttp import ClientSession, ClientResponseError
from yarl import URL

from client.ai.model import ModelClient
from client.grobid.dto import CitationDTO
from client.openalex.config import POLITE_RATE_LIMIT_SLEEP, DEFAULT_RATE_LIMIT_SLEEP, MAX_PER_PAGE, OPENALEX_BASE, \
    QUERY_JSON_RE, MAX_RETRIES, NL_TO_QUERY_CONTEXT_FILE
from client.openalex.dto import PaperDTO
from client.openalex.exception import MissingOpenAlexEntryError, ExceedMaxQueryGenerationAttemptsError
from db.config import DOI_PREFIX
from db.paper_database import PaperDatabase
from util.logger import logger

"""
File: openalex_client.py
Description: Client for interacting with the OpenAlex Database

https://docs.openalex.org/
https://docs.openalex.org/api-guide-for-llms
@author Derek Garcia
"""


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
        if not email:
            logger.warn("No email provided for OpenAlex - Using slower API")
        self._rate_limit = POLITE_RATE_LIMIT_SLEEP if email else DEFAULT_RATE_LIMIT_SLEEP
        self._api_key_available = False  # default false
        # check for OpenAlex API key
        if os.getenv('OPENALEX_API_KEY'):
            self._api_key_available = True
            logger.info("Found OpenAlex API key")
        # load content for few-shot
        with open(NL_TO_QUERY_CONTEXT_FILE, 'r') as f:
            self._nl_to_query_context = f.read()

    def _add_auth(self, params_obj: Dict[str, str | int]) -> None:
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

    async def _fetch(self, session: ClientSession,
                     openalex_endpoint: str,
                     params: Dict[str, str | int] = None,
                     per_page: int = MAX_PER_PAGE) -> Dict[str, Any]:
        """
        Fetch data from openalex api

        :param session: HTTP session to use
        :param openalex_endpoint: Endpoint to query
        :param params: Optional additional url params to include (Default: None)
        :param per_page: Number of results to fetch per page (Default: 200)
        :return: JSON response
        """
        # init params if dne
        if not params:
            params = dict()
        # update params
        self._add_auth(params)
        params['per_page'] = per_page
        # block to respect rate limit
        await asyncio.sleep(POLITE_RATE_LIMIT_SLEEP)
        # make the request, remove key for logging
        url = URL(f"{OPENALEX_BASE}/{openalex_endpoint.removeprefix('/')}").with_query(params).without_query_params(
            'OPENALEX_API_KEY')
        logger.debug_msg(f"Querying '{url}'")
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def _fetch_page(self, session: ClientSession, query: str, cursor: str = '*') -> Tuple[str, List[PaperDTO]]:
        """
        Search for query from OpenAlex

        :param session: HTTP session to use
        :param query: Query string to use
        :param cursor: Cursor to use for pagination (Default: starting cursor)
        https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging?q=per_page#cursor-paging
        :return: Cursor for next page and list of papers
        """
        # fetch papers
        params = {'filter': f"title_and_abstract.search:{query}", 'cursor': cursor}
        result = await self._fetch(session, "works", params)

        # parse findings
        return result['meta']['next_cursor'], [
            PaperDTO(p['title'],
                     openalex_id=p['id'],
                     doi=p['doi'],
                     is_open_access=bool(p['open_access']['is_oa']),
                     pdf_url=p['open_access']['oa_url'])
            for p in result.get('results', [])
        ]

    async def _fetch_paper_count(self, session: ClientSession, query: str) -> int:
        """
        Get the number of papers that match the given query

        :param session: HTTP session to use
        :param query: Query string to use
        :return: Number of papers the query matches
        """
        result = await self._fetch(session, "works", {'filter': f"title_and_abstract.search:{query}"}, 1)
        return int(result['meta']['count'])

    async def _fetch_by_doi(self, session: ClientSession, doi: str) -> PaperDTO | None:
        """
        Fetch paper from OpenAlex by DOI

        :param session: HTTP session to use
        :param doi: DOI identifier to search for
        :return: Matching OpenAlex paper, None if no matches
        """
        # make request
        try:
            paper = await self._fetch(session, f"works/{DOI_PREFIX}{doi}", per_page=1)
            # parse results
            return PaperDTO(paper['title'],
                            openalex_id=paper['id'],
                            doi=doi,
                            is_open_access=bool(paper['open_access']['is_oa']),
                            pdf_url=paper['open_access']['oa_url'])
        except ClientResponseError:
            # todo - assuming 404
            # return none if no hits
            return None

    async def _fetch_by_exact_title(self, session: ClientSession, title: str) -> PaperDTO | None:
        """
        Fetch paper from OpenAlex by exact title match

        :param session: HTTP session to use
        :param title: Title of paper to search for
        :return: Matching OpenAlex paper, None if no matches
        """
        # make request
        # remove commas since aren't supported in search
        # https://github.com/ropensci/openalexR/issues/254
        result = await self._fetch(session,
                                   "works", {'filter': f"title_and_abstract.search:{title.replace(',', ' ')}"},
                                   1)

        # return none if no hits
        if not result['meta']['count']:
            return None
        # else parse results
        paper = result['results'][0]
        # not exact match
        if paper['title'].lower() != title.lower():
            return None
        return PaperDTO(paper['title'],
                        openalex_id=paper['id'],
                        doi=paper['doi'],
                        is_open_access=bool(paper['open_access']['is_oa']),
                        pdf_url=paper['open_access']['oa_url'])

    async def _fetch_citation_task(self,
                                   semaphore: Semaphore,
                                   session: ClientSession,
                                   doi: str | None,
                                   title: str | None) -> PaperDTO | None:
        """
        Task to fetch details for a single citation / paper
        First attempt to get by DOI if provided, then title

        :param semaphore: Semaphore to limit concurrent requests
        :param session: HTTP session to use
        :param doi: DOI id to search for
        :param title: Fallback title to search for
        :return: Matching OpenAlex paper, None if no matches
        """
        method = None
        paper = None
        async with semaphore:
            # first search by doi
            if doi:
                paper = await self._fetch_by_doi(session, doi)
                method = 'doi'
            # if couldn't find by doi or no doi, try title
            if title and not paper:
                paper = await self._fetch_by_exact_title(session, title)
                method = 'title'
        # return paper if found
        if paper:
            logger.debug_msg(f"Found '{paper.id}' by {method}")
            return paper
        # else raise error
        raise MissingOpenAlexEntryError(doi, title)

    async def save_seed_papers(self, run_id: int, paper_db: PaperDatabase, query: str) -> None:
        """
        Fetch papers from OpenAlex based on a query and save to database

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
                    next_cursor, papers = await self._fetch_page(session, query, next_cursor)
                    paper_db.insert_run_paper_batch(run_id, papers)
                    break
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

    async def save_citation_papers(self, paper_db: PaperDatabase, citations: List[CitationDTO]) -> None:
        """
        Fetch details for the given list of citations and save to database

        :param paper_db: Database to save papers to
        :param citations: List of citations to fetch details for
        """
        semaphore = Semaphore()  # semaphore so only 1 request at a time to prevent tripping rate limit
        num_success = 0
        num_missing = 0
        logger.debug_msg(f"Searching for details for {len(citations)} citations")
        async with ClientSession() as session:
            # todo - bulk requests https://docs.openalex.org/api-guide-for-llms#bulk-lookup-by-dois
            tasks = [self._fetch_citation_task(semaphore, session, c.doi, c.id) for c in citations]
            for future in logger.get_data_queue(tasks, "Fetching citation details", "citation", is_async=True):
                try:
                    paper: PaperDTO = await future
                    # update citation
                    paper_db.upsert_paper(paper)
                    num_success += 1
                except MissingOpenAlexEntryError as e:
                    logger.error_exp(e)
                    paper_db.upsert_paper(PaperDTO(e.title, openalex_status=404))
                    num_missing += 1
                except Exception as e:
                    # todo - handle exceed requests per day
                    logger.error_exp(e)

        # report results
        percent = lambda a, b: f"{(a / b) * 100:.01f}%"
        logger.info(
            f"Search complete, successfully updated {num_success} citations ({percent(num_success, len(citations))})")
        logger.info(f"Failed to find {num_success} citations ({percent(num_missing, len(citations))})")

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

    @property
    def model(self) -> str:
        return self._model_client.model
