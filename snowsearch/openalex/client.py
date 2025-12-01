import asyncio
import os
from asyncio import Semaphore
from typing import List, Dict, Tuple, Any

from aiohttp import ClientSession
from yarl import URL

from ai.model import ModelClient
from db.config import DOI_PREFIX
from db.paper_database import PaperDatabase
from grobid.dto import CitationDTO
from openalex.config import POLITE_RATE_LIMIT_SLEEP, DEFAULT_RATE_LIMIT_SLEEP, MAX_PER_PAGE, OPENALEX_BASE, \
    QUERY_JSON_RE, MAX_RETRIES, NL_TO_QUERY_CONTEXT_FILE, MAX_DOI_PER_PAGE
from openalex.dto import PaperDTO
from openalex.exception import MissingOpenAlexEntryError, ExceedMaxQueryGenerationAttemptsError
from util.logger import logger

"""
File: client.py
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
        # load content for one-shot
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

    async def _fetch_page(self, session: ClientSession, oa_query: str, cursor: str = '*') -> Tuple[str, List[PaperDTO]]:
        """
        Search for query from OpenAlex

        :param session: HTTP session to use
        :param oa_query: OpenAlex query string to use
        :param cursor: Cursor to use for pagination (Default: starting cursor)
        https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging?q=per_page#cursor-paging
        :return: Cursor for next page and list of papers
        """
        # fetch papers
        params = {'filter': f"title_and_abstract.search:{oa_query}", 'cursor': cursor}
        result = await self._fetch(session, "works", params)

        # parse findings
        return result['meta']['next_cursor'], [
            PaperDTO(p['title'],
                     openalex_url=p['id'],
                     doi=p['doi'],
                     is_open_access=bool(p['open_access']['is_oa']),
                     pdf_url=p['open_access']['oa_url'])
            for p in result.get('results', [])
        ]

    async def _fetch_paper_count(self, session: ClientSession, oa_query: str) -> int:
        """
        Get the number of papers that match the given query

        :param session: HTTP session to use
        :param oa_query: OpenAlex query string to use
        :return: Number of papers the query matches
        """
        result = await self._fetch(session, "works", {'filter': f"title_and_abstract.search:{oa_query}"}, 1)
        return int(result['meta']['count'])

    async def _batch_fetch_by_doi(self, session: ClientSession,
                                  doi_batch: List[str]) -> Tuple[List[PaperDTO], List[str]]:
        """
        Fetch a batch of papers using DOIs

        :param session: HTTP session to use
        :param doi_batch: Batch of DOIs to search for
        :return: List of papers and missing DOIs
        """
        # make request
        params = {'filter': f"doi:{f'|{DOI_PREFIX}'.join(doi_batch)}"}
        results: Dict[str, Any] = await self._fetch(session, f"works", params, per_page=MAX_DOI_PER_PAGE)

        # process results
        missing_doi_ids = set(doi_batch)
        papers = []
        for p in results.get('results', []):
            paper = PaperDTO(p['title'],
                             openalex_url=p['id'],
                             doi=p['doi'],
                             is_open_access=bool(p['open_access']['is_oa']),
                             pdf_url=p['open_access']['oa_url'])
            papers.append(paper)
            missing_doi_ids.remove(paper.doi)
            logger.debug_msg(f"Found '{paper.id}' by doi")

        return papers, list(missing_doi_ids)

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
                        openalex_url=paper['id'],
                        doi=paper['doi'],
                        is_open_access=bool(paper['open_access']['is_oa']),
                        pdf_url=paper['open_access']['oa_url'])

    async def search_and_save_metadata(self, run_id: int, paper_db: PaperDatabase, oa_query: str) -> None:
        """
        Fetch paper metadata from OpenAlex based on a query and save to database

        :param run_id: ID of current run
        :param paper_db: Database to save papers to
        :param oa_query: OpenAlex query string to use
        """
        oa_query = oa_query.replace("'", '"')
        async with ClientSession() as session:
            hits = await self._fetch_paper_count(session, oa_query)
            logger.debug_msg(f"Found {hits} papers in OpenAlex")
            progress = logger.get_data_queue(hits, "Querying OpenAlex Database", "paper")
            # fetch all papers
            next_cursor = "*"
            rank_offset = 0
            while True:
                try:
                    # save results
                    # todo when papers are null
                    next_cursor, papers = await self._fetch_page(session, oa_query, next_cursor)
                    ranked_papers = [(papers[i], i + rank_offset) for i in range(0, len(papers))]
                    rank_offset += len(papers)
                    if ranked_papers:
                        paper_db.insert_run_paper_batch(run_id, ranked_papers)
                    # break
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

    async def fetch_and_save_citation_metadata(self, paper_db: PaperDatabase, citations: List[CitationDTO]) -> None:
        """
        Fetch details for the given list of citations and save to database

        :param paper_db: Database to save papers to
        :param citations: List of citations to fetch details for
        """
        semaphore = Semaphore()  # semaphore so only 1 request at a time to prevent tripping rate limit
        num_doi = 0
        num_title = 0
        num_missing = 0
        logger.debug_msg(f"Searching for details for {len(citations)} citations")
        # split into doi, title and just doi search
        doi_and_title = []
        doi_reverse_lookup = {}
        titles: List[CitationDTO] = []
        for c in citations:
            if c.doi:
                doi_and_title.append(c.doi)
                doi_reverse_lookup[c.doi] = c
            else:
                titles.append(c)
        doi_chunks = [doi_and_title[i:i + MAX_DOI_PER_PAGE] for i in range(0, len(doi_and_title), MAX_DOI_PER_PAGE)]

        async with ClientSession() as session:
            # pass 1 - fetch by DOI
            doi_tasks = [_fetch_doi_batch_wrapper(semaphore, self._batch_fetch_by_doi(session, chunk))
                         for chunk in doi_chunks]
            for future in logger.get_data_queue(doi_tasks, "Fetching citation details by doi", "batch", is_async=True):
                found_papers, missing_doi_ids = await future
                num_doi += len(found_papers)
                # save titles for second pass
                titles += [doi_reverse_lookup.get(doi) for doi in missing_doi_ids]
                # save rest if any
                if found_papers:
                    paper_db.insert_paper_batch(found_papers)

            # pass 2 - fetch by title
            title_tasks = [_fetch_title_wrapper(semaphore, c, self._fetch_by_exact_title(session, c.id)) for c in
                           titles]
            for future in logger.get_data_queue(title_tasks, "Fetching citation details by title", "citation",
                                                is_async=True):
                try:
                    paper = await future
                    num_title += 1
                    paper_db.upsert_paper(paper)
                    logger.debug_msg(f"Found '{paper.id}' by title")
                except MissingOpenAlexEntryError as e:
                    num_missing += 1
                    logger.error_exp(e)
                    paper_db.upsert_paper(PaperDTO(e.title, openalex_status=404))
                except Exception as e:
                    num_missing += 1
                    logger.error_exp(e)

        # report results
        percent = lambda a, b: f"{(a / b) * 100:.01f}%"
        num_success = num_doi + num_title
        logger.info(
            f"Search complete, successfully updated {num_success} citations ({percent(num_success, len(citations))})")
        logger.debug_msg(f"Found {num_doi} citations by DOI ({percent(num_doi, len(citations))})")
        logger.debug_msg(f"Found {num_title} citations by title ({percent(num_title, len(citations))})")
        logger.debug_msg(f"Failed to find {num_missing} citations ({percent(num_missing, len(citations))})")

    async def generate_openalex_query(self, nl_query: str) -> str:
        """
        Use an LLM to convert a natural language search query
        to an OpenAlex style search query based on Elasticsearch query

        https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities#boolean-searches

        :param nl_query: Natural language query for papers
        :raises ExceedMaxQueryGenerationAttemptsError: If fail to extract query from model reply
        :return: OpenAlex query string
        """
        nl_query = nl_query.replace("'", '"')
        # error if exceed retries
        for attempt in range(0, MAX_RETRIES):
            logger.info(f"Generating OpenAlex query ({attempt + 1}/{MAX_RETRIES}) | prompt: {nl_query.strip()}")
            completion, timer = await self._model_client.prompt(
                messages=[
                    {"role": "system", "content": self._nl_to_query_context},
                    {"role": "user", "content": f"\nNatural language prompt:\n{nl_query.strip()}"}
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


async def _fetch_doi_batch_wrapper(semaphore: Semaphore, callback) -> Tuple[List[PaperDTO], List[str]]:
    """
    Util wrapper for DOI batch fetching to respect semaphore

    :param semaphore: Semaphore
    :param callback: DOI batch fetch callback function
    :return: List of papers and missing DOIs
    """
    async with semaphore:
        return await callback


async def _fetch_title_wrapper(semaphore: Semaphore, citation: CitationDTO, callback) -> PaperDTO:
    """
    Util wrapper for title fetching to respect semaphore

    :param semaphore: Semaphore
    :param citation: Citation of Paper to fetch
    :param callback: Title fetch callback function
    :raises MissingOpenAlexEntryError: If could not find paper by title
    :return: Paper details
    """
    async with semaphore:
        r = await callback
    if r:
        return r
    raise MissingOpenAlexEntryError(citation.doi, citation.id)
