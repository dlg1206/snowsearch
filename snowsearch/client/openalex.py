import asyncio
import dataclasses
import os
from asyncio import Semaphore
from typing import List, Dict, Tuple

from aiohttp import ClientSession

from db.paper_database import PaperDatabase
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

OPENALEX_ENDPOINT = "https://api.openalex.org"


@dataclasses.dataclass
class OpenAlexDTO:
    doi: str
    is_open_access: bool
    pdf_url: str


class OpenAlexClient:
    def __init__(self, email: str = None):
        """
        Create new OpenAlex client

        https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication#the-polite-pool

        :param email: Optional email to add to the polite pool
        """
        self._email = email
        self._api_key_available = False  # default false
        # check for OpenAlex API key
        if os.getenv('OPENALEX_API_KEY'):
            self._api_key_available = True
            logger.info("Found OpenAlex API key")

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

    async def _fetch_by_exact_title(self, semaphore: Semaphore, session: ClientSession, title) -> Tuple[
        str, OpenAlexDTO | None]:
        """
        Search for exact title match from OpenAlex

        :param semaphore: Semaphore to respect rate limiting
        :param session: HTTP session to use
        :param title: Title of paper to get
        :return: Title and data, None if nothing found
        """
        params = {'filter': f"title_and_abstract.search:{title}", 'per-page': 1}
        self._add_auth(params)
        # block to respect rate limit
        async with semaphore:
            await asyncio.sleep(RATE_LIMIT_SLEEP)
            # make the request
            async with session.get(f"{OPENALEX_ENDPOINT}/works", params=params) as response:
                logger.debug_msg(f"Querying '{response.url}'")
                response.raise_for_status()
                result = await response.json()
            # ensure exact match
            if not len(result['results']) or result['results'][0]['title'] != title:
                logger.warn(f"Did not find any results matching title '{title}'")
                return title, None
                # else build dto
            paper = result['results'][0]
            return title, OpenAlexDTO(paper['doi'], bool(paper['open_access']['is_oa']), paper['open_access']['oa_url'])

    async def save_metadata(self, paper_db: PaperDatabase, titles: List[str]) -> None:
        """
        Fetch metadata from OpenAlex and save results to the paper database

        :param paper_db: Database to save data to
        :param titles: List of paper titles to search for
        """
        semaphore = Semaphore()
        async with ClientSession() as session:
            # todo search by doi if available
            tasks = [self._fetch_by_exact_title(semaphore, session, t.replace(',', '')) for t in titles]
            # wait for batch to process
            for future in logger.get_data_queue(tasks, "Querying OpenAlex Database", "paper", is_async=True):
                try:
                    data: OpenAlexDTO | None
                    title, data = await future
                    # save findings to db
                    if data:
                        paper_db.update_paper(title, doi=data.doi, is_open_access=data.is_open_access,
                                              pdf_url=data.pdf_url)
                except Exception as e:
                    logger.error_exp(e)
