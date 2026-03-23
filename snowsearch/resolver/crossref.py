"""
File: crossref.py

Description: Client for interacting with the Crossref (https://api.crossref.org/)

https://www.crossref.org/documentation/retrieve-metadata/rest-api/
https://api.crossref.org/swagger-ui/index.html

@author Derek Garcia
"""
import asyncio
import os
from typing import Dict

from aiohttp import ClientSession
from sympy import asec
from yarl import URL

from dto.crossref_dto import CrossrefDTO, SimplePersonDTO, SimpleDateDTO, ConferencePaperDTO, JournalArticleDTO, \
    ThesisDTO, BookSectionDTO
from dto.paper_dto import PaperDTO
from util.logger import logger

CROSSREF_API_KEY_ENV = "CROSSREF_API_KEY"
# https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/#74000
# 5 requests per second
PUBLIC_RATE_LIMIT_SLEEP = 0.2
# 10 requests per second
POLITE_RATE_LIMIT_SLEEP = 0.1
# 150 requests per second
PLUS_RATE_LIMIT_SLEEP = 0.007   # todo - <10ms may be finicky

CROSSREF_BASE="https://api.crossref.org"

class CrossrefClient:
    """
    Client to interact with the Crossref API
    """

    def __init__(self, email: str = None):
        """
        Create new Crossref client

        https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/

        :param email: Optional email to add to the polite pool
        """
        self._email = email
        if not email:
            logger.warn("No email provided for Crossref - Using slower API")

        self._rate_limit = POLITE_RATE_LIMIT_SLEEP if email else PUBLIC_RATE_LIMIT_SLEEP
        self._api_key_available = False  # default false

        # check for Crossref API key
        # https://www.crossref.org/services/metadata-retrieval/metadata-plus/
        if os.getenv(CROSSREF_API_KEY_ENV):
            self._api_key_available = True
            self._rate_limit = PLUS_RATE_LIMIT_SLEEP
            logger.info("Found CrossRef API key")

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
            params_obj['api_key'] = os.getenv(CROSSREF_API_KEY_ENV)

    async def fetch_paper(self, session: ClientSession, doi: str) -> CrossrefDTO:
        # block to respect rate limit
        await asyncio.sleep(self._rate_limit)

        # set the auth
        params, headers = {}, {}
        if self._api_key_available:
            headers = {'Crossref-Plus-API-Token' : f"Bearer {os.getenv(CROSSREF_API_KEY_ENV)}"}
        elif self._email:
            params = {'mailto': self._email}

        # make the request
        url = f"{CROSSREF_BASE}/{doi.strip()}"
        logger.debug_msg(f"Querying '{url}'")
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            result = await response.json()
            data = result.get('message', {})

            # get shared fields
            # has abstract: https://api.crossref.org/works/10.1007/978-3-031-69852-1_7 - in jats format
            ref_type = data.get('type')
            crossref_data = {
                'type': ref_type,
                'title': data.get('title', [None])[0],
                'doi': doi,
                'url': data.get('resource', {}).get('primary', {}).get('URL'),
                'authors': [SimplePersonDTO(a.get('given'), a.get('family')) for a in data.get('author', [])],
                'date': SimpleDateDTO(data.get('issued', {}).get('date-parts', [])[0]),
            }

            # match type with supported DTO
            match ref_type:
                case 'proceedings-article':
                    return ConferencePaperDTO()
                case 'journal-article':
                    return JournalArticleDTO()
                case 'dissertation':
                    return ThesisDTO()
                case 'book-chapter':
                    return BookSectionDTO()
                case _:
                    CrossrefDTO()

# CROSSREF_TO_ZOTERO = {
#     "journal-article": "journalArticle",
#     "proceedings-article": "conferencePaper",
#     "book": "book",
#     "book-chapter": "bookSection",
#     "dissertation": "thesis"
# }