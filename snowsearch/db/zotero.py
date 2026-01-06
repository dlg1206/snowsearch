"""
File: zotero.py

Description: Client for interacting with a zotero library

@author Derek Garcia
"""
import os
from enum import Enum
from typing import List, Dict, Any, Set

from pyzotero.zotero import Zotero

from dto.paper_dto import PaperDTO
from util.logger import logger


class LibraryType(Enum):
    USER = "user"
    GROUP = "group"


def _fetch_existing_doi(items: List[Dict[str, Any]]) -> Set[str]:
    existing_doi = set()
    for item in items:
        doi = item['data'].get('DOI')
        if doi:
            existing_doi.add(doi.lower())
    return existing_doi


def zotero_upload(library_id: str,
                  library_type: LibraryType,
                  papers: List[PaperDTO],
                  collection_key: str = None) -> None:
    # todo - check for api key
    zot_client = Zotero(library_id, library_type.value, os.getenv('ZOTERO_API_KEY'))

    if collection_key:
        logger.info(f"Fetching details from collection '{collection_key}'")
    zot_items = zot_client.everything(
        zot_client.collection_items(collection_key) if collection_key else zot_client.items())

    existing_doi = _fetch_existing_doi(zot_items)
