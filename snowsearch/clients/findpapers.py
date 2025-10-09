import json
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Dict, List

import findpapers

"""
File: findpapers.py

Description: Client for interacting with findpapers api

@author Derek Garcia
"""


@dataclass
class PaperDTO:
    """
    Util DTO for tracking paper details
    """
    title: str
    abstract: str
    doi: str


class FindpapersClient:
    def __init__(self, config: Dict[str, str | int | List[str] | None]):
        """
        Create new findpapers client for searching for papers

        :param config: findpapers config details
        """
        self._config = config

    def search(self, query: str) -> List[PaperDTO]:
        """
        Search for matching papers using findpapers query

        query lang: https://github.com/jonatasgrosman/findpapers?tab=readme-ov-file#search-query-construction

        :param query: findpapers style query to search for papers from
        :return: List of relevant papers
        """
        with NamedTemporaryFile(prefix="findpapers", suffix=".json") as fp_tmp:
            # todo - supress output?
            findpapers.search(fp_tmp.name, query, **self._config)
            # load results
            with open(fp_tmp.name, 'r') as f:
                results = json.load(f)
        # extract relevant details
        return [PaperDTO(r['title'], r['abstract'], r['doi']) for r in results['papers']]
