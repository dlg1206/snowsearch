from typing import List

from dto.paper_dto import PaperDTO

"""
File: verify.py

Description: Util verify functions for paper matches

@author Derek Garcia
"""


def validate_all_papers_found(paper_titles: List[str], found_papers: List[PaperDTO]) -> List[str]:
    """
    Verify that all the papers were found in the database search

    :param paper_titles: List of titles attempted to find in the database
    :param found_papers: List of matching papers
    :return: List of titles not found in the database
    """
    return list(set(paper_titles) - {p.id for p in found_papers})
