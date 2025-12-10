import re
from typing import List, Tuple

from tabulate import tabulate

from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from util.logger import logger

"""
File: search.py

Description: Prebuilt commands to search the neo4j database

@author Derek Garcia
"""

RED = "\033[31m"
RESET = "\033[0m"


def _print_exact_papers(db: PaperDatabase, search_term: str, paper_matches: List[PaperDTO]) -> None:
    """
    Print the exact matches and highlight the match

    :param db: Paper database to get additional details from
    :param paper_matches: List of matching paper DTOs
    """
    highlight = lambda m: f"{RED}{m.group(0)}{RESET}"

    headers = ['#', 'Title', 'DOI', 'URL', 'OpenAlex', 'Citations']
    table = []
    for i, paper in enumerate(paper_matches, start=1):
        table.append([i,
                      re.sub(rf'{search_term}', highlight, paper.id, flags=re.IGNORECASE),
                      paper.doi,
                      paper.pdf_url,
                      paper.openalex_url,
                      len(db.get_citations(paper.id))
                      ])
    # output table
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))


def _print_similar_papers(db: PaperDatabase, paper_matches: List[Tuple[PaperDTO, float, float]]) -> None:
    """
    Print matching papers to stdout

    :param db: Paper database to get additional details from
    :param paper_matches: List of tuples of matching papers and similarity scores to titles and abstracts
    """
    headers = ['#', 'Title', 'Title Match', 'Abstract Match', 'DOI', 'URL', 'OpenAlex', 'Citations']
    table = []
    for i, data in enumerate(paper_matches, start=1):
        paper, title_score, abstract_score = data
        table.append([i,
                      paper,
                      f"{100 * title_score:.01f}%",
                      f"{100 * abstract_score:.01f}%" if abstract_score else None,
                      paper.doi,
                      paper.pdf_url,
                      paper.openalex_url,
                      len(db.get_citations(paper.id))
                      ])

    # output table
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))


def run_search(db: PaperDatabase,
               nl_query: str,
               paper_limit: int = None,
               exact_match: bool = False,
               only_open_access: bool = False,
               only_processed: bool = False,
               min_similarity_score: float = None,
               order_by_abstract: bool = False) -> None:
    """
    Run a search in the neo4j database

    :param db: Database to fetch details and save paper data to
    :param nl_query: Natural language search query to match relevant papers (Default: None)
    :param paper_limit: Number of papers to display (Default: All)
    :param exact_match: Return papers that contain an exact, case-insensitive, match to the nl query (Default: False)
    :param only_open_access: Return papers that are open access (Default: False)
    :param only_processed: Return papers that have been processed with Grobid (Default: False)
    :param min_similarity_score: Minimum similarity score for filter cutoff (Default: None)
    :param order_by_abstract: Return search order by abstract match then title match (Default: False)
    """
    if exact_match:
        papers = db.search_papers_by_title_match(nl_query,
                                                 require_abstract=only_processed,
                                                 # if abstract then processed by grobid
                                                 only_open_access=only_open_access,
                                                 paper_limit=paper_limit)
        if not papers:
            logger.warn(f"Found no paper titles that contain the term '{nl_query}'")
        _print_exact_papers(db, nl_query, papers)
    else:
        papers = db.search_papers_by_nl_query(nl_query,
                                              require_abstract=only_processed,  # if abstract then processed by grobid
                                              only_open_access=only_open_access,
                                              paper_limit=paper_limit,
                                              min_score=min_similarity_score,
                                              order_by_abstract=order_by_abstract)
        if not papers:
            logger.warn(f"Found no paper titles that match the search within the search filters | '{nl_query}'")
        _print_similar_papers(db, papers)
