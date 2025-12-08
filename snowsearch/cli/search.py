from typing import List, Tuple

from tabulate import tabulate

from db.paper_database import PaperDatabase

"""
File: search.py

Description: Prebuilt commands to search the neo4j database

@author Derek Garcia
"""


def _print_papers(db: PaperDatabase, paper_matches: List[Tuple[str, float, float]]) -> None:
    """
    Print matching papers to stdout

    :param db: Paper database to get additional details from
    :param paper_matches: List of tuples of matching papers and similarity scores to titles and abstracts
    """
    headers = ['Title', 'Title Match', 'Abstract Match', 'DOI', 'URL', 'OpenAlex', 'Citations']
    table = []
    paper_map = {p.id: p for p in db.get_papers([t for t, _, _ in paper_matches])}
    for title, title_score, abstract_score in paper_matches:
        table.append([title,
                      f"{100 * title_score:.01f}%",
                      f"{100 * abstract_score:.01f}%" if abstract_score else None,
                      paper_map.get(title).doi,
                      paper_map.get(title).pdf_url,
                      paper_map.get(title).openalex_url,
                      len(db.get_citations(title))
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
    :param nl_query: Natural langauge search query to match relevant papers (Default: None)
    :param paper_limit: Number of papers to display (Default: All)
    :param exact_match: Return papers that contain an exact, case-insensitive, match to the nl query (Default: False)
    :param only_open_access: Return papers that are open access (Default: False)
    :param only_processed: Return papers that have been processed with Grobid (Default: False)
    :param min_similarity_score: Minimum similarity score for filter cutoff (Default: None)
    :param order_by_abstract: Return search order by abstract match then title match (Default: False)
    """

    papers = db.search_papers_by_nl_query(nl_query,
                                          require_abstract=only_processed,  # if abstract then processed by grobid
                                          only_open_access=only_open_access,
                                          paper_limit=paper_limit,
                                          min_score=min_similarity_score,
                                          order_by_abstract=order_by_abstract)
    _print_papers(db, papers)
