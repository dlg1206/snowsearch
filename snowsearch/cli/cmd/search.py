"""
File: search.py

Description: Prebuilt commands to search the neo4j database

@author Derek Garcia
"""

from typing import List

import loggy

from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from util.output import print_ranked_papers, write_papers_to_json


def run_search(db: PaperDatabase, nl_query: str,
               paper_limit: int = None,
               exact_match: bool = False,
               only_open_access: bool = False,
               only_processed: bool = False,
               min_similarity_score: float = None,
               order_by_abstract: bool = False,
               json_output: str = None) -> List[PaperDTO]:
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
    :param json_output: Path to save results to instead of printing to stdout (Default: None)
    :return: List of resulting papers
    """
    loggy.info("Searching database")

    # get all papers if none provided
    if paper_limit is None:
        paper_limit = db.get_paper_count()

    if exact_match:
        papers = db.search_papers_by_title_match(nl_query,
                                                 # if abstract then processed by grobid
                                                 require_abstract=only_processed,
                                                 only_open_access=only_open_access,
                                                 paper_limit=paper_limit)
        if not papers:
            loggy.warn(f"Found no paper titles that contain the term '{nl_query}'")
        else:
            loggy.info(f"Found {len(papers)} paper titles that contain the term '{nl_query}'")

    else:
        # todo add heartbeat
        papers = db.search_papers_by_nl_query(nl_query,
                                              # if abstract then processed by grobid
                                              require_abstract=only_processed,
                                              only_open_access=only_open_access,
                                              paper_limit=paper_limit,
                                              min_score=min_similarity_score,
                                              order_by_abstract=order_by_abstract)
        if not papers:
            loggy.warn(f"Found no paper titles that match the search within the search filters | '{nl_query}'")

    if json_output:
        json_output = write_papers_to_json(db, json_output, papers, nl_query=nl_query)
        loggy.info(f"Results saved to '{json_output}'")
    else:
        # pretty print results
        params = {'include_abstract': True, f"{'exact_match' if exact_match else 'nl_query'}": nl_query}
        print_ranked_papers(db, papers, **params)

    return papers
