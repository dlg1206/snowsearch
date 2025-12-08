from tabulate import tabulate

from db.paper_database import PaperDatabase
from util.logger import logger

"""
File: inspect.py

Description: Get details about a paper printed to stdout

@author Derek Garcia
"""


def run_inspect(db: PaperDatabase, paper_title: str) -> None:
    """
    Print details about a specific paper.

    :param db: Paper database with paper details
    :param paper_title: Title of paper to get details for
    """
    paper = db.get_paper(paper_title)
    # exit early if couldn't find paper
    if not paper:
        logger.warn(f"Could not find paper '{paper_title}'")
        return

    # print table
    headers = ['Title', 'Abstract', 'DOI', 'URL', 'OpenAlex', 'Citations']
    table = [[paper.id, paper.format_abstract(), paper.doi, paper.pdf_url, paper.openalex_url,
              len(db.get_citations(paper.id))]]
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))
