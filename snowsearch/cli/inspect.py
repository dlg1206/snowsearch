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

    # format abstract to fit in table
    abstract_text = paper.abstract_text.split()
    line = "\t"
    abstract_text_formatted = ""
    while abstract_text:
        line += f"{abstract_text.pop(0)} "
        # print if exceed char limit or nothing left to add
        if len(line) > 100 or not abstract_text:
            abstract_text_formatted += f"{line.strip()}\n"
            line = "\t"

    # print table
    headers = ['Title', 'Abstract', 'DOI', 'URL', 'OpenAlex', 'Citations']
    table = [[paper.id, abstract_text_formatted, paper.doi, paper.pdf_url, paper.openalex_url,
              len(db.get_citations(paper.id))]]
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))
