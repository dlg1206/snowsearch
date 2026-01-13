"""
File: output.py

Description: Format papers to stdout and json

@author Derek Garcia
"""

import json
import re
from datetime import datetime
from re import Match
from typing import List

from tabulate import tabulate

from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO

RED = "\033[31m"
RESET = "\033[0m"


def _highlight(m: Match) -> str:
    """
    Highlight match in red

    :param m: Match
    :return: Highlighted match
    """
    return f"{RED}{m.group(0)}{RESET}"


def write_papers_to_json(db: PaperDatabase, json_output: str, papers: List[PaperDTO],
                         model_used: str = None, nl_query: str = None) -> str:
    """
    Write paper details to json

    :param db: Paper database to get additional papers details
    :param json_output: Path to json output file
    :param papers: Order list of papers to write to file
    :param model_used: Model used to rank the papers (Default: None)
    :param nl_query: Natural language query to score against title and abstract (Default: None)
    :return: Path to output path
    """
    # format abstracts
    results = {}
    for rank, paper in enumerate(papers, start=1):
        # calc match if nl_query provided
        if nl_query:
            title_score, abstract_score = db.get_embedding_match_score(paper.id, nl_query)
        else:
            title_score, abstract_score = None, None

        # save data
        results[rank] = {
            'title': paper.id,
            'title_match': title_score,
            'abstract_match': abstract_score,
            'doi': paper.doi,
            'url': paper.pdf_url,
            'openalex': paper.openalex_url,
            'citations': len(db.get_citations(paper.id)),
            'abstract': paper.abstract_text
        }
    data = {
        'generated': datetime.now().isoformat(),
        'model': model_used,
        'original_search': nl_query,
        'rankings': results
    }

    # remove values if not used
    if not model_used:
        data.pop('model')
    if not nl_query:
        data.pop('original_search')

    # write json
    json_output = json_output if json_output.endswith('.json') else f"{json_output}.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    return json_output


def print_ranked_papers(db: PaperDatabase, papers: List[PaperDTO],
                        include_abstract: bool = False,
                        nl_query: str = None,
                        exact_match: str = None) -> None:
    """
    Print paper details to stdout

    :param db: Paper database to get additional papers details
    :param papers: List of papers and their title and abstract match
    :param include_abstract: Include abstract in the output table
    :param nl_query: Natural language query to score against title and abstract if present (Default: None)
    :param exact_match: Exact match in title to highlight (default: None)
    """

    headers = ['#', 'Title', 'Open Access', 'DOI', 'URL', 'OpenAlex', 'Citations']

    # add score if requested
    if nl_query:
        headers.insert(2, 'Title Match')
        headers.insert(3, 'Abstract Match')

    # add abstract if requested
    if include_abstract:
        headers.append('Abstract')

    table = []
    for r, p in enumerate(papers, start=1):
        # highlight match if match term provided
        title = re.sub(rf'{exact_match}', _highlight, p.id, flags=re.IGNORECASE) if exact_match else p.id
        row = [r,
               title,
               p.is_open_access,
               p.doi,
               p.pdf_url,
               p.openalex_url,
               len(db.get_citations(p.id))
               ]

        # add score if requested
        if nl_query:
            title_score, abstract_score = db.get_embedding_match_score(p.id, nl_query)
            row.insert(2, f"{100 * title_score:.01f}%" if title_score else None)
            row.insert(3, f"{100 * abstract_score:.01f}%" if abstract_score else None)

        # add abstract if requested
        if include_abstract:
            row.append(p.format_abstract())

        table.append(row)

    # output table
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))
