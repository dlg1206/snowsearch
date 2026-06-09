"""
File: rank.py

Description: Use an LLM to rank papers based on a search term

@author Derek Garcia
"""

from typing import List

from llumpy import AsyncModelClient

from db.paper_database import PaperDatabase
from db.zotero import ZoteroClient
from rank.abstract_ranker import AbstractRanker
from util.logger import logger
from util.output import write_papers_to_json, print_ranked_papers
from util.verify import validate_all_papers_found


async def run_rank(db: PaperDatabase,
                   rank_client: AsyncModelClient,
                   tokens_per_word: float,
                   nl_query: str,
                   paper_limit: int,
                   min_similarity_score: float,
                   json_output: str = None,
                   paper_titles_to_rank: List[str] = None,
                   zotero_client: ZoteroClient = None) -> None:
    """
    Use an LLM to rank papers based on their relevance to the query.

    :param db: Database to store paper results in
    :param rank_client: Client to AI model to use for ranking
    :param tokens_per_word: Estimate of tokens per word for the model
    :param nl_query: Natural language search query to match papers to
    :param paper_limit: Max number of papers to rank that overrides the config (Default: None)
    :param json_output: Path to save results to instead of printing to stdout (Default: None)
    :param min_similarity_score: Minimum similarity score for filter cutoff (Default: None)
    :param paper_titles_to_rank: List of papers to rank (Default: None)
    :param zotero_client: Client to use to upload resulting papers to zotero (Default: None)
    """
    # get papers from database if provided
    if paper_titles_to_rank:
        papers_to_rank = db.get_papers(paper_titles_to_rank)
        for missing_title in validate_all_papers_found(paper_titles_to_rank, papers_to_rank):
            logger.warn(f"Could not find paper '{missing_title}' in the database")
    else:
        papers_to_rank = db.search_papers_by_nl_query(nl_query,
                                                      require_abstract=True,
                                                      paper_limit=paper_limit,
                                                      min_score=min_similarity_score,
                                                      order_by_abstract=True)

    # error if no papers
    if not papers_to_rank:
        logger.warn("No papers to rank, exiting early. . .")

    # rank papers
    ranker = AbstractRanker(rank_client, tokens_per_word)
    ranked_papers = await ranker.rank_paper_abstracts(nl_query, papers_to_rank)

    # handle output
    if json_output:
        json_output = write_papers_to_json(db, json_output, ranked_papers, model_used=rank_client.model,
                                           nl_query=nl_query)
        logger.info(f"Results saved to '{json_output}'")
    else:
        # pretty print results
        print_ranked_papers(db, ranked_papers, include_abstract=True, nl_query=nl_query)

    # upload papers if created
    if zotero_client:
        await zotero_client.upload_papers(ranked_papers)
