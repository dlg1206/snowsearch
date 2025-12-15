"""
File: rank.py

Description: Use an LLM to rank papers based on a search term

@author Derek Garcia
"""

import os
from dataclasses import asdict
from typing import List

from ai.ollama import OllamaClient
from ai.openai import OpenAIClient, OPENAI_API_KEY_ENV
from config.parser import RankingConfigDTO
from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from rank.abstract_ranker import AbstractRanker
from util.logger import logger
from util.output import write_papers_to_json, print_ranked_papers
from util.verify import validate_all_papers_found


async def rank_papers(rank_config: RankingConfigDTO, nl_query: str, papers: List[PaperDTO]) -> List[PaperDTO]:
    """
    Use an LLM to rank papers

    :param rank_config: Config for ranking LLMs
    :param nl_query: Search query to compare abstracts to
    :param papers: List of papers to rank
    :return: Ordered list of papers that best match the nl_query
    """
    # init ranker client
    abstract_model = OpenAIClient(rank_config.agent_config.model_name, rank_config.agent_config.context_window) \
        if os.getenv(OPENAI_API_KEY_ENV) else OllamaClient(**asdict(rank_config.agent_config))
    ranker = AbstractRanker(abstract_model, rank_config.tokens_per_word)

    # rank papers
    return await ranker.rank_paper_abstracts(nl_query, papers)


async def run_rank(db: PaperDatabase, rank_config: RankingConfigDTO, nl_query: str,
                   paper_limit: int,
                   min_similarity_score: float,
                   json_output: str = None,
                   paper_titles_to_rank: List[str] = None) -> None:
    """
    Use an LLM to rank papers based on their relevance to the query.

    :param db: Database to store paper results in
    :param rank_config: Ranking config details for performing the search
    :param nl_query: Natural language search query to match papers to
    :param paper_limit: Max number of papers to rank that overrides the config (Default: None)
    :param json_output: Path to save results to instead of printing to stdout (Default: None)
    :param min_similarity_score: Minimum similarity score for filter cutoff (Default: None)
    :param paper_titles_to_rank: List of papers to rank (Default: None)
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
        raise ValueError("No papers to rank")

    # rank and print results
    ranked_papers = await rank_papers(rank_config, nl_query, papers_to_rank)
    if json_output:
        model = f"{rank_config.agent_config.model_name}:{rank_config.agent_config.model_tag}"
        json_output = write_papers_to_json(db, json_output, ranked_papers, model_used=model, nl_query=nl_query)
        logger.info(f"Results saved to '{json_output}'")
    else:
        # pretty print results
        print_ranked_papers(db, ranked_papers, include_abstract=True, nl_query=nl_query)
