import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import List

from ai.ollama import OllamaClient
from ai.openai import OpenAIClient, OPENAI_API_KEY_ENV
from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from rank.abstract_ranker import AbstractRanker
from util.config_parser import Config, RankingConfigDTO
from util.logger import logger

"""
File: rank.py

Description: Use an LLM to rank papers based on a search term

@author Derek Garcia
"""


def _print_results(nl_query: str, ranked_papers: List[PaperDTO]) -> None:
    """
    Pretty print the ranked results

    :param nl_query: Original search used to rank the abstracts
    :param ranked_papers: List of ranked papers
    """
    print(f"\nOriginal search: {nl_query.strip()}")
    for rank, paper in enumerate(ranked_papers, start=1):
        print(f"\n\t{rank}: '{paper.id}'")
        print(f"\turl: {paper.pdf_url}")
        print(f"==Abstract==")
        # pretty print abstract
        print(paper.format_abstract())


async def _rank(rank_config: RankingConfigDTO, nl_query: str, papers: List[PaperDTO]) -> List[PaperDTO]:
    """
    Use an LLM to rank papers

    :param rank_config: Config for ranking LLMs
    :param nl_query: Search query to compare abstracts to
    :param papers: List of papers to rank
    :return: Ordered list of papers that best match the nl_query
    """
    # init ranker client
    abstract_model = OpenAIClient(rank_config.agent_config.model_name) if os.getenv(OPENAI_API_KEY_ENV) \
        else OllamaClient(**asdict(rank_config.agent_config))
    ranker = AbstractRanker(abstract_model,
                            rank_config.context_window,
                            rank_config.tokens_per_word)

    # rank papers
    return await ranker.rank_paper_abstracts(nl_query, papers)


async def run_rank(db: PaperDatabase, config: Config, nl_query: str,
                   paper_limit: int = None,
                   json_output: str = None,
                   min_similarity_score: float = None,
                   paper_titles_to_rank: List[str] = None) -> None:
    """
    Use an LLM to rank papers based on their relevance to the query.
    If papers are not provided, first find


    :param db: Database to store paper results in
    :param config: Config details for performing the search
    :param nl_query: Natural langauge search query to match papers to
    :param paper_limit: Max number of papers to rank that overrides the config (Default: None)
    :param json_output: Path to save results to instead of printing to stdout (Default: None)
    :param min_similarity_score: Minimum similarity score for filter cutoff (Default: None)
    :param paper_titles_to_rank: List of papers to rank (Default: None)
    """
    # get papers from database if provided
    if paper_titles_to_rank:
        paper_titles = paper_titles_to_rank
    else:
        paper_titles = [t for t, _, _ in
                        db.search_papers_by_nl_query(nl_query,
                                                     require_abstract=True,
                                                     paper_limit=paper_limit if paper_limit else config.ranking.top_n_papers,
                                                     min_score=min_similarity_score if min_similarity_score else config.ranking.min_abstract_score,
                                                     order_by_abstract=True)]
    papers_to_rank = db.get_papers(paper_titles)
    if paper_titles_to_rank and len(papers_to_rank) != len(paper_titles_to_rank):
        # warn if couldn't find requested paper
        requested_papers = set(paper_titles_to_rank)
        found_papers = {p.id for p in papers_to_rank}
        for missing_title in requested_papers - found_papers:
            logger.warn(f"Could not find seed paper '{missing_title}' in the database")

    # error if no papers
    if not papers_to_rank:
        raise Exception("No papers to rank")

    # rank and print results
    ranked_papers = await _rank(config.ranking, nl_query, papers_to_rank)
    if json_output:
        # format abstracts
        results = {}
        for rank, paper in enumerate(ranked_papers, start=1):
            results[rank] = {
                'title': paper.id,
                'url': paper.pdf_url,
                'abstract': paper.abstract_text
            }
        # write json
        with open(json_output if json_output.endswith('.json') else f"{json_output}.json", 'w') as f:
            data = {
                'generated': datetime.now().isoformat(),
                'original_search': nl_query,
                'rankings': results
            }
            json.dump(data, f, indent=4)
        logger.info(f"Results saved to '{json_output}'")
    else:
        # pretty print results
        _print_results(nl_query, ranked_papers)
