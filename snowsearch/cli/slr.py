import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import List

from ai.ollama import OllamaClient
from ai.openai import OpenAIClient, OPENAI_API_KEY_ENV
from cli.snowball import snowball
from db.paper_database import PaperDatabase
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
from dto.paper_dto import PaperDTO
from rank.abstract_ranker import AbstractRanker
from util.config_parser import Config
from util.logger import logger
from util.timer import Timer

"""
File: slr.py
Description: Orchestrate entire strategic literature review pipeline

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
        print(f"\n\t{rank + 1}: '{paper.id}'")
        print(f"\turl: {paper.pdf_url}")
        print(f"==Abstract==")
        # pretty print abstract
        print(paper.format_abstract())


async def run_slr(db: PaperDatabase,
                  config: Config,
                  nl_query: str,
                  oa_query: str = None,
                  skip_paper_ranking: bool = False,
                  json_output: str = None) -> None:
    """
    Perform a full literature search

    :param db: Database to store paper results in
    :param config: Config details for performing the search
    :param nl_query: Natural langauge search query to match papers to
    :param oa_query: Elasticsearch-like query to use for search OpenAlex instead of generating one (Default: None)
    :param skip_paper_ranking: Skip ranking the most relevant papers using an LLM after snowballing (Default: False)
    :param json_output: Path to save results to instead of printing to stdout (Default: None)
    """
    # init OpenAlex client
    oa_query_model = OpenAIClient(config.query_generation.model_name) if os.getenv(
        OPENAI_API_KEY_ENV) else OllamaClient(**asdict(config.query_generation))
    openalex_client = OpenAlexClient(oa_query_model, config.openalex.email)
    # init ranker client
    abstract_model = OpenAIClient(config.ranking.agent_config.model_name) if os.getenv(
        OPENAI_API_KEY_ENV) else OllamaClient(**asdict(config.ranking.agent_config))
    ranker = AbstractRanker(abstract_model,
                            config.ranking.context_window,
                            config.ranking.tokens_per_word)
    # init grobid client
    grobid_worker = GrobidWorker(
        config.grobid.max_grobid_requests,
        config.grobid.max_concurrent_downloads,
        config.grobid.max_local_pdfs,
        config.grobid.client_params
    )

    # init run
    run_id = db.start_run()

    # generate query if none provided
    if oa_query:
        logger.info(f"Using provided OpenAlex | {oa_query}")
        db.insert_openalex_query(run_id, None, nl_query, oa_query)
    else:
        oa_query = await openalex_client.generate_openalex_query(nl_query)
        db.insert_openalex_query(run_id, openalex_client.model, nl_query, oa_query)

    # fetch openalex metadata from papers found by the query
    await openalex_client.search_and_save_metadata(run_id, db, oa_query)

    # perform N rounds of snowballing
    timer = Timer()
    logger.info(f"Starting {config.snowball.rounds} rounds of snowballing")
    await snowball(db, openalex_client, grobid_worker, config.snowball.rounds,
                   nl_query=nl_query,
                   papers_per_round=config.snowball.papers_per_round,
                   min_similarity_score=config.snowball.min_similarity_score)
    logger.info(f"Snowballing complete in {timer.format_time()}s")

    # exit early if not ranking
    if skip_paper_ranking:
        logger.warn("Skipping paper ranking")
        return

    # after snowballing, get top N papers that best match the prompt and rank them
    ranking_seed_titles = [t for t, _, _ in
                           db.search_papers_by_nl_query(nl_query,
                                                        require_abstract=True,
                                                        paper_limit=config.ranking.top_n_papers,
                                                        min_score=config.ranking.min_abstract_score)]
    papers = db.get_papers(ranking_seed_titles)
    if not ranking_seed_titles:
        raise Exception("No papers to rank")
    ranked_papers = await ranker.rank_paper_abstracts(nl_query, papers)
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
