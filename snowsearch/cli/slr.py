import os
from dataclasses import asdict
from typing import List

from ai.ollama import OllamaClient
from ai.openai import OpenAIClient, OPENAI_API_KEY_ENV
from cli.rank import rank_papers
from cli.snowball import snowball
from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
from util.config_parser import Config
from util.logger import logger
from util.output import write_ranked_papers_to_json, print_ranked_papers
from util.timer import Timer

"""
File: slr.py
Description: Orchestrate entire strategic literature review pipeline

@author Derek Garcia
"""


async def run_slr(db: PaperDatabase, config: Config, nl_query: str,
                  oa_query: str = None,
                  skip_paper_ranking: bool = False,
                  json_output: str = None) -> None:
    """
    Perform a full literature search

    :param db: Database to store paper results in
    :param config: Config details for performing the search
    :param nl_query: Natural language search query to match papers to
    :param oa_query: Elasticsearch-like query to use for search OpenAlex instead of generating one (Default: None)
    :param skip_paper_ranking: Skip ranking the most relevant papers using an LLM after snowballing (Default: False)
    :param json_output: Path to save results to instead of printing to stdout (Default: None)
    """
    # init OpenAlex client
    oa_query_model = OpenAIClient(config.query_generation.model_name) if os.getenv(
        OPENAI_API_KEY_ENV) else OllamaClient(**asdict(config.query_generation))
    openalex_client = OpenAlexClient(oa_query_model, config.openalex.email)

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
    seed_papers = db.search_papers_by_nl_query(nl_query,
                                               unprocessed=True,
                                               only_open_access=True,
                                               paper_limit=config.snowball.papers_per_round,
                                               min_score=config.snowball.min_similarity_score)
    timer = Timer()
    logger.info(f"Starting {config.snowball.rounds} rounds of snowballing")
    await snowball(db, openalex_client, grobid_worker, config.snowball.rounds, seed_papers,
                   nl_query=nl_query,
                   papers_per_round=config.snowball.papers_per_round,
                   min_similarity_score=config.snowball.min_similarity_score)
    logger.info(f"Snowballing complete in {timer.format_time()}s")

    # exit early if not ranking
    if skip_paper_ranking:
        logger.warn("Skipping paper ranking")
        return

    # after snowballing, get top N papers that best match the prompt and rank them
    papers = db.search_papers_by_nl_query(nl_query,
                                          require_abstract=True,
                                          paper_limit=config.ranking.top_n_papers,
                                          min_score=config.ranking.min_abstract_score)

    if not papers:
        raise Exception("No papers to rank")
    # rank and print output
    ranked_papers = await rank_papers(config.ranking, nl_query, papers)
    if json_output:
        model = f"{config.ranking.agent_config.model_name}:{config.ranking.agent_config.model_tag}"
        json_output = write_ranked_papers_to_json(db, json_output, model, nl_query, ranked_papers)
        logger.info(f"Results saved to '{json_output}'")
    else:
        # pretty print results
        print_ranked_papers(db, ranked_papers, include_abstract=True, nl_query=nl_query)
