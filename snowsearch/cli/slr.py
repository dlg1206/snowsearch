import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import List

from ai.ollama import OllamaClient
from ai.openai import OpenAIClient, OPENAI_API_KEY_ENV
from db.paper_database import PaperDatabase
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
from openalex.dto import PaperDTO
from rank.abstract_ranker import AbstractRanker
from rank.dto import AbstractDTO
from util.config_parser import Config
from util.logger import logger
from util.timer import Timer

"""
File: slr.py
Description: Orchestrate entire strategic literature review pipeline

@author Derek Garcia
"""


async def _snowball(db: PaperDatabase,
                    openalex_client: OpenAlexClient,
                    grobid_worker: GrobidWorker,
                    nl_query: str,
                    n_rounds: int,
                    papers_per_round: int,
                    seed_papers: List[PaperDTO] = None,
                    min_similarity_score: float = None) -> None:
    """
    Perform rounds of snowballing to search for reverent related papers

    :param db: Databases to fetch details and save paper data to
    :param openalex_client: Client to make requests to OpenAlex
    :param grobid_worker: Client to make requests to Grobid server
    :param nl_query: Natural langauge search query to match relevant papers
    :param n_rounds: Number of rounds of snowballing to perform
    :param papers_per_round: Max number of papers to process per round
    :param seed_papers: List of papers to start snowballing with (Default: None)
    :param min_similarity_score: Minimum similarity score for snowball cutoff (Default: None)
    """

    def __get_papers_for_round() -> List[PaperDTO]:
        """
        Util method to fetch relevant papers for the round
        :return: List of papers
        """
        titles = [t for t, _, _ in
                  db.search_papers_by_nl_query(nl_query,
                                               True, True, paper_limit=papers_per_round,
                                               min_score=min_similarity_score)]
        return db.get_papers(titles) if titles else []

    # perform n rounds of snowballing
    for r in range(0, n_rounds):
        # get papers for round
        paper_dtos = db.get_papers([p.id for p in seed_papers]) if seed_papers else __get_papers_for_round()
        if not paper_dtos:
            # at least 1 round, exit early
            if r:
                logger.warn("Did not find more valid papers to continue snowballing; exiting early")
                break
            else:
                # no rounds, error
                # todo custom error
                raise Exception("No papers to snowball with")

        logger.info(f"Starting Snowball Round {r + 1} / {n_rounds}")
        timer = Timer()

        # enrich papers
        await grobid_worker.enrich_papers(db, paper_dtos)

        # fetch metadata for new citations
        citations = []
        for p in paper_dtos:
            citations += db.get_citations(p.id, True)
        await openalex_client.fetch_and_save_citation_metadata(db, citations)  # todo - config to skip title search

        # Repeat
        logger.info(f"Snowball Round {r + 1} completed in {timer.format_time()}s")


def _format_results(db: PaperDatabase, original_search: str, ranked_abstracts: List[AbstractDTO]) -> None:
    """
    Pretty print the ranked results

    :param db: Database to fetch paper details from
    :param original_search: Original search used to rank the abstracts
    :param ranked_abstracts: List of ranked abstracts
    """
    print(f"\nOriginal search: {original_search.strip()}")
    for rank in range(len(ranked_abstracts)):
        p_full = db.get_paper(ranked_abstracts[rank].paper_title)
        print(f"\n\t{rank + 1}: '{p_full.id}'")
        print(f"\turl: {p_full.pdf_url}")
        print(f"==Abstract==")
        # pretty print abstract
        abstract_text = p_full.abstract_text.split()
        line = "\t"
        while abstract_text:
            line += f"{abstract_text.pop(0)} "
            # print if exceed char limit or nothing left to add
            if len(line) > 100 or not abstract_text:
                print(line.strip())
                line = "\t"
        rank += 1


async def slr(db: PaperDatabase,
              config: Config,
              nl_query: str,
              oa_query: str = None,
              json_output_path: str = None) -> None:
    """
    Perform a full literature search

    :param db: Database to store paper results in
    :param config: Config details for performing the search
    :param nl_query: Natural langauge search query to match papers to
    :param oa_query: Elasticsearch-like query to use for search OpenAlex instead of generating one (Default: None)
    :param json_output_path: Path to save results to instead of printing to stdout (Default: None)
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
    # init grobid clien
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
    await _snowball(db, openalex_client, grobid_worker, nl_query, config.snowball.rounds,
                    config.snowball.papers_per_round, min_similarity_score=config.snowball.min_similarity_score)
    logger.info(f"Snowballing complete in {timer.format_time()}s")

    # after snowballing, get top N papers that best match the prompt and rank them
    ranking_seed_titles = [t for t, _, _ in
                           db.search_papers_by_nl_query(nl_query,
                                                        require_abstract=True,
                                                        paper_limit=config.ranking.top_n_papers,
                                                        min_score=config.ranking.min_abstract_score)]
    papers = db.get_papers(ranking_seed_titles)
    if not ranking_seed_titles:
        raise Exception("No papers to rank")
    ranked_abstracts = await ranker.rank_abstracts(nl_query, [AbstractDTO(r.id, r.abstract_text) for r in papers])
    if json_output_path:
        # format abstracts
        results = {}
        for rank in range(len(ranked_abstracts)):
            p_full = db.get_paper(ranked_abstracts[rank].paper_title)
            results[rank + 1] = {
                'title': p_full.id,
                'url': p_full.pdf_url,
                'abstract': p_full.abstract_text
            }
        # write json
        if not json_output_path.endswith('.json'):
            json_output_path += '.json'
        with open(json_output_path, 'w') as f:
            data = {
                'generated': datetime.now().isoformat(),
                'original_search': nl_query,
                'rankings': results
            }
            json.dump(data, f, indent=4)
        logger.info(f"Results saved to '{json_output_path}'")
    else:
        # pretty print results
        _format_results(db, nl_query, ranked_abstracts)
