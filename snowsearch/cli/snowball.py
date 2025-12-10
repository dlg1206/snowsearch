from typing import List, Tuple

from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
from util.config_parser import Config
from util.logger import logger
from util.timer import Timer
from util.verify import validate_all_papers_found

"""
File: snowball.py

Description: Perform rounds of snowballing from the cli

@author Derek Garcia
"""


async def snowball(db: PaperDatabase, openalex_client: OpenAlexClient, grobid_worker: GrobidWorker,
                   n_rounds: int, seed_papers: List[PaperDTO],
                   nl_query: str = None,
                   papers_per_round: int = None,
                   min_similarity_score: float = None) -> Tuple[int, int]:
    """
    Perform rounds of snowballing to search for reverent related papers

    :param db: Database to fetch details and save paper data to
    :param openalex_client: Client to make requests to OpenAlex
    :param grobid_worker: Client to make requests to Grobid server
    :param n_rounds: Number of rounds of snowballing to perform
    :param seed_papers: List of papers to start snowballing with
    :param nl_query: Natural language search query to match relevant papers (Default: None)
    :param papers_per_round: Max number of papers to attempt to process per round (Default: All papers)
    :param min_similarity_score: Minimum similarity score for snowball cutoff (Default: None)
    :return: Number of new papers processed and citation metadata found successfully
    """

    def __get_papers_for_round() -> List[PaperDTO]:
        """
        Util method to fetch relevant papers for the round
        :return: List of papers
        """
        # sort by best match
        if nl_query:
            return db.search_papers_by_nl_query(nl_query,
                                                unprocessed=True,
                                                only_open_access=True,
                                                paper_limit=papers_per_round,
                                                min_score=min_similarity_score)
        # else get unprocessed papers
        else:
            return db.get_unprocessed_papers(papers_per_round)

    # perform n rounds of snowballing
    processed_papers = 0
    new_citations = 0
    round_papers = seed_papers  # start round with seed papers
    for r in range(0, n_rounds):
        # exit early if no papers to snowball with
        if not round_papers:
            logger.warn("Did not find more valid papers to continue snowballing; exiting early")
            break

        logger.info(f"Starting Snowball Round {r + 1} / {n_rounds}")
        timer = Timer()

        # enrich papers
        processed_papers += await grobid_worker.enrich_papers(db, round_papers)

        # fetch metadata for new citations
        citations = []
        for p in round_papers:
            citations += db.get_citations(p.id, True)
        # todo - config to skip title search
        new_citations += await openalex_client.fetch_and_save_citation_metadata(db, citations)

        # Repeat
        logger.info(f"Snowball Round {r + 1} completed in {timer.format_time()}s")
        round_papers = __get_papers_for_round()

    # return stats
    return processed_papers, new_citations


async def run_snowball(db: PaperDatabase, config: Config,
                       nl_query: str = None,
                       papers_per_round: int = None,
                       seed_paper_titles: List[str] = None) -> None:
    """
    Run the snowball process

    :param db: Database to fetch details and save paper data to
    :param config: Config details for performing the search
    :param nl_query: Natural language search query to match relevant papers (Default: None)
    :param papers_per_round: Max number of papers to attempt to process per round (Default: All papers)
    :param seed_paper_titles: List of paper titles to start snowballing with (Default: None)
    """

    # init OpenAlex client
    openalex_client = OpenAlexClient(None, config.openalex.email)  # todo - make model optional

    # init grobid client
    grobid_worker = GrobidWorker(
        config.grobid.max_grobid_requests,
        config.grobid.max_concurrent_downloads,
        config.grobid.max_local_pdfs,
        config.grobid.client_params
    )

    # fetch seed papers
    if seed_paper_titles:
        seed_papers = db.get_papers(seed_paper_titles)
        for missing_title in validate_all_papers_found(seed_paper_titles, seed_papers):
            logger.warn(f"Could not find seed paper '{missing_title}' in the database")

    else:
        seed_papers = db.search_papers_by_nl_query(nl_query,
                                                   unprocessed=True,
                                                   only_open_access=True,
                                                   paper_limit=papers_per_round,
                                                   min_score=config.snowball.min_similarity_score)

    # perform N rounds of snowballing
    timer = Timer()
    logger.info(f"Starting {config.snowball.rounds} rounds of snowballing")
    logger.info(f"Starting snowball with {len(seed_paper_titles)} seed papers")
    processed_papers, new_citations = await snowball(db, openalex_client, grobid_worker,
                                                     config.snowball.rounds, seed_papers,
                                                     nl_query=nl_query,
                                                     papers_per_round=papers_per_round,
                                                     min_similarity_score=config.snowball.min_similarity_score)

    # log stats
    logger.info(f"Snowballing complete in {timer.format_time()}s")
    logger.info(f"Processed {processed_papers} new papers")
    logger.info(f"Fetched details for {new_citations} new citations")
