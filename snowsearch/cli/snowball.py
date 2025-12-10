from typing import List, Tuple

from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
from util.config_parser import Config
from util.logger import logger
from util.timer import Timer

"""
File: snowball.py

Description: Perform rounds of snowballing from the cli

@author Derek Garcia
"""


async def snowball(db: PaperDatabase, openalex_client: OpenAlexClient, grobid_worker: GrobidWorker, n_rounds: int,
                   nl_query: str = None,
                   papers_per_round: int = None,
                   seed_papers: List[str] = None,
                   min_similarity_score: float = None) -> Tuple[int, int]:
    """
    Perform rounds of snowballing to search for reverent related papers

    :param db: Database to fetch details and save paper data to
    :param openalex_client: Client to make requests to OpenAlex
    :param grobid_worker: Client to make requests to Grobid server
    :param n_rounds: Number of rounds of snowballing to perform
    :param nl_query: Natural language search query to match relevant papers (Default: None)
    :param papers_per_round: Max number of papers to attempt to process per round (Default: All papers)
    :param seed_papers: List of paper titles to start snowballing with (Default: None)
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
    for r in range(0, n_rounds):
        # get papers for round
        paper_dtos = db.get_papers(seed_papers) if seed_papers else __get_papers_for_round()
        if not paper_dtos:
            # at least 1 round, exit early
            if r:
                logger.warn("Did not find more valid papers to continue snowballing; exiting early")
                break
            else:
                # no rounds, error
                # todo custom error
                raise Exception("No papers to snowball with")

        # warn if couldn't find seed paper
        if seed_papers and len(paper_dtos) != len(seed_papers) and not r:
            seed_papers_set = set(seed_papers)
            round_one_titles = {p.id for p in paper_dtos}
            for missing_title in seed_papers_set - round_one_titles:
                logger.warn(f"Could not find seed paper '{missing_title}' in the database")

        logger.info(f"Starting Snowball Round {r + 1} / {n_rounds}")
        timer = Timer()

        # enrich papers
        processed_papers += await grobid_worker.enrich_papers(db, paper_dtos)

        # fetch metadata for new citations
        citations = []
        for p in paper_dtos:
            citations += db.get_citations(p.id, True)
        # todo - config to skip title search
        new_citations += await openalex_client.fetch_and_save_citation_metadata(db, citations)

        # Repeat
        logger.info(f"Snowball Round {r + 1} completed in {timer.format_time()}s")
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

    # perform N rounds of snowballing
    timer = Timer()
    logger.info(f"Starting {config.snowball.rounds} rounds of snowballing")
    if seed_paper_titles:
        logger.info(f"Starting snowball with {len(seed_paper_titles)} seed papers")
    processed_papers, new_citations = await snowball(db, openalex_client, grobid_worker, config.snowball.rounds,
                                                     nl_query=nl_query,
                                                     papers_per_round=papers_per_round,
                                                     seed_papers=seed_paper_titles,
                                                     min_similarity_score=config.snowball.min_similarity_score)
    # log stats
    logger.info(f"Snowballing complete in {timer.format_time()}s")
    logger.info(f"Processed {processed_papers} new papers")
    logger.info(f"Fetched details for {new_citations} new citations")
