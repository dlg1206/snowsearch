from typing import List, Tuple

from config.parser import Config
from db.paper_database import PaperDatabase
from dto.paper_dto import PaperDTO
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
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
                   round_quota: int = None,
                   min_similarity_score: float = None,
                   ignore_quota: bool = False,
                   seed_provided: bool = False) -> Tuple[int, int]:
    """
    Perform rounds of snowballing to search for reverent related papers

    :param db: Database to fetch details and save paper data to
    :param openalex_client: Client to make requests to OpenAlex
    :param grobid_worker: Client to make requests to Grobid server
    :param n_rounds: Number of rounds of snowballing to perform
    :param seed_papers: List of papers to start snowballing with
    :param nl_query: Natural language search query to match relevant papers (Default: None)
    :param round_quota: Number of papers to attempt to process per round (Default: All papers)
    :param min_similarity_score: Minimum similarity score for snowball cutoff (Default: None)
    :param ignore_quota: Skip fetching more papers to process if did not meet round quota round (Default: False)
    :param seed_provided: If the provided seed papers were explicitly provided by the user (Default: False)
    :return: Number of new papers processed and citation metadata found successfully
    """

    def __get_papers_for_round(num_papers: int) -> List[PaperDTO]:
        """
        Util method to fetch relevant papers for the round

        :param num_papers: Number of papers to fetch
        :return: List of papers
        """
        # sort by best match
        if nl_query:
            return db.search_papers_by_nl_query(nl_query,
                                                unprocessed=True,
                                                only_open_access=True,
                                                paper_limit=num_papers,
                                                min_score=min_similarity_score)
        # else get unprocessed papers
        else:
            return db.get_unprocessed_papers(num_papers)

    # perform n rounds of snowballing
    processed_papers = 0
    new_citations = 0
    for r in range(0, n_rounds):
        # start round with seed papers, else fetch papers
        round_papers = __get_papers_for_round(round_quota) if r else seed_papers

        # exit early if no papers to snowball with
        if not round_papers:
            logger.warn("Did not find more valid papers to continue snowballing; exiting early")
            break

        logger.info(f"Starting Snowball Round {r + 1} / {n_rounds}")
        timer = Timer()

        # process until reach round quota if not lazy
        remaining_quota = round_quota
        all_round_papers = []

        while True:
            # cache titles
            all_round_papers.extend([p.id for p in round_papers])
            # enrich papers
            n_success = await grobid_worker.enrich_papers(db, round_papers)
            # update metadata
            processed_papers += n_success
            remaining_quota -= n_success
            # break if not retrying
            if ignore_quota:
                if remaining_quota:
                    logger.info("Lazy snowball -- ignoring round quota")
                break
            # quota met or (first round and seed provided)
            if remaining_quota <= 0 or (not r and seed_provided):
                # don't use other papers for seed if user provided the seed
                break

            # else get a new batch to attempt to meet the quota
            logger.warn(f"Did not meet round quota, processing {remaining_quota} additional papers")
            round_papers = __get_papers_for_round(remaining_quota)

            # exit early if no papers to snowball with
            if not round_papers:
                logger.warn("Did not find more valid papers to meet quota; exiting early")
                break

        # fetch metadata for new citations
        citations = set()
        for p in all_round_papers:
            citations.update(db.get_citations(p, True))

        # todo - config to skip title search
        new_citations += await openalex_client.fetch_and_save_paper_metadata(db, list(citations))

        # Repeat
        logger.info(f"Snowball Round {r + 1} completed in {timer.format_time()}s")

    # return stats
    return processed_papers, new_citations


async def run_snowball(db: PaperDatabase, config: Config,
                       nl_query: str = None,
                       round_quota: int = None,
                       seed_paper_titles: List[str] = None,
                       ignore_quota: bool = False) -> None:
    """
    Run the snowball process

    :param db: Database to fetch details and save paper data to
    :param config: Config details for performing the search
    :param nl_query: Natural language search query to match relevant papers (Default: None)
    :param round_quota: Number of papers to attempt to process per round (Default: All papers)
    :param seed_paper_titles: List of paper titles to start snowballing with (Default: None)
    :param ignore_quota: Skip fetching more papers to process if did not meet round quota round (Default: False)
    """

    # init OpenAlex client
    openalex_client = OpenAlexClient(config.openalex.email)
    # init grobid client
    grobid_worker = GrobidWorker(
        config.grobid.max_grobid_requests,
        config.grobid.max_concurrent_downloads,
        config.grobid.max_local_pdfs,
        config.grobid.client_params
    )

    # fetch seed papers
    if seed_paper_titles:
        logger.debug_msg("Using provided papers for seed")
        seed_papers = db.get_papers(seed_paper_titles)
        for missing_title in validate_all_papers_found(seed_paper_titles, seed_papers):
            logger.warn(f"Could not find seed paper '{missing_title}' in the database")

    elif nl_query:
        logger.debug_msg("Using best query match for seed")
        seed_papers = db.search_papers_by_nl_query(nl_query,
                                                   unprocessed=True,
                                                   only_open_access=True,
                                                   paper_limit=round_quota,
                                                   min_score=config.snowball.min_similarity_score)
    else:
        logger.debug_msg("Using default unprocessed papers")
        seed_papers = db.get_unprocessed_papers(round_quota)

    # perform N rounds of snowballing
    timer = Timer()
    logger.info(f"Starting {config.snowball.rounds} rounds of snowballing")
    logger.info(f"Starting snowball with {len(seed_papers)} seed papers")
    processed_papers, new_citations = await snowball(db, openalex_client, grobid_worker,
                                                     config.snowball.rounds, seed_papers,
                                                     nl_query=nl_query,
                                                     round_quota=round_quota,
                                                     min_similarity_score=config.snowball.min_similarity_score,
                                                     ignore_quota=ignore_quota,
                                                     seed_provided=True if seed_paper_titles else False)

    # log stats
    logger.info(f"Snowballing complete in {timer.format_time()}s")
    logger.info(f"Processed {processed_papers} new papers")
    logger.info(f"Fetched details for {new_citations} new citations")
