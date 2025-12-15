"""
File: upload.py

Description: Upload local pdf papers to be stored in the database

@author Derek Garcia
"""

from typing import List

from config.parser import Config
from db.paper_database import PaperDatabase
from dto.grobid_dto import GrobidDTO
from dto.paper_dto import PaperDTO
from grobid.exception import GrobidProcessError
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
from util.logger import logger
from util.timer import Timer
from util.verify import validate_file_is_pdf


async def run_upload(db: PaperDatabase, config: Config, paper_pdf_paths: List[str]) -> None:
    """
    Upload local pdf papers to be stored in the database

    :param db: Database to store paper results in
    :param config: Config details for performing the search
    :param paper_pdf_paths: List of paths to pdfs to upload
    """
    # verify files
    valid_paper_files = []
    for p in paper_pdf_paths:
        if not validate_file_is_pdf(p):
            logger.warn(f"'{p}' is not a valid pdf, skipping")
        else:
            logger.debug_msg(f"'{p}' is a valid pdf")
            valid_paper_files.append(p)

    # error if nothing to process
    if not valid_paper_files:
        raise ValueError("No valid pdfs to upload")

    # init grobid client
    grobid_worker = GrobidWorker(
        config.grobid.max_grobid_requests,
        config.grobid.max_concurrent_downloads,
        config.grobid.max_local_pdfs,
        config.grobid.client_params
    )

    # preempt model load
    db.load_embedding_model()

    logger.info(f"Uploading {len(valid_paper_files)} papers")

    timer = Timer()
    papers = []
    num_success = 0
    num_fail_process = 0

    tasks = [grobid_worker.process_paper(p) for p in valid_paper_files]
    # save results as completed
    for future in logger.get_data_queue(tasks, "Processing papers", "papers", is_async=True):
        try:
            result: GrobidDTO = await future
            result.paper.grobid_status = 200
            papers.append(result.paper)
            # add referenced papers, if any
            if result.citations:
                db.insert_citation_paper_batch(result.paper.id, result.citations)

            num_success += 1

        # failed to parse pdf
        except GrobidProcessError as e:
            logger.error_exp(e)
            db.upsert_paper(PaperDTO(e.paper_title, grobid_status=e.status_code, grobid_error_msg=e.error_msg))
            num_fail_process += 1

    # fetch paper metadata
    openalex_client = OpenAlexClient(config.openalex.email)
    n_metadata_found = await openalex_client.fetch_and_save_paper_metadata(db, papers)
    # fetch citation metadata
    citations = set()  # use set to prevent dupe citations
    for p in papers:
        citations.update(db.get_citations(p.id, True))
    n_metadata_found += await openalex_client.fetch_and_save_paper_metadata(db, list(citations))

    # log stats
    logger.info(f"Upload complete in {timer.format_time()}s")
    logger.info(f"Processed {num_success} papers")
    logger.info(f"Fetched details for {n_metadata_found} papers")
