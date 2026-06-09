"""
File: worker.py
Description: Grobid worker that handles downloading and processing open access papers

@author Derek Garcia
"""

import asyncio
import logging
import os
from asyncio import Semaphore
from datetime import datetime
from os.path import exists
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Dict, Any

import grobid_tei_xml
import loggy
from aiohttp import ClientSession
from grobid_client.grobid_client import GrobidClient
from loggy import Timer

from db.paper_database import PaperDatabase
from download.config import MAX_CONCURRENT_DOWNLOADS, MAX_PDF_COUNT
from download.exception import NoFileDataError, InvalidFileFormatError, PaperDownloadError
from download.pdf import download_pdf
from dto.grobid_dto import GrobidDTO
from dto.paper_dto import PaperDTO
from grobid.config import MAX_GROBID_REQUESTS
from grobid.exception import GrobidProcessError

# Mute grobid client logs
logging.getLogger("grobid_client").setLevel(logging.CRITICAL)  # todo doesn't work


class GrobidWorker:
    """
    Client for interacting with Grobid and processing papers
    """

    def __init__(self,
                 max_grobid_requests: int = MAX_GROBID_REQUESTS,
                 max_concurrent_downloads: int = MAX_CONCURRENT_DOWNLOADS,
                 max_local_pdfs: int = MAX_PDF_COUNT,
                 client_config: Dict[str, Any] = None):
        """
        Create new Grobid worker to download and process urls

        :param max_grobid_requests: Max number of requests allowed to be made to the grobid server at once (Default: 1)
        :param max_concurrent_downloads: Max number of PDFs to allowed to download at the same time (Default: 10)
        :param max_local_pdfs: Max number of PDFs to be downloaded locally at one time (Default: 100)
        :param client_config: Optional Grobid client details (Default: None)
        """
        # init with params if provided
        logger.debug_msg("Attempting to access Grobid server")
        self._grobid_client = GrobidClient(**client_config) if client_config else GrobidClient()
        logger.debug_msg("Connected Successfully")
        # set semaphore limits
        self._grobid_semaphore = Semaphore(max_grobid_requests)
        self._download_semaphore = Semaphore(max_concurrent_downloads)
        self._pdf_file_semaphore = Semaphore(max_local_pdfs)

    async def process_paper(self, pdf_file_path: str, title: str = None) -> GrobidDTO:
        """
        Submit paper to grobid to process

        :param pdf_file_path: Path to pdf to process
        :param title: Optional title of paper (Default: None)
        :return: Grobid DTO with grobid results
        """
        timer = Timer()
        # block to prevent overwhelming grobid server
        async with self._grobid_semaphore:
            logger.debug_msg(f"Processing '{pdf_file_path}'")
            _, status, content = await asyncio.to_thread(self._grobid_client.process_pdf,
                                                         service="processFulltextDocument",
                                                         pdf_file=pdf_file_path,
                                                         generateIDs=None,
                                                         consolidate_header=None,
                                                         consolidate_citations=None,
                                                         include_raw_citations=None,
                                                         include_raw_affiliations=None,
                                                         tei_coordinates=None,
                                                         segment_sentences=None
                                                         )
        # raise error on non-200 response
        if status != 200:
            raise GrobidProcessError(title, status, content)
        # else parse TEI
        doc = grobid_tei_xml.parse_document_xml(content)
        loggy.debug_info(f"Processed '{pdf_file_path}' in {timer.format_time()}s | {title}")
        timestamp = datetime.now()
        citations = [PaperDTO(c.title, doi=c.doi, time_added=timestamp) for c in doc.citations if c.title]
        # use provided title if provided, else use one parsed by grobid
        # todo - doc.header.title null?
        paper = PaperDTO(title if title else doc.header.title,
                         doi=doc.header.doi, abstract_text=doc.abstract,
                         grobid_status=200, time_grobid_processed=timestamp, time_added=timestamp)
        return GrobidDTO(paper, citations)

    async def _process_paper_task(self, session: ClientSession, work_dir: str, title: str, pdf_url: str) -> GrobidDTO:
        """
        Task to attempt to download and process a pdf for abstract and citations

        :param session: HTTP session to use to download pdf
        :param work_dir: Working directory to download pdfs to
        :param title: Title of paper to download
        :param pdf_url: URL of pdf to download
        :raises ClientResponseError: If fail to download PDF
        :raises GrobidProcessError: If grobid fails to process the file
        :return: DTO with paper title, abstract, and list of citations
        """
        tmp_pdf = ""
        try:
            with NamedTemporaryFile(dir=work_dir, prefix="grobid-", suffix=".pdf", delete=False) as tmp_pdf:
                # todo - change to download n papers successfully, then send all to grobid to process?
                # limit number of downloaded pdfs
                async with self._pdf_file_semaphore:
                    # block to prevent excessive downloads
                    async with self._download_semaphore:
                        timer = Timer()
                        await download_pdf(session, title, pdf_url, tmp_pdf.name)
                        loggy.debug_info(f"Saved '{tmp_pdf.name}' in {timer.format_time()}s | {pdf_url}")

                    # process paper
                    return await self.process_paper(tmp_pdf.name, title)
        finally:
            # delete file to support windows
            if tmp_pdf and exists(tmp_pdf.name):
                os.remove(tmp_pdf.name)

    async def enrich_papers(self, paper_db: PaperDatabase, papers: List[PaperDTO]) -> int:
        """
        Process a list of papers and save them to the database

        :param paper_db: Database to save paper details to
        :param papers: List of paper titles and their download urls
        :return Number of papers successfully processed
        """
        num_success = 0
        unique_citations = set()
        num_fail_download = 0
        num_fail_process = 0
        num_misc_error = 0
        # create tmp working directory to download all papers to
        with TemporaryDirectory(prefix='grobid-') as work_dir:
            async with ClientSession() as session:
                # queue tasks
                tasks = [self._process_paper_task(session, work_dir, p.id, p.pdf_url) for p in papers]
                loggy.debug_info(f"Processing {len(papers)} papers")
                # save results as complete
                for future in loggy.async_data_queue(tasks, "Processing papers", "papers"):
                    try:
                        result: GrobidDTO = await future
                        result.paper.download_status = 200
                        # update abstract
                        paper_db.upsert_paper(result.paper)

                        # add referenced papers, if any
                        if result.citations:
                            unique_citations.update(result.citations)
                            paper_db.insert_citation_paper_batch(result.paper.id, result.citations)

                        num_success += 1

                    # no file to download
                    except NoFileDataError as e:
                        loggy.error(e)
                        paper_db.upsert_paper(PaperDTO(e.paper_title, download_status=204))
                        num_fail_download += 1

                    # bad file format
                    except InvalidFileFormatError as e:
                        loggy.error(e)
                        paper_db.upsert_paper(PaperDTO(e.paper_title, download_status=415))
                        num_fail_download += 1

                    # failed to download pdf
                    except PaperDownloadError as e:
                        loggy.error(e)
                        paper_db.upsert_paper(
                            PaperDTO(e.paper_title, download_status=e.status_code, download_error_msg=e.error_msg))
                        num_fail_download += 1
                    # failed to parse pdf
                    except GrobidProcessError as e:
                        loggy.error(e)
                        paper_db.upsert_paper(
                            PaperDTO(e.paper_title, grobid_status=e.status_code, grobid_error_msg=e.error_msg))
                        num_fail_process += 1
                    # misc exception
                    except Exception as e:
                        loggy.error(e)
                        num_misc_error += 1

        # report results
        if len(papers):
            def __percent(a: int, b: int) -> str:
                """
                Format a percent

                :param a: Numerator
                :param b: Denominator
                :return: Formated percent
                """
                return f"{(a / b) * 100:.01f}%"

            loggy.info(f"Processing complete, successfully downloaded and processed "
                       f"{num_success} papers ({__percent(num_success, len(papers))})")
            loggy.debug_info(f"Found {len(unique_citations)} citations")
            loggy.debug_warn(
                f"Failed to download {num_fail_download} papers ({__percent(num_fail_download, len(papers))})")
            loggy.debug_warn(f"Failed to process {num_fail_process} papers "
                             f"({__percent(num_fail_process, len(papers))})")

        return num_success
