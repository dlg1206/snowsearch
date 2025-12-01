import asyncio
import logging
from asyncio import Semaphore
from datetime import datetime
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Dict, Any

import grobid_tei_xml
from aiohttp import ClientSession, ClientResponseError
from grobid_client.grobid_client import GrobidClient

from db.paper_database import PaperDatabase
from grobid.config import MAX_CONCURRENT_DOWNLOADS, MAX_PDF_COUNT, KILOBYTE, DOWNLOAD_HEADERS, PDF_MAGIC, \
    MAX_GROBID_REQUESTS
from grobid.dto import GrobidDTO
from grobid.exception import PaperDownloadError, GrobidProcessError, NoFileDataError, InvalidFileFormatError
from openalex.dto import PaperDTO
from util.logger import logger
from util.timer import Timer

"""
File: worker.py
Description: Grobid worker that handles downloading and processing open access papers

@author Derek Garcia
"""

# Mute grobid client logs
logging.getLogger("grobid_client").setLevel(logging.CRITICAL)  # todo doesn't work


class GrobidWorker:
    def __init__(self,
                 max_grobid_requests: int = MAX_GROBID_REQUESTS,
                 max_concurrent_downloads: int = MAX_CONCURRENT_DOWNLOADS,
                 max_local_pdfs: int = MAX_PDF_COUNT,
                 client_config: Dict[str, Any] = None):
        """
        Create new Grobid worker to download and process urls

        :param max_grobid_requests: Max number of requests allowed to be made to the grobid server at a time (Default: 1)
        :param max_concurrent_downloads: Max number of PDFs to allowed to download at the same time (Default: 10)
        :param max_local_pdfs: Max number of PDFs to be downloaded locally at one time (Default: 100)
        :param client_config: Optional Grobid client details (Default: None)
        """
        # init with params if provided
        self._grobid_client = GrobidClient(**client_config) if client_config else GrobidClient()
        # set semaphore limits
        self._grobid_semaphore = Semaphore(max_grobid_requests)
        self._download_semaphore = Semaphore(max_concurrent_downloads)
        self._pdf_file_semaphore = Semaphore(max_local_pdfs)

    async def _download_pdf(self, session: ClientSession, title: str, pdf_url: str, output_path: str) -> None:
        """
        Download pdf to file

        :param session: HTTP session to use
        :param title: Name of paper to download
        :param pdf_url: URL of pdf to download
        :param output_path: Path to write PDF to
        :raises ClientResponseError: If fail to download PDF
        """
        # block to prevent excessive downloads
        async with self._download_semaphore:
            try:
                async with session.get(pdf_url, headers=DOWNLOAD_HEADERS) as response:
                    logger.debug_msg(f"Downloading '{response.url}'")
                    response.raise_for_status()
                    # download pdf
                    timer = Timer()
                    first_pass = True
                    with open(output_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(KILOBYTE)

                            if not chunk:
                                # no data to write
                                if first_pass:
                                    raise NoFileDataError(title, pdf_url)
                                # break if no data left to read
                                break

                            # validate pdf
                            if first_pass:
                                # file is not a pdf
                                if not chunk.startswith(PDF_MAGIC):
                                    raise InvalidFileFormatError(title, pdf_url)
                                first_pass = False

                            f.write(chunk)
            except ClientResponseError as e:
                raise PaperDownloadError(title, e.status, e.message, pdf_url) from e

        logger.debug_msg(f"Saved '{output_path}' in {timer.format_time()}s | {pdf_url}")

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
        with NamedTemporaryFile(dir=work_dir, prefix="grobid-", suffix=".pdf") as tmp_pdf:
            timer = Timer()
            # limit number of downloaded pdfs
            async with self._pdf_file_semaphore:
                # attempt to download paper with retry logic
                await self._download_pdf(session, title, pdf_url, tmp_pdf.name)
                # block to prevent overwhelming grobid server
                async with self._grobid_semaphore:
                    logger.debug_msg(f"Processing '{tmp_pdf.name}' | {title}")
                    _, status, content = await asyncio.to_thread(self._grobid_client.process_pdf,
                                                                 service="processFulltextDocument",
                                                                 pdf_file=tmp_pdf.name,
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
        logger.debug_msg(f"Processed '{tmp_pdf.name}' in {timer.format_time()}s | {title}")

        return GrobidDTO(title, doc.abstract, [PaperDTO(c.title, c.doi) for c in doc.citations])

    async def enrich_papers(self, paper_db: PaperDatabase, papers: List[PaperDTO]) -> int:
        """
        Process a list of papers and save them to the database

        :param paper_db: Database to save paper details to
        :param papers: List of paper titles and their download urls
        :return Number of papers successfully processed
        """
        num_success = 0
        citations = set()
        num_fail_download = 0
        num_fail_process = 0
        num_misc_error = 0
        # create tmp working directory to download all papers to
        with TemporaryDirectory(prefix='grobid-') as work_dir:
            async with ClientSession() as session:
                # queue tasks
                tasks = [self._process_paper_task(session, work_dir, p.id, p.pdf_url) for p in papers]
                logger.debug_msg(f"Processing {len(papers)} papers")
                # save results as complete
                for future in logger.get_data_queue(tasks, "Processing papers", "papers", is_async=True):
                    try:
                        result: GrobidDTO = await future
                        # update abstract
                        paper_db.upsert_paper(PaperDTO(result.id,
                                                       abstract_text=result.abstract,
                                                       download_status=200,
                                                       grobid_status=200,
                                                       time_grobid_processed=datetime.now()))
                        # add referenced papers, if any
                        new_papers = []
                        if result.citations:
                            for c in result.citations:
                                new_papers.append(PaperDTO(c.id, doi=c.doi, time_added=datetime.now()))
                                citations.add(c)
                            paper_db.insert_citation_paper_batch(result.id, new_papers)

                        num_success += 1

                    # no file to download
                    except NoFileDataError as e:
                        logger.error_exp(e)
                        paper_db.upsert_paper(PaperDTO(e.paper_title, download_status=204))
                        num_fail_download += 1

                    # bad file format
                    except InvalidFileFormatError as e:
                        logger.error_exp(e)
                        paper_db.upsert_paper(PaperDTO(e.paper_title, download_status=415))
                        num_fail_download += 1

                    # failed to download pdf
                    except PaperDownloadError as e:
                        logger.error_exp(e)
                        paper_db.upsert_paper(
                            PaperDTO(e.paper_title, download_status=e.status_code, download_error_msg=e.error_msg))
                        num_fail_download += 1
                    # failed to parse pdf
                    except GrobidProcessError as e:
                        logger.error_exp(e)
                        paper_db.upsert_paper(
                            PaperDTO(e.paper_title, grobid_status=e.status_code, grobid_error_msg=e.error_msg))
                        num_fail_process += 1
                    # misc exception
                    except Exception as e:
                        logger.error_exp(e)
                        num_misc_error += 1

        # report results
        if len(papers):
            percent = lambda a, b: f"{(a / b) * 100:.01f}%"
            logger.info(
                f"Processing complete, successfully download and processed {num_success} papers ({percent(num_success, len(papers))})")
            logger.debug_msg(f"Found {len(citations)} citations")
            logger.debug_msg(
                f"Failed to download {num_fail_download} papers ({percent(num_fail_download, len(papers))})")
            logger.debug_msg(f"Failed to process {num_fail_process} papers ({percent(num_fail_process, len(papers))})")
        return num_success
