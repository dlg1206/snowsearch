import asyncio
import logging
from asyncio import Semaphore
from datetime import datetime
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Dict, Any, Callable, Coroutine

import grobid_tei_xml
from aiohttp import ClientSession, ClientResponseError
from grobid_client.grobid_client import GrobidClient

from db.paper_database import PaperDatabase
from grobid.config import MAX_CONCURRENT_DOWNLOADS, MAX_PDF_COUNT, KILOBYTE, DOWNLOAD_HEADERS, MAX_RETRIES, PDF_MAGIC
from grobid.dto import GrobidDTO, CitationDTO
from grobid.exception import PaperDownloadError, GrobidProcessError, NoFileDataError, InvalidFileFormatError
from openalex.dto import PaperDTO
from util.logger import logger
from util.timer import Timer

"""
File: worker.py
Description: 

@author Derek Garcia
"""

# Mute grobid client logs
logging.getLogger("grobid_client").setLevel(logging.CRITICAL + 1)  # todo doesn't work


class GrobidWorker:
    def __init__(self, config: Dict[str, str | int | List[str] | None] = None):
        """
        Create new Grobid worker to download and process urls

        :param config: Grobid config file
        """
        if config:
            self._grobid_client = GrobidClient(**config)
        else:
            self._grobid_client = GrobidClient()
        self._grobid_semaphore = Semaphore()  # todo config to handle max requests at one time
        self._download_semaphore = Semaphore(MAX_CONCURRENT_DOWNLOADS)  # todo config to handle max requests at one time
        self._pdf_file_semaphore = Semaphore(MAX_PDF_COUNT)  # todo config to handle max requests at one time

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
                await _retry_wrapper(lambda: self._download_pdf(session, title, pdf_url, tmp_pdf.name), backoff=.1)
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

        return GrobidDTO(title, doc.abstract, [CitationDTO(c.title, c.doi) for c in doc.citations])

    async def process_papers(self, paper_db: PaperDatabase, papers: List[PaperDTO]):
        """
        Process a list of papers and save them to the database

        :param paper_db: Database to save paper details to
        :param papers: List of paper titles and their download urls
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
        percent = lambda a, b: f"{(a / b) * 100:.01f}%"
        logger.info(
            f"Processing complete, successfully download and processed {num_success} papers ({percent(num_success, len(papers))})")
        logger.info(f"Found {len(citations)} citations")
        logger.info(f"Failed to download {num_fail_download} papers ({percent(num_fail_download, len(papers))})")
        logger.info(f"Failed to process {num_fail_process} papers ({percent(num_fail_process, len(papers))})")


async def _retry_wrapper(callback: Callable[[], Coroutine[Any, Any, Any]],
                         retries: int = MAX_RETRIES,
                         backoff: float = 0.0) -> Any:
    """
    Util method to wrap retry logic

    :param callback: Async callback function to attempt
    :param retries: Number to retries (Default: 3)
    :param backoff: Delay time for exponential backoff (Default: 0 - ie no backoff)
    :return: Return values of the callback function
    """
    last_exception = None
    for attempt in range(0, retries + 1):
        try:
            return await callback()
        except Exception as e:
            last_exception = e
            # exponential backoff
            await asyncio.sleep(backoff * 2 ** attempt)
    # exceed retries, raise exception
    raise last_exception
