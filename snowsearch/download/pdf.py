"""
File: pdf.py

Description: Attempt to Download PDF from file

@author Derek Garcia
"""
from aiohttp import ClientSession, ClientResponseError

from download.config import DOWNLOAD_HEADERS, KILOBYTE, PDF_MAGIC
from download.exception import NoFileDataError, InvalidFileFormatError, PaperDownloadError
from util.logger import logger


async def download_pdf(session: ClientSession, title: str, pdf_url: str, output_path: str) -> None:
    """
    Download pdf to file

    :param session: HTTP session to use
    :param title: Name of paper to download
    :param pdf_url: URL of pdf to download
    :param output_path: Path to write PDF to
    :raises NoFileDataError: If no data to download
    :raises InvalidFileFormatError: If the file is not a PDF
    :raises PaperDownloadError: If fail to download PDF
    """
    try:
        async with session.get(pdf_url, headers=DOWNLOAD_HEADERS) as response:
            logger.debug_msg(f"Downloading '{response.url}'")
            response.raise_for_status()
            # download pdf
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
    except Exception as e:
        raise PaperDownloadError(title, 500, str(e), pdf_url) from e
