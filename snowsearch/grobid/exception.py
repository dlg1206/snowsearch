"""
File: exception.py
Description: Exceptions for Grobid

@author Derek Garcia
"""


class InvalidFileFormatError(Exception):
    """
    File is not a PDF
    """

    def __init__(self, paper_title: str, pdf_url: str):
        """
        File is not a PDF

        :param paper_title: Title of paper failed to process
        :param pdf_url: URL of invalid pdf
        """
        super().__init__(f"'{paper_title}' is not a pdf, skipping | {pdf_url}")
        self._paper_title = paper_title
        self._pdf_url = pdf_url

    @property
    def paper_title(self) -> str:
        """
        :return: Paper title
        """
        return self._paper_title

    @property
    def pdf_url(self) -> str:
        """
        :return: PDF url
        """
        return self._pdf_url


class NoFileDataError(Exception):
    """
    No data to read from the url
    """

    def __init__(self, paper_title: str, pdf_url: str):
        """
        No data to read from the url

        :param paper_title: Title of paper failed to process
        :param pdf_url: URL of invalid file
        """
        super().__init__(f"'{paper_title}' returned no data, skipping | {pdf_url}")
        self._paper_title = paper_title
        self._pdf_url = pdf_url

    @property
    def paper_title(self) -> str:
        """
        :return: Paper title
        """
        return self._paper_title

    @property
    def pdf_url(self) -> str:
        """
        :return: PDF url
        """
        return self._pdf_url


class PaperDownloadError(Exception):
    """
    Failed to download paper
    """

    def __init__(self, paper_title: str, status_code: int, error_msg: str, pdf_url: str):
        """
        Create new failure error

        :param paper_title: Title of paper failed to process
        :param status_code: Grobid server status code
        :param error_msg: Grobid server error message
        :param pdf_url: URL of failed pdf download
        """
        super().__init__(f"Failed to download '{paper_title}' | {status_code} | {error_msg} | {pdf_url}")
        self._paper_title = paper_title
        self._status_code = status_code
        self._error_msg = error_msg
        self._pdf_url = pdf_url

    @property
    def paper_title(self) -> str:
        """
        :return: Paper title
        """
        return self._paper_title

    @property
    def status_code(self) -> int:
        """
        :return: HTTP status code
        """
        return self._status_code

    @property
    def error_msg(self) -> str:
        """
        :return: Error Message
        """
        return self._error_msg


class GrobidProcessError(Exception):
    """
    Failed to process paper with grobid
    """

    def __init__(self, paper_title: str, status_code: int, error_msg: str):
        """
        Create new failure error

        :param paper_title: Title of paper failed to process
        :param status_code: Grobid server status code
        :param error_msg: Grobid server error message
        """
        super().__init__(f"Failed to process '{paper_title}' | {status_code} | {error_msg}")
        self._paper_title = paper_title
        self._status_code = status_code
        self._error_msg = error_msg

    @property
    def paper_title(self) -> str:
        """
        :return: Paper title
        """
        return self._paper_title

    @property
    def status_code(self) -> int:
        """
        :return: HTTP status code
        """
        return self._status_code

    @property
    def error_msg(self) -> str:
        """
        :return: Error Message
        """
        return self._error_msg
