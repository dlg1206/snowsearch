"""
File: exception.py
Description: Exceptions for Grobid

@author Derek Garcia
"""


class PaperDownloadError(Exception):
    def __init__(self, paper_title: str, status_code: int, error_msg: str):
        """
        Create new failure error

        :param paper_title: Title of paper failed to process
        :param status_code: Grobid server status code
        :param error_msg: Grobid server error message
        """
        super().__init__(f"Failed to download '{paper_title}' | {status_code} | {error_msg}")
        self._paper_title = paper_title
        self._status_code = status_code
        self._error_msg = error_msg

    @property
    def paper_title(self) -> str:
        return self._paper_title

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def error_msg(self) -> str:
        return self._error_msg


class GrobidProcessError(Exception):
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
        return self._paper_title

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def error_msg(self) -> str:
        return self._error_msg
