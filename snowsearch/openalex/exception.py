"""
File: exception.py
Description: Exceptions for OpenAlex

@author Derek Garcia
"""


class ExceedMaxQueryGenerationAttemptsError(Exception):
    """
    Exceed query string generation attempts
    """

    def __init__(self, model: str):
        """
        Failed to generate valid query string

        :param model: Model used to attempt to generate query
        """
        super().__init__(f"Exceeded OpenAlex query generation using '{model}'")
        self._model = model

    @property
    def model(self) -> str:
        """
        :return: Model Name
        """
        return self._model


class MissingOpenAlexEntryError(Exception):
    """
    Could not find an entry in OpenAlex
    """

    def __init__(self, doi: str, title: str):
        """
        Failed to find citation in OpenAlex database

        :param doi: DOI id of citation
        :param title: Paper title of citation
        """
        error_msg = f"Could not find citation in OpenAlex | '{title}'"
        if doi:
            error_msg += f" | {doi}"
        super().__init__(error_msg)
        self._doi = doi
        self._title = title

    @property
    def doi(self) -> str:
        """
        :return: DOI
        """
        return self._doi

    @property
    def title(self) -> str:
        """
        :return: Paper title
        """
        return self._title
