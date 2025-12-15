"""
File: exception.py
Description: Exceptions for Abstract Ranker

@author Derek Garcia
"""


class ExceedMaxRankingGenerationAttemptsError(Exception):
    """
    Failed to generate a valid ranking
    """

    def __init__(self, model: str):
        """
        Failed to generate valid ranking

        :param model: Model used to attempt to generate ranking
        """
        super().__init__(f"Exceeded abstract ranking generation using '{model}'")
        self._model = model

    @property
    def model(self) -> str:
        """
        :return: Model name
        """
        return self._model
