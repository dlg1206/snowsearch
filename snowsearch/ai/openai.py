import os

from openai import AuthenticationError

from ai.model import ModelClient
from util.logger import logger

"""
File: openai.py

Description: Base client for interacting with OpenAI API like GPT

@author Derek Garcia
"""

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

class InvalidAPIKeyError(Exception):
    def __init__(self):
        """
        Failed to validate OpenAI API key
        """
        super().__init__("Failed to validate OpenAI API key")


class OpenAIClient(ModelClient):

    def __init__(self, model_name: str, context_window: int):
        """
        Create new Openai Client for use with OpenAI api like gpt
        Configured for 1 client 1 model

        :param model_name: Name of model to use
        :param context_window: Context window of model
        :raises EnvironmentError: If the 'OPENAI_API_KEY' env var is not defined
        """
        logger.debug_msg("Using openai client")
        # ensure openapi key is present
        if not os.getenv('OPENAI_API_KEY'):
            raise EnvironmentError("Missing API key in environment variable: OPENAI_API_KEY")
        super().__init__(model=model_name, context_window=context_window, api_key=os.getenv('OPENAI_API_KEY'))

        # validate key
        self._verify_api_key()

    def _verify_api_key(self) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model

        :raises InvalidAPIKey: If failed to verify key and model
        """
        try:
            self._model_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyError() from e
