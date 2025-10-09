import os

from snowsearch.clients.model import ModelClient

"""
File: openai.py

Description: Base client for interacting with OpenAI API like GPT

@author Derek Garcia
"""


class OpenAIClient(ModelClient):

    def __init__(self, model_name: str):
        """
        Create new Openai Client for use with OpenAI api like gpt
        Configured for 1 client 1 model

        :param model_name: Name of model to use
        """
        # ensure openapi key is present
        if not os.getenv('OPENAI_API_KEY'):
            raise EnvironmentError("Missing API key in environment variable: OPENAI_API_KEY")
        super().__init__(model=model_name, api_key=os.getenv('OPENAI_API_KEY'))

        # validate key
        self._verify_api_key()

    def _verify_api_key(self) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model
        :raises Exception: If failed to verify key and model
        """
        self._model_client.models.retrieve(self._model)
