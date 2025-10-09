from abc import ABC

import requests
from requests import HTTPError

from snowsearch.clients.model import ModelClient

"""
File: ollama.py

Description: Base client for interacting with ollama server

https://docs.ollama.com/api

@author Derek Garcia
"""

# ollama details
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434
# endpoints
MODEL_LIBRARY = "https://ollama.com/library"
MODEL_DOWNLOAD_ENDPOINT = "api/pull"
MODEL_VIEW_ENDPOINT = "api/show"


class OllamaClient(ModelClient):

    def __init__(self, ollama_host: str, ollama_port: int, model_name: str, model_tag: str = "latest"):
        """
        Create new Ollama Client
        Configured for 1 client 1 model
        :param ollama_host: Host of ollama server
        :param ollama_port: Host of ollama port
        :param model_name: Name of model to use
        :param model_tag: Optional model tag (default: latest)
        """
        super().__init__(
            model=f"{model_name}:{model_tag if model_tag else 'latest'}",
            api_key="ollama",
            base_url=f"http://{ollama_host}:{ollama_port}/v1"
        )
        self._ollama_server = f"http://{ollama_host}:{ollama_port}"
        # err if model invalid
        self._validate_model()

        # download if needed
        if not self._is_model_local():
            self._download_model()

    def _validate_model(self) -> None:
        """
        Ensure the model is valid to download

        :raises HTTPError: If model is invalid
        """
        with requests.get(f"{MODEL_LIBRARY}/{self._model}") as r:
            r.raise_for_status()

    def _is_model_local(self) -> bool:
        """
        Check if the model is already downloaded / available locally

        :return: True if downloaded, false otherwise
        """
        try:
            with requests.post(f"{self._ollama_server}/{MODEL_VIEW_ENDPOINT}", json={"model": self._model}) as r:
                r.raise_for_status()
        except HTTPError:
            # todo - just assuming error is 404
            return False
        # model exists
        return True

    def _download_model(self) -> None:
        """
        Download a model locally from ollama
        See https://ollama.com/search for all available models

        :raises HTTPError: If fail to download model
        """
        # todo - stream for custom loading bar
        with requests.post(f"{self._ollama_server}/{MODEL_DOWNLOAD_ENDPOINT}", json={"model": self._model}) as r:
            r.raise_for_status()
