import requests
from requests import HTTPError, ConnectionError

from client.model import ModelClient
from util.logger import logger
from util.timer import Timer

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


class InvalidOllamaServerError(Exception):
    def __init__(self, ollama_url: str):
        """
        Failed to connect to Ollama server

        :param ollama_url: URL of ollama server
        """
        super().__init__(f"Failed to connect to ollama server at '{ollama_url}'")
        self._ollama_url = ollama_url

    @property
    def ollama_url(self) -> str:
        return self._ollama_url


class UnknownOllamaModelError(Exception):
    def __init__(self, model: str):
        """
        Attempt to download an unknown model

        :param model: Model on Ollama to attempt to download
        """
        super().__init__(f"Could not find model '{model}'")
        self._model = model

    @property
    def model(self) -> str:
        return self._model


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
        logger.debug_msg("Using ollama client")
        super().__init__(
            model=f"{model_name}:{model_tag if model_tag else 'latest'}",
            api_key="ollama",
            base_url=f"http://{ollama_host}:{ollama_port}/v1"
        )
        self._ollama_server = f"http://{ollama_host}:{ollama_port}"

        # err if model invalid
        self._validate()

        # download if needed
        if not self._is_model_local():
            self._download_model()

    def _validate(self) -> None:
        """
        Ensure the ollama server is available and model is valid to download

        :raises InvalidOllamaServer: If could not connect to the Ollama server
        :raises UnknownOllamaModel: If could not find requested model in Ollama library
        """
        # verify ollama server up
        try:
            with requests.get(self._ollama_server) as r:
                r.raise_for_status()
        except ConnectionError as e:
            raise InvalidOllamaServerError(self._ollama_server) from e
        # verify model exists
        try:
            with requests.get(f"{MODEL_LIBRARY}/{self._model}") as r:
                r.raise_for_status()
        except HTTPError as e:
            raise UnknownOllamaModelError(self._model) from e

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
            logger.warn(f"'{self._model}' not downloaded locally")
            return False
        # model exists
        logger.debug_msg(f"'{self._model}' downloaded locally")
        return True

    def _download_model(self) -> None:
        """
        Download a model locally from ollama
        See https://ollama.com/search for all available models

        :raises HTTPError: If fail to download model
        """
        logger.info(f"Model '{self._model}' is not available locally, downloading. . .")
        timer = Timer()
        # todo - stream for custom loading bar
        with requests.post(f"{self._ollama_server}/{MODEL_DOWNLOAD_ENDPOINT}", json={"model": self._model}) as r:
            r.raise_for_status()
        logger.info(f"Downloaded '{self._model}' in {timer.format_time()}s")
