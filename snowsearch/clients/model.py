from abc import ABC
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

"""
File: model.py

Description: Generic model client for connecting and authenticating with an OpenAI compatible servers

@author Derek Garcia
"""


class ModelClient(ABC):

    def __init__(self, model: str, api_key: str, base_url: str = None):
        """
        Initialize connection to OpenAI compatible server

        :param model: Model to use
        :param api_key: API key to use
        :param base_url: Optional url of server (Default: assumes openai)
        """
        self._model = model

        params = {'api_key': api_key}
        # for ollama server
        if base_url:
            params['base_url'] = base_url
        # create client
        self._model_client = OpenAI(**params)

    def prompt(self, **prompt_kwargs: Any) -> ChatCompletion:
        """
        Prompt a gpt model

        :param prompt_kwargs: OpenAI kwargs for chat
        :return: Completed chat object
        """
        return self._model_client.chat.completions.create(model=self._model, **prompt_kwargs)
