from abc import ABC
from typing import Any, Tuple


from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from util.logger import logger
from util.timer import Timer

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
        self._model_client = AsyncOpenAI(**params)

    async def prompt(self, **prompt_kwargs: Any) -> Tuple[ChatCompletion, Timer]:
        """
        Prompt a model

        :param prompt_kwargs: OpenAI kwargs for chat
        :return: Completed chat object and timer
        """
        logger.debug_msg(f"Prompting '{self._model}'")
        timer = Timer()
        completion = await self._model_client.chat.completions.create(model=self._model, **prompt_kwargs)
        logger.debug_msg(f"Response from '{self._model}' generated in {timer.format_time()}s")
        return completion, timer

    @property
    def model(self) -> str:
        return self._model
