import os
from dataclasses import dataclass
from typing import Dict, Any

import yaml

from ai import ollama
from openalex.config import NL_TO_QUERY_CONTEXT_FILE

"""
File: config_parser.py

Description: Load config yaml file into series of DTOs

@author Derek Garcia
"""

DEFAULT_CONFIG_PATH = "config.yaml"

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_PORT_ENV = "OLLAMA_PORT"


@dataclass
class AgentConfigDTO:
    model: str
    tag: str = None
    prompt_path: str = None
    ollama_host: str = None
    ollama_port: int = None

    def __post_init__(self):
        # ensure path exists
        if not os.path.exists(self.prompt_path):
            raise ValueError(f"Could not find prompt file '{self.prompt_path}'")

        # check port
        if self.ollama_port <= 0:
            raise ValueError(f"{self.ollama_port} is an invalid port")


@dataclass
class RankingConfigDTO:
    agent_config: AgentConfigDTO
    context_window: int
    abstracts_per_comparison: int
    tokens_per_word: int

    def __post_init__(self) -> None:
        # ensure context is positive
        if self.context_window <= 0:
            raise ValueError("Context window must be greater than 0")

        # ensure abstract comparisons is positive
        if self.abstracts_per_comparison < 2:
            raise ValueError("Need at least 2 abstracts to compare")

        # ensure tokens per word is positive
        if self.tokens_per_word <= 0:
            raise ValueError("A word must have a positive token count")


@dataclass
class OpenAlexConfigDTO:
    pass


@dataclass
class GrobidConfigDTO:
    pass


class Config:

    def __init__(self, config_file: str = DEFAULT_CONFIG_PATH) -> None:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # load query generation config
        if "query_generation" not in config:
            raise KeyError("Missing required key 'query_generation'")
        if "agent" not in config['query_generation']:
            raise KeyError("Missing required key 'query_generation.agent'")
        self._query_generation_config = _load_agent_config('query_generation.agent',
                                                           config['query_generation']['agent'])

    def _load_ranking_config(self) -> RankingConfigDTO:
        pass

    def load_openalex_config(self) -> OpenAlexConfigDTO:
        pass

    def load_grobid_config(self) -> GrobidConfigDTO:
        pass


def _load_agent_config(key: str, config: Dict[str, Any]) -> AgentConfigDTO:
    """
    Load LLM agent details from config file

    :param key: Root of config
    :param config: Dict of the agent details
    :raises KeyError: If agent config is missing a required key
    :return: LLM Agent DTO
    """
    # ensure required keys are present
    if "model" not in config:
        raise KeyError(f"Missing required key '{key}.model'")

    # use openai api
    if os.getenv(OPENAI_API_KEY_ENV):
        return AgentConfigDTO(config['model'])

    # else use ollama
    ollama_host = os.getenv(OLLAMA_HOST_ENV, config.get('ollama_host', ollama.DEFAULT_HOST))
    ollama_port = os.getenv(OLLAMA_PORT_ENV, config.get('ollama_port', ollama.DEFAULT_PORT))
    return AgentConfigDTO(config['model'],
                          config.get('tag', "latest"),
                          config.get('prompt_path', NL_TO_QUERY_CONTEXT_FILE),
                          ollama_host,
                          ollama_port)
