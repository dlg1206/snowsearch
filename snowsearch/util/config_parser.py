import os
from dataclasses import dataclass
from typing import Dict, Any

import yaml

from ai import ollama
from rank.config import MIN_ABSTRACT_PER_COMPARISON, AVG_TOKEN_PER_WORD

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
    tag: str = "latest"
    ollama_host: str = ollama.DEFAULT_HOST
    ollama_port: int = ollama.DEFAULT_PORT

    def __post_init__(self):
        # check port
        if self.ollama_port <= 0:
            raise ValueError(f"{self.ollama_port} is an invalid port")


@dataclass
class RankingConfigDTO:
    agent_config: AgentConfigDTO
    context_window: int
    abstracts_per_comparison: int = None
    tokens_per_word: float = None

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
        # load config file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # load query generation config
        if 'query_generation' not in config:
            raise KeyError("Missing required key 'query_generation'")
        if 'agent' not in config['query_generation']:
            raise KeyError("Missing required key 'query_generation.agent'")
        self._query_generation_config = _load_agent_config('query_generation.agent',
                                                           config['query_generation']['agent'])

        # load abstract ranking config
        if 'abstract_ranking' not in config:
            raise KeyError("Missing required key 'abstract_ranking'")
        self._ranking_config = _load_ranking_config('abstract_ranking', config['abstract_ranking'])

        # load openalex config

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
                          ollama_host,
                          ollama_port)


def _load_ranking_config(key: str, config: Dict[str, Any]) -> RankingConfigDTO:
    """
    Load abstract ranking details from config

    :param key: Root of config
    :param config: Dict of the ranking details
    :raises KeyError: If ranking config is missing a required key
    :return: Abstract Ranking DTO
    """

    # load agent details
    if 'agent' not in config:
        raise KeyError(f"Missing required key '{key}.agent'")
    agent_config = _load_agent_config(f'{key}.agent', config['agent'])

    # load additional ranking details
    if 'context_window' not in config:
        raise KeyError(f"Missing required key '{key}.context_window'")

    # return dto
    return RankingConfigDTO(agent_config,
                            config['context_window'],
                            config.get('abstracts_per_comparison', MIN_ABSTRACT_PER_COMPARISON),
                            config.get('tokens_per_word', AVG_TOKEN_PER_WORD))


def _load_openalex_config(key: str, config: Dict[str, Any]) -> OpenAlexConfigDTO:
    pass
