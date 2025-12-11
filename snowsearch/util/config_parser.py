import os
from dataclasses import dataclass
from typing import Dict, Any

import yaml

from ai import ollama
from ai.ollama import OLLAMA_HOST_ENV, OLLAMA_PORT_ENV
from ai.openai import OPENAI_API_KEY_ENV
from grobid.config import MAX_GROBID_REQUESTS, MAX_CONCURRENT_DOWNLOADS, MAX_PDF_COUNT, GROBID_SERVER_ENV
from rank.config import AVG_TOKEN_PER_WORD

"""
File: config_parser.py

Description: Load config yaml file into series of DTOs

@author Derek Garcia
"""

DEFAULT_CONFIG_PATH = "config.yaml"


@dataclass
class AgentConfigDTO:
    model_name: str
    model_tag: str = "latest"
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
    min_abstract_score: float
    top_n_papers: int
    tokens_per_word: float = None

    def __post_init__(self) -> None:
        # ensure context is positive
        if self.context_window <= 0:
            raise ValueError("Context window must be greater than 0")

        # ensure tokens per word is positive
        if self.tokens_per_word and self.tokens_per_word <= 0:
            raise ValueError("A word must have a positive token count")


@dataclass
class OpenAlexConfigDTO:
    email: str | None = None


@dataclass
class GrobidConfigDTO:
    client_params: Dict[str, Any] = None
    max_grobid_requests: int = MAX_GROBID_REQUESTS
    max_concurrent_downloads: int = MAX_CONCURRENT_DOWNLOADS
    max_local_pdfs: int = MAX_PDF_COUNT

    def __post_init__(self):
        # check requests
        if self.max_grobid_requests < 1:
            raise ValueError("Must make at least one request to Grobid server")

        # check downloads
        if self.max_concurrent_downloads < 1:
            raise ValueError("Must make at least one paper download request")

        # check load pdfs
        if self.max_local_pdfs < 1:
            raise ValueError("Must have at least one paper downloaded at a time")


@dataclass
class SnowballConfigDTO:
    rounds: int
    min_similarity_score: float
    seed_paper_limit: int = None
    round_quota: int = None

    def __post_init__(self):
        if self.rounds < 0:
            raise ValueError("Snowball rounds cannot be negative")

        if self.min_similarity_score < -1 or self.min_similarity_score > 1:
            raise ValueError("Min similarity score must between -1 and 1")

        if self.seed_paper_limit and self.seed_paper_limit < 1:
            raise ValueError("Seed round needs at least 1 paper")

        if self.round_quota and self.round_quota < 1:
            raise ValueError("Each snowball round needs at least 1 paper")


class Config:

    def __init__(self, config_file: str = DEFAULT_CONFIG_PATH) -> None:
        # load config file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # load snowball config
        if not config.get('snowball'):
            raise KeyError("Missing required key 'snowball'")
        self._snowball = _load_snowball_config('snowball', config['snowball'])

        # load query generation config
        if not config.get('query_generation'):
            raise KeyError("Missing required key 'query_generation'")
        self._query_generation = _load_agent_config('query_generation.agent',
                                                    config['query_generation'].get('agent'))

        # load abstract ranking config
        if not config.get('abstract_ranking'):
            raise KeyError("Missing required key 'abstract_ranking'")
        self._ranking = _load_ranking_config('abstract_ranking', config['abstract_ranking'])

        # load openalex config if data, else use default
        self._openalex = _load_openalex_config(config['openalex']) if config.get('openalex') else OpenAlexConfigDTO()

        # load grobid config if data, else use default
        self._grobid = _load_grobid_config(config['grobid']) if config.get('grobid') else GrobidConfigDTO()

    @property
    def snowball(self) -> SnowballConfigDTO:
        return self._snowball

    @property
    def query_generation(self) -> AgentConfigDTO:
        return self._query_generation

    @property
    def ranking(self) -> RankingConfigDTO:
        return self._ranking

    @property
    def openalex(self) -> OpenAlexConfigDTO:
        return self._openalex

    @property
    def grobid(self) -> GrobidConfigDTO:
        return self._grobid


def _load_snowball_config(key: str, config: Dict[str, Any]) -> SnowballConfigDTO:
    """
    Load snowball details from config

    :param key: Root of config
    :param config: Dict of snowball details
    :return: Snowball DTO
    """
    # ensure required keys are present
    if not config.get('rounds'):
        raise KeyError(f"Missing required key '{key}.rounds'")
    if not config.get('min_similarity_score'):
        raise KeyError(f"Missing required key '{key}.min_similarity_score'")

    # return dto
    return SnowballConfigDTO(config['rounds'],
                             config['min_similarity_score'],
                             config.get('seed_paper_limit'),
                             config.get('round_quota'))


def _load_agent_config(key: str, config: Dict[str, Any]) -> AgentConfigDTO:
    """
    Load LLM agent details from config file

    :param key: Root of config
    :param config: Dict of the agent details
    :raises KeyError: If agent config is missing a required key
    :return: LLM Agent DTO
    """
    # ensure agent is present
    if not config:
        raise KeyError(f"Missing required key '{key}")

    # ensure required keys are present
    if not config.get('model'):
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
    agent_config = _load_agent_config(f'{key}.agent', config.get('agent'))

    # load additional ranking details
    if not config.get('context_window'):
        raise KeyError(f"Missing required key '{key}.context_window'")
    if not config.get('min_abstract_score'):
        raise KeyError(f"Missing required key '{key}.min_abstract_score'")
    if not config.get('top_n_papers'):
        raise KeyError(f"Missing required key '{key}.top_n_papers'")

    # return dto
    return RankingConfigDTO(agent_config,
                            config['context_window'],
                            config['min_abstract_score'],
                            config['top_n_papers'],
                            config.get('tokens_per_word', AVG_TOKEN_PER_WORD))


def _load_openalex_config(config: Dict[str, Any]) -> OpenAlexConfigDTO:
    """
    Load Open Alex details from config

    :param config: Dict of the Open Alex details
    :return: Open Alex DTO
    """
    return OpenAlexConfigDTO(config.get('email'))


def _load_grobid_config(config: Dict[str, Any]) -> GrobidConfigDTO:
    """
    Load grobid details from config

    :param config: Dict of the grobid details
    :return: Grobid DTO
    """
    # use env var if available
    client_params = config.get('client_config', {})
    if os.getenv(GROBID_SERVER_ENV):
        client_params['grobid_server'] = os.getenv(GROBID_SERVER_ENV)

    # return dto
    return GrobidConfigDTO(client_params,
                           config.get('max_grobid_requests', MAX_GROBID_REQUESTS),
                           config.get('max_concurrent_downloads', MAX_CONCURRENT_DOWNLOADS),
                           config.get('max_local_pdfs', MAX_PDF_COUNT))
