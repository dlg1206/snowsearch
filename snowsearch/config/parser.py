import os
from dataclasses import dataclass
from typing import Dict, Any

import yaml

from config.default import OllamaDefaults, AgentDefaults, AbstractRankingDefaults, GrobidDefaults, SnowballDefaults
from grobid.config import MAX_CONCURRENT_DOWNLOADS, MAX_PDF_COUNT

"""
File: parser.py

Description: Load config yaml file into series of DTOs

@author Derek Garcia
"""

DEFAULT_CONFIG_PATH = "config.yaml"


@dataclass
class AgentConfigDTO:
    model_name: str = None
    model_tag: str = None
    context_window: int = None
    ollama_host: str = None
    ollama_port: int = None

    def __post_init__(self):
        # assign env or default if not set with config
        self.model_name = self.model_name or os.getenv('SS_AGENT_MODEL', AgentDefaults.MODEL)
        self.model_tag = self.model_tag or os.getenv('SS_AGENT_TAG', AgentDefaults.TAG)
        self.context_window = int(
            self.context_window or os.getenv('SS_AGENT_CONTEXT_WINDOW', AgentDefaults.CONTEXT_WINDOW))
        self.ollama_host = self.ollama_host or os.getenv('SS_OLLAMA_HOST', OllamaDefaults.OLLAMA_HOST)
        self.ollama_port = int(self.ollama_port or os.getenv('SS_OLLAMA_PORT', OllamaDefaults.OLLAMA_PORT))

        # validate config
        # check port
        if self.ollama_port <= 0:
            raise ValueError(f"{self.ollama_port} is an invalid port")

        # ensure context is positive
        if self.context_window <= 0:
            raise ValueError("Context window must be greater than 0")


@dataclass
class RankingConfigDTO:
    agent_config: AgentConfigDTO
    min_abstract_score: float = AbstractRankingDefaults.MIN_ABSTRACT_SCORE
    top_n_papers: int = AbstractRankingDefaults.TOP_N_PAPERS
    tokens_per_word: float = AbstractRankingDefaults.AVG_TOKEN_PER_WORD

    def __post_init__(self) -> None:
        # ensure tokens per word is positive
        if self.tokens_per_word and self.tokens_per_word <= 0:
            raise ValueError("A word must have a positive token count")

    @property
    def context_window(self) -> int:
        return self.agent_config.context_window


@dataclass
class OpenAlexConfigDTO:
    email: str = None

    def __post_init__(self):
        self.email = self.email or os.getenv('SS_OA_EMAIL')


@dataclass
class GrobidConfigDTO:
    client_params: Dict[str, Any] = None
    max_grobid_requests: int = GrobidDefaults.MAX_GROBID_REQUESTS
    max_concurrent_downloads: int = MAX_CONCURRENT_DOWNLOADS
    max_local_pdfs: int = MAX_PDF_COUNT

    def __post_init__(self):
        # set server endpoint if provided
        env_grobid_server = os.getenv('SS_GROBID_SERVER')
        if env_grobid_server:
            self.client_params = self.client_params or {}
            self.client_params.setdefault('grobid_server', env_grobid_server)

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
    rounds: int = SnowballDefaults.ROUNDS
    min_similarity_score: float = SnowballDefaults.MIN_SIMILARITY_SCORE
    seed_paper_limit: int = SnowballDefaults.SEED_PAPER_LIMIT
    round_quota: int = SnowballDefaults.ROUND_QUOTA

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

    def __init__(self, config_file: str = None) -> None:
        # use env + defaults if no config to use
        if not config_file:
            agent_config = AgentConfigDTO()
            self._snowball = SnowballConfigDTO()
            self._query_generation = agent_config
            self._ranking = RankingConfigDTO(agent_config)
            self._openalex = OpenAlexConfigDTO()
            self._grobid = GrobidConfigDTO()
            return

        # load config file if provided
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # get config overrides lambda
        get_params = lambda c, keys: {k: c[k] for k in keys if c.get(k)}

        # load configs
        # agent
        agent_params = get_params(config.get('agent', []), ['model_name', 'model_tag', 'context_window'])
        agent_params.update(get_params(config.get('ollama'), ['ollama_host', 'ollama_port']))
        agent_config = AgentConfigDTO(**agent_params)
        # snowball
        snowball_params = get_params(config.get('snowball', []),
                                     ['seed_paper_limit', 'rounds', 'round_quota', 'min_similarity_score'])
        self._snowball = SnowballConfigDTO(**snowball_params)
        # query generation
        self._query_generation = agent_config
        # ranking
        ranking_params = get_params(config.get('abstract_ranking'),
                                    ['tokens_per_word', 'min_abstract_score', 'top_n_papers'])
        self._ranking = RankingConfigDTO(agent_config=agent_config, **ranking_params)
        # openalex
        self._openalex = OpenAlexConfigDTO(**get_params(config.get('openalex', []), ['email']))
        # grobid
        grobid_params = get_params(config.get('grobid', []),
                                   ['max_grobid_requests', 'max_concurrent_downloads', 'max_local_pdfs'])
        grobid_client_params = get_params(config.get('grobid', {}).get('client_config', []),
                                          ['grobid_server', 'batch_size', 'sleep_time', 'timeout', 'coordinates'])
        self._grobid = GrobidConfigDTO(client_params=grobid_client_params, **grobid_params)

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
