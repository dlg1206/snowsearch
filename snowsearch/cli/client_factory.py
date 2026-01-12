"""
File: client_factory.py

Description: Util factory to streaming the creation of Client interfaces

@author Derek Garcia
"""
import os
from dataclasses import asdict
from multiprocessing.managers import Namespace

from ai.model import ModelClient
from ai.ollama import OllamaClient
from ai.openai import OpenAIClient, OPENAI_API_KEY_ENV, InvalidAPIKeyError
from config.parser import Config, AgentConfigDTO
from db.zotero import LibraryType, ZoteroClient
from grobid.worker import GrobidWorker
from openalex.client import OpenAlexClient
from util.logger import logger


class ClientFactory:
    """
    Create a new factory to create clients
    """

    def __init__(self, config: Config):
        """
        Create new factory

        :param config: Object with config details
        """
        self._config = config

    def create_openalex_client(self) -> OpenAlexClient:
        """
        :return: OpenAlex Client
        """
        return OpenAlexClient(self._config.openalex.email)

    def create_grobid_worker(self) -> GrobidWorker:
        """
        :return: Grobid Client
        """
        return GrobidWorker(
            self._config.grobid.max_grobid_requests,
            self._config.grobid.max_concurrent_downloads,
            self._config.grobid.max_local_pdfs,
            self._config.grobid.client_params
        )

    def create_query_generation_client(self) -> ModelClient:
        """
        :return: Model Client for query generation
        """
        return _create_model_client(self._config.query_generation)

    def create_rank_client(self) -> ModelClient:
        """
        :return: Model Client for ranking papers
        """
        return _create_model_client(self._config.ranking.agent_config)

    @staticmethod
    def create_zotero_client(args: Namespace) -> ZoteroClient | None:
        """
        Create new Zotero Client from args

        :param args: Argument object with zotero args
        :return: ZoteroClient or None if required args are missing
        """
        if args.zotero_group_library:
            return ZoteroClient(args.zotero_group_library, LibraryType.GROUP)
        elif args.zotero_user_library:
            # user library
            return ZoteroClient(args.zotero_user_library, LibraryType.USER, args.zotero_collection)
        else:
            # required args not present
            return None


def _create_model_client(agent_config: AgentConfigDTO) -> ModelClient:
    """
    Create a model client using the provided agent config
    Will return OpenAI if OPENAI_API_KEY env var is set and Ollama if OpenAI fails or key is not present

    :param agent_config: Agent config to use to make the client
    :return: Model Client
    """
    if os.getenv(OPENAI_API_KEY_ENV):
        try:
            return OpenAIClient(agent_config.model_name, agent_config.context_window)
        except InvalidAPIKeyError as e:
            logger.error_exp(e)
            logger.warn("Failed to verify OpenAI key, using Ollama as fallback", e)

    return OllamaClient(**asdict(agent_config))
