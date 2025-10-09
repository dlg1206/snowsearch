import json
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Dict, List

import findpapers

from snowsearch.clients.ollama import OllamaClient

"""
File: findpapers.py

Description: Client for interacting with findpapers api

@author Derek Garcia
"""

NL_TO_FPQ_CONTEXT_FILE = "snowsearch/prompts/nl_to_findpapers_query.prompt"


@dataclass
class PaperDTO:
    """
    Util DTO for tracking paper details
    """
    title: str
    abstract: str
    doi: str


class FindpapersClient:
    def __init__(self, config: Dict[str, str | int | List[str] | None]):
        """
        Create new findpapers client for searching for papers

        :param config: findpapers config details
        """
        self._config = config

    def search(self, query: str) -> List[PaperDTO]:
        """
        Search for matching papers using findpapers query

        query lang: https://github.com/jonatasgrosman/findpapers?tab=readme-ov-file#search-query-construction

        :param query: findpapers style query to search for papers from
        :return: List of relevant papers
        """
        with NamedTemporaryFile(prefix="findpapers", suffix=".json") as fp_tmp:
            # todo - supress output?
            findpapers.search(fp_tmp.name, query, **self._config)
            # load results
            with open(fp_tmp.name, 'r') as f:
                results = json.load(f)
        # extract relevant details
        return [PaperDTO(r['title'], r['abstract'], r['doi']) for r in results['papers']]


class FindpapersOllamaClient(OllamaClient):
    def __init__(self, ollama_host: str, ollama_port: int, model_name: str, model_tag: str = "latest"):
        """
        Create new Ollama Client for findpapers
        Configured for 1 client 1 model

        :param ollama_host: Host of ollama server
        :param ollama_port: Host of ollama port
        :param model_name: Name of model to use
        :param model_tag: Optional model tag (default: latest)
        """
        super().__init__(ollama_host, ollama_port, model_name, model_tag)
        # load content for few-shot
        with open(NL_TO_FPQ_CONTEXT_FILE, 'r') as f:
            self._nl_to_fpq_context = f.read()

    def prompt_to_fpq(self, prompt: str) -> str:
        """
        Use an LLM to convert a natural language search query
        to a findpapers style search query

        :param prompt: Natural language query for papers
        :return: findpapers query string
        """
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._nl_to_fpq_context},
                {"role": "user", "content": f"\nNatural language prompt:\n{prompt.lower().strip()}"}
            ],
            temperature=0,
        )

        return completion.choices[0].message.content
