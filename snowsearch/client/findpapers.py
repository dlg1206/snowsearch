import json
import re
import sys
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Dict, List

import findpapers

from client.ai.model import ModelClient
from util.logger import logger
from util.timer import Timer

"""
File: findpapers.py

Description: Client for interacting with findpapers api

@author Derek Garcia
"""

NL_TO_FPQ_CONTEXT_FILE = "snowsearch/prompts/nl_to_findpapers_query.prompt"
FPQ_JSON_RE = re.compile(r'\{\n.*"query": "(.*?)"')
# attempts to generate findpapers query
MAX_RETRIES = 3


@dataclass
class PaperDTO:
    """
    Util DTO for tracking paper details
    """
    title: str
    abstract: str
    doi: str


class ExceedMaxQueryGenerationAttemptsError(Exception):
    def __init__(self, model: str):
        super().__init__(f"Exceeded findpapers query generation using '{model}'")
        self._model = model

    @property
    def model(self) -> str:
        return self._model


# DEPRECATED Deprecated in favor of using OpenAlex directly, may reintegrate later
class FindpapersClient:
    def __init__(self, model_client: ModelClient,
                 config: Dict[str, str | int | List[str] | None]):
        """
        Create new findpapers client for searching for papers

        :param config: findpapers config details
        :param model_client: Client to use for making openai api requests
        """
        self._model_client = model_client
        self._config = config
        # load content for few-shot
        with open(NL_TO_FPQ_CONTEXT_FILE, 'r') as f:
            self._nl_to_fpq_context = f.read()

    def prompt_to_fpq(self, prompt: str) -> str:
        """
        Use an LLM to convert a natural language search query
        to a findpapers style search query

        :param prompt: Natural language query for papers
        :raises ExceedMaxQueryGenerationAttemptsError: If fail to extract query from model reply
        :return: findpapers query string
        """
        # error if exceed retries
        for attempt in range(0, MAX_RETRIES):
            logger.info(f"Generating findpapers query ({attempt + 1}/{MAX_RETRIES})")
            completion, timer = self._model_client.prompt(
                messages=[
                    {"role": "system", "content": self._nl_to_fpq_context},
                    {"role": "user", "content": f"\nNatural language prompt:\n{prompt.strip()}"}
                ],
                temperature=0
            )
            '''
            Attempt to extract the query from the response. 
            This is to safeguard against wordy and descriptive replies 
            '''
            query_match = FPQ_JSON_RE.findall(completion.choices[0].message.content.strip())
            if query_match:
                query = query_match[0].strip().replace("'", '"')  # replace double quotes
                # report success
                logger.info(f"Generated findpapers query in {timer.format_time()}s")
                logger.debug_msg(f"Generated query: {query}")
                return query
            # else retry
            if attempt + 1 < MAX_RETRIES:
                logger.warn("Failed to generate findpapers query, retrying. . .")
        # error if exceed retries
        raise ExceedMaxQueryGenerationAttemptsError(self._model_client.model)

    def search(self, query: str) -> List[PaperDTO]:
        """
        Search for matching papers using findpapers query

        query lang: https://github.com/jonatasgrosman/findpapers?tab=readme-ov-file#search-query-construction

        :param query: findpapers style query to search for papers from
        :return: List of relevant papers
        """
        with NamedTemporaryFile(prefix="findpapers-", suffix=".json") as fp_tmp:
            logger.info("Starting findpapers search, this may take a while")
            sys.stdout.flush()  # print msg before stderr
            timer = Timer()
            # todo - suppress output?
            # todo wrap in try catch in case query fails
            findpapers.search(fp_tmp.name, query, **self._config)
            sys.stderr.flush()  # flush all findpapers log messages first
            logger.info(f"Completed seed search in {timer.format_time()}s")
            # load results
            with open(fp_tmp.name, 'r') as f:
                results = json.load(f)
        # extract relevant details
        # todo - save paper details like doi and url to db
        return [PaperDTO(r['title'], r['abstract'], r['doi']) for r in results['papers']]

    @property
    def model(self) -> str:
        return self._model_client.model
