import asyncio
import os
from argparse import ArgumentParser
from typing import Dict, List

import yaml
from dotenv import load_dotenv

from client.ai import ollama
from client.ai.ollama import OllamaClient
from client.ai.openai import OpenAIClient
from client.grobid.grobid_worker import GrobidWorker
from client.openalex.openalex_client import OpenAlexClient
from db.paper_database import PaperDatabase
from util.logger import logger

"""
File: __main__.py

Description: Shell entry for interacting with service

@author Derek Garcia
"""
PROG_NAME = "snowsearch"
DEFAULT_CONFIG_PATH = "conf.yaml"


def _load_config(conf_file: str) -> Dict[str, Dict[str, str | int | List[str] | None]]:
    """
    Load yaml config file

    :param conf_file: Path to config file to use
    :return: Dict of entire config file
    """
    with open(conf_file, 'r') as file:
        return yaml.safe_load(file)


def _create_parser() -> ArgumentParser:
    """
    Create the Arg parser

    :return: Arg parser
    """
    parser = ArgumentParser(
        description="SnowSearch: AI-powered snowball systemic literature review assistant",
        prog=PROG_NAME
    )

    # add optional config file arg
    parser.add_argument('-c', '--config',
                        metavar="<path to config file>",
                        help=f"Path to config file to use (Default: {DEFAULT_CONFIG_PATH})",
                        default=DEFAULT_CONFIG_PATH)

    return parser


def main(db):
    """
    Parse initial arguments and execute commands
    """
    args = _create_parser().parse_args()
    config = _load_config(args.config)
    # use openai api
    if os.getenv('OPENAI_API_KEY'):
        model_client = OpenAIClient(config['agent']['model'])
    # use ollama
    else:
        # check env, conf, then default for values
        ollama_host = os.getenv('OLLAMA_HOST', config.get('ollama_host', ollama.DEFAULT_HOST))
        ollama_port = os.getenv('OLLAMA_PORT', config.get('ollama_port', ollama.DEFAULT_PORT))
        model_client = OllamaClient(ollama_host, ollama_port, config['agent']['model'], config['agent'].get('tag', "latest"))
    run_id = db.start_run()
    openalex_config = config['openalex']
    oa_client = OpenAlexClient(model_client, openalex_config.get('email') if openalex_config else "foo@foo.com")
    prompt = 'Agentic AI that uses "formal methods" to ensure consistency'
    # query = oa_client.prompt_to_query(prompt)
    query = '"agentic" AND ("AI" OR "artificial intelligence") AND "formal methods" OR "consistency"'

    # save to db
    # query = "[agentic] AND ([AI] OR [artificial intelligence]) AND [formal methods] OR [consistency]"
    # query = '"agentic" AND ("AI" OR "artificial intelligence") AND "formal methods" OR "consistency"'
    # query = "[artificial intelligence]"
    # query = "([artificial intelligence] OR [AI] OR [machine learning] OR [ML] OR [deep learning] OR [DL]) AND ([music] OR [s?ng])"
    # query = "[sbom] AND [landscape]"

    db.insert_openalex_query(run_id, oa_client.model, prompt, query)
    asyncio.run(oa_client.save_seed_papers(run_id, db, query))
    #
    # # # todo - rerun query if exact match?
    #
    def snowball(grobid):
        papers = db.get_unprocessed_pdf_urls(run_id, 10)
        asyncio.run(grobid.process_papers(db, papers))
        snowball_seed = db.search_by_prompt_papers(prompt, paper_limit=3, min_score=.4)
        citations = set()
        for p, _ in snowball_seed:
            citations.update(db.get_unprocessed_citations(p.id, 5))
        asyncio.run(oa_client.save_citation_papers(db, list(citations)))

    grobid = GrobidWorker()
    print("Round 1")
    snowball(grobid)
    # print("Round 2")
    # snowball(grobid)
    # print("Round 3")
    # snowball(grobid)

if __name__ == '__main__':
    logger.set_log_level('DEBUG')
    # logger.set_log_level('INFO')
    load_dotenv()
    with PaperDatabase() as db:
        db.init()
        main(db)
