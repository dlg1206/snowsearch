"""
File: __main__.py

Description: Entry for interacting with snowball

@author Derek Garcia
"""

import asyncio
import csv
from argparse import Namespace
from os.path import exists
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from cli.inspect import run_inspect
from cli.parser import create_parser
from cli.rank import run_rank
from cli.search import run_search
from cli.slr import run_slr
from cli.snowball import run_snowball
from cli.upload import run_upload
from config.parser import Config, DEFAULT_CONFIG_PATH
from db.paper_database import PaperDatabase
from util.logger import logger, Level

async def _execute(db: PaperDatabase, config: Config, args: Namespace) -> None:
    """
    Execute a given cli command

    :param db: Paper database
    :param config: Config details
    :param args: args to get command details from
    """

    def __load_papers_from_csv(csv_path: str) -> List[str]:
        """
        Load a list of papers from a csv file

        :param csv_path: Path to csv file with paper titles
        :return: List of loaded titles
        """
        with open(csv_path, 'r', encoding='utf-8') as f:
            return list({r[0] for r in csv.reader(f)})

    match args.command:
        case 'slr':
            # todo - log if fail to connect to ollama / openai
            await run_slr(db, config, args.semantic_search,
                          oa_query=args.query,
                          skip_paper_ranking=args.skip_ranking,
                          json_output=args.json,
                          ignore_quota=args.ignore_quota)
        case 'snowball':
            # load args
            round_quota = None if args.no_limit else config.snowball.round_quota
            papers = None
            # load titles if provided
            if args.papers_input:
                papers = __load_papers_from_csv(args.papers_input)
            if args.papers:
                papers = list(set(args.papers))

            # start snowball
            await run_snowball(db, config,
                               nl_query=args.semantic_search,
                               round_quota=round_quota,
                               seed_paper_titles=papers,
                               ignore_quota=args.ignore_quota)

        case 'search':
            run_search(db, args.semantic_search,
                       paper_limit=args.limit,
                       exact_match=args.exact_match,
                       only_open_access=args.only_open_access,
                       only_processed=args.only_processed,
                       min_similarity_score=args.min_similarity_score,
                       order_by_abstract=args.order_by_abstract,
                       json_output=args.json)

        case 'inspect':
            run_inspect(db, args.paper_title)

        case 'rank':
            rank_config = config.ranking
            # load args
            papers = None
            # load titles if provided
            if args.papers_input:
                papers = __load_papers_from_csv(args.papers_input)
            if args.papers:
                papers = list(set(args.papers))

            # start rank
            top_n_papers = args.limit or rank_config.top_n_papers  # use config as fallback
            min_score = rank_config.min_abstract_score if args.min_similarity_score is None \
                else args.min_similarity_score

            await run_rank(db, rank_config, args.semantic_search,
                           top_n_papers,
                           min_score,
                           json_output=args.json,
                           paper_titles_to_rank=papers)
        case 'upload':
            # set file paths
            paper_pdf_paths = [args.file] if args.file else [str(f) for f in Path(args.directory).iterdir() if
                                                             f.is_file()]
            await run_upload(db, config, paper_pdf_paths)


def main() -> None:
    """
    Parse initial arguments and execute commands
    """
    args = create_parser().parse_args()
    # set logging level
    if args.silent:
        # silent override all
        logger.set_log_level(Level.SILENT)
    elif args.log_level is not None:
        # else update if option
        logger.set_log_level(args.log_level)

    # load config details
    config = Config(args.config or (DEFAULT_CONFIG_PATH if exists(DEFAULT_CONFIG_PATH) else None))
    with PaperDatabase() as db:
        try:
            asyncio.run(_execute(db, config, args))
        except Exception as e:
            logger.fatal(e)


if __name__ == "__main__":
    load_dotenv()
    main()
