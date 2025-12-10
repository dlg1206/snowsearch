import asyncio
import csv
from argparse import Namespace
from typing import List

from dotenv import load_dotenv

from cli.inspect import run_inspect
from cli.parser import create_parser
from cli.rank import run_rank
from cli.search import run_search
from cli.slr import run_slr
from cli.snowball import run_snowball
from db.paper_database import PaperDatabase
from util.config_parser import Config
from util.logger import logger, Level

"""
File: __main__.py

Description: Entry for interacting with snowball

@author Derek Garcia
"""


async def _execute(db: PaperDatabase, args: Namespace) -> None:
    """
    Execute a given cli command

    :param db: Paper database
    :param args: args to get command details from
    """

    def __load_papers_from_csv(csv_path: str) -> List[str]:
        """
        Load a list of papers from a csv file

        :param csv_path: Path to csv file with paper titles
        :return: List of loaded titles
        """
        with open(csv_path, 'r') as f:
            return list({r[0] for r in csv.reader(f)})

    match args.command:
        case 'slr':
            # todo - log if fail to connect to ollama / openai
            await run_slr(db, Config(args.config), args.semantic_search,
                          oa_query=args.query,
                          skip_paper_ranking=args.skip_ranking,
                          json_output=args.json)
        case 'snowball':
            config = Config(args.config)
            # load args
            papers_per_round = None if args.no_limit else config.snowball.papers_per_round
            papers = __load_papers_from_csv(args.papers_input) if args.papers_input else list(set(args.papers))

            # start snowball
            await run_snowball(db, config,
                               nl_query=args.semantic_search,
                               papers_per_round=papers_per_round,
                               seed_paper_titles=papers)

        case 'search':
            run_search(db, args.semantic_search,
                       paper_limit=args.limit,
                       exact_match=args.exact_match,
                       only_open_access=args.only_open_access,
                       only_processed=args.only_processed,
                       min_similarity_score=args.min_similarity_score,
                       order_by_abstract=args.order_by_abstract)

        case 'inspect':
            run_inspect(db, args.paper_title)

        case 'rank':
            rank_config = Config(args.config).ranking
            # load args
            papers = None
            if args.papers_input or args.papers:
                papers = __load_papers_from_csv(args.papers_input) if args.papers_input else list(set(args.papers))

            # start rank
            await run_rank(db, rank_config, args.semantic_search,
                           rank_config.top_n_papers if args.limit is None else args.limit,
                           rank_config.min_abstract_score if args.min_similarity_score is None else args.min_similarity_score,
                           json_output=args.json,
                           paper_titles_to_rank=papers)


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

    with PaperDatabase() as db:
        try:
            db.init()
            asyncio.run(_execute(db, args))
        except Exception as e:
            logger.fatal(e)


if __name__ == "__main__":
    load_dotenv()
    main()
