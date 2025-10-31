import asyncio
from argparse import Namespace

from dotenv import load_dotenv

from cli.parser import create_parser
from cli.slr import slr
from db.paper_database import PaperDatabase
from util.config_parser import Config
from util.logger import logger, Level

"""
File: __main__.py

Description: Entry for interacting with snowball

@author Derek Garcia
"""


def _execute(db: PaperDatabase, args: Namespace) -> None:
    """
    Execute a given cli command

    :param db: Paper database
    :param args: args to get command details from
    """

    match args.command:
        case 'slr':
            # todo - log if fail to connect to ollama / openai
            asyncio.run(slr(db, Config(args.config), args.search, args.query, args.json))


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
            _execute(db, args)
        except Exception as e:
            logger.fatal(e)


if __name__ == "__main__":
    load_dotenv()
    main()
