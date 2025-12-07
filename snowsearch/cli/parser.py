from argparse import ArgumentParser

from util.config_parser import DEFAULT_CONFIG_PATH
from util.logger import Level, DEFAULT_LOG_LEVEL

"""
File: parser.py
Description: CLI command parser

@author Derek Garcia
"""

PROG_NAME = "snowsearch"


def create_parser() -> ArgumentParser:
    """
    Create the Arg parser

    :return: Arg parser
    """
    parser = ArgumentParser(
        description="SnowSearch: AI-powered snowball systemic literature review assistant",
        prog=PROG_NAME
    )

    # add optional config file arg
    config = parser.add_argument_group("Configuration")
    config.add_argument('-c', '--config',
                        metavar="<path to config file>",
                        help=f"Path to config file to use (Default: {DEFAULT_CONFIG_PATH})",
                        default=DEFAULT_CONFIG_PATH)

    # Create subparsers for different commands
    commands = parser.add_subparsers(dest='command', required=True)

    # slr command - main toolchain
    slr = commands.add_parser('slr',
                              help="Perform a complete literature search with search generation, rounds of snowballing,"
                                   " and final abstract LLM ranking.")

    slr.add_argument('semantic_search',
                     metavar="<semantic-search>",
                     type=str,
                     help="Descriptive, natural language search for desired papers. i.e. \"AI-driven optimization of renewable energy systems\"")

    slr.add_argument('-q', '--query',
                     metavar="<query>",
                     type=str,
                     help="OpenAlex formatted query to use directly. "
                          "See https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities#boolean-searches for formatting rules")  # skip llm generation step

    slr_group = slr.add_mutually_exclusive_group()
    slr_group.add_argument('-j', '--json',
                           metavar="<json-file-path>",
                           type=str,
                           help="Save the results to json file instead of printing to stdout")

    # exclusive since can't write to json if skip ranking
    slr_group.add_argument('--skip-ranking',
                           action="store_true",
                           help="Skip the final paper ranking using an LLM",
                           default=False)

    # snowball command - just perform snowballing without openalex search or llm ranking
    snowball = commands.add_parser('snowball',
                                   help="Perform snowballing using papers stored in the database without the initial "
                                        "OpenAlex search or LLM ranking")

    snowball.add_argument('-ss', '--semantic-search',
                          metavar="<semantic-search>",
                          type=str,
                          help="Descriptive, natural language search for papers. i.e. \"AI-driven optimization of renewable energy systems\" "
                               "Default will use all unprocessed papers with no order")

    snowball.add_argument('-nl', '--no-limit',
                          action="store_true",
                          help="Set no citation cap and process all new unprocessed papers")

    snowball_group = snowball.add_mutually_exclusive_group()
    snowball_group.add_argument('-sp', '--seed-papers',
                          metavar="<paper-title>",
                          type=str,
                          nargs="+",
                          help="One or more paper titles to start snowballing with. i.e \"Graph Attention Networks\" \"GINE\"")

    snowball_group.add_argument('-i', '--seed-papers-input',
                                metavar="<csv-file-path>",
                                type=str,
                                help="Path to csv file with list of paper titles to start snowballing with")

# logging flags
    logging = parser.add_argument_group("Logging")
    logging.add_argument("-l", "--log-level",
                         metavar="<log level>",
                         choices=[Level.INFO.name, Level.WARN.name, Level.ERROR.name, Level.DEBUG.name],
                         help=f"Set log level (Default: {DEFAULT_LOG_LEVEL.name}) ({[Level.INFO.name, Level.WARN.name, Level.ERROR.name, Level.DEBUG.name]})",
                         default=DEFAULT_LOG_LEVEL.name)
    logging.add_argument("-s", "--silent",
                         action="store_true",
                         help="Run in silent mode",
                         default=False)

    return parser
