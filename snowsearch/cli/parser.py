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
    slr = commands.add_parser('slr', help="Perform a literature search with rounds of snowballing")

    slr.add_argument('search',
                     metavar="<search>",
                     type=str,
                     help="Descriptive search query for desired papers. i.e. \"AI-driven optimization of renewable energy systems\"")

    slr.add_argument('-q', '--query',
                     metavar="<query>",
                     type=str,
                     help="OpenAlex formatted query to use directly. "
                          "See https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities#boolean-searches for formatting rules")  # skip llm generation step

    slr.add_argument('-j', '--json',
                     metavar="<json-file-path>",
                     type=str,
                     help="Save the results to json file instead of printing to stdout")

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
