from argparse import ArgumentParser

from util.logger import Level, DEFAULT_LOG_LEVEL

"""
File: parser.py
Description: CLI command parser

@author Derek Garcia
"""

PROG_NAME = "snowsearch"


#
# Generic Flags
#

def _add_semantic_search_arg(command, positional: bool = True) -> None:
    """
    Add a semantic search to the command

    :param command: Command to add arg to
    :param positional: Use as positional arg (Default: True)
    """
    params = {
        'metavar': "<semantic-search>",
        'type': str,
        'help': "Descriptive, natural language search for desired papers. "
                "i.e. \"AI-driven optimization of renewable energy systems\""
    }
    # set value
    if positional:
        command.add_argument('semantic_search', **params)
    else:
        command.add_argument('-ss', '--semantic-search', **params)


def _add_json_flag_arg(command) -> None:
    """
    Add json flag to command

    :param command: Command to add arg to
    """
    command.add_argument('-j', '--json',
                         metavar="<json-file-path>",
                         type=str,
                         help="Save the results to json file instead of printing to stdout")


def _add_paper_titles_flag_arg(command) -> None:
    """
    Add paper titles flag to command

    :param command: Command to add arg to
    """
    command.add_argument('-p', '--papers',
                         metavar="<paper-titles>",
                         type=str,
                         nargs="+",
                         help="One or more paper titles to start with. i.e \"Graph Attention Networks\" \"GINE\"")


def _add_paper_titles_input_flag_arg(command) -> None:
    """
    Add paper titles input flag to command

    :param command: Command to add arg to
    """
    command.add_argument('-i', '--papers-input',
                         metavar="<csv-file-path>",
                         type=str,
                         help="Path to csv file with list of paper titles to start with")


def _add_limit_flag_arg(command) -> None:
    """
    Add limit flag to command

    """
    command.add_argument('-l', '--limit',
                         metavar="<limit>",
                         type=int,
                         help="Limit the number of papers to return")


def _add_min_similarity_score_flag_arg(command) -> None:
    """
    Add min similarity score flag to command

    :param command: Command to add arg to
    """
    command.add_argument('-m', '--min-similarity-score',
                         metavar="<score>",
                         type=float,
                         help="Score between -1 and 1 to be the minimum similarity match to filter for")


def _add_ignore_quota_process_flag_arg(command) -> None:
    """
    Add ignore_quota flag to command

    :param command: Command to add arg to
    """
    command.add_argument('--ignore-quota',
                         action="store_true",
                         help="Do not retry to process additional papers to meet the round paper quota")


#
# Commands
#

def _add_slr_command(root_command) -> None:
    """
    Add the slr command

    :param root_command: Command to add arg to
    """
    desc = ("Perform a complete literature search with search generation, "
            "rounds of snowballing, and final abstract LLM ranking.")
    slr = root_command.add_parser('slr', description=desc, help=desc)
    # add generic args
    _add_semantic_search_arg(slr)
    _add_ignore_quota_process_flag_arg(slr)

    # add unique args
    slr.add_argument('-q', '--query',
                     metavar="<query>",
                     type=str,
                     help="OpenAlex formatted query to use directly. "
                          "See https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities#boolean-searches for formatting rules")  # skip llm generation step

    slr_group = slr.add_mutually_exclusive_group()
    _add_json_flag_arg(slr_group)
    # exclusive since can't write to json if skip ranking
    slr_group.add_argument('--skip-ranking',
                           action="store_true",
                           help="Skip the final paper ranking using an LLM",
                           default=False)


def _add_snowball_command(root_command) -> None:
    """
    Add the snowball command

    :param root_command: Command to add arg to
    """
    desc = "Perform snowballing using papers stored in the database without the initial OpenAlex search or LLM ranking"
    snowball = root_command.add_parser('snowball', description=desc, help=desc)
    # add generic args
    _add_semantic_search_arg(snowball, False)
    _add_ignore_quota_process_flag_arg(snowball)

    # add unique args
    snowball.add_argument('--no-limit',
                          action="store_true",
                          help="Set no citation cap and process all new unprocessed papers")

    snowball_group = snowball.add_mutually_exclusive_group()
    _add_paper_titles_flag_arg(snowball_group)
    _add_paper_titles_input_flag_arg(snowball_group)


def _add_search_command(root_command) -> None:
    """
    Add the search command

    :param root_command: Command to add arg to
    """
    desc = "Search the database for matching papers"
    search = root_command.add_parser('search', description=desc, help=desc)

    # add generic args
    _add_semantic_search_arg(search)
    _add_limit_flag_arg(search)
    _add_min_similarity_score_flag_arg(search)
    _add_json_flag_arg(search)

    # add unique args
    search.add_argument('--only-open-access',
                        action="store_true",
                        help="Only return papers that are publicly accessible")

    search.add_argument('--only-processed',
                        action="store_true",
                        help="Only return papers that have been successfully processed with Grobid")

    search.add_argument('--order-by-abstract',
                        action="store_true",
                        help="Order by abstract similarity first")

    search.add_argument('-e', '--exact-match',
                        action="store_true",
                        help="Title must contain the search query exactly (case insensitive).")


def _add_inspect_command(root_command) -> None:
    """
    Add the inspect command

    :param root_command: Command to add arg to
    """
    desc = "Get details about a paper"
    inspect = root_command.add_parser('inspect', description=desc, help=desc)
    inspect.add_argument('paper_title',
                         metavar="<title-of-paper>",
                         type=str,
                         help="Print details of a given paper")


def _add_rank_command(root_command) -> None:
    """
    Add the rank command

    :param root_command: Command to add arg to
    """
    desc = "Rank papers that best match the provided search"
    rank = root_command.add_parser('rank', description=desc, help=desc)
    # add generic args
    _add_semantic_search_arg(rank)
    _add_limit_flag_arg(rank)
    _add_min_similarity_score_flag_arg(rank)
    _add_json_flag_arg(rank)

    rank_group = rank.add_mutually_exclusive_group()
    _add_paper_titles_flag_arg(rank_group)
    _add_paper_titles_input_flag_arg(rank_group)


def _add_upload_command(root_command) -> None:
    """
    Add the upload command

    :param root_command: Command to add arg to
    """
    desc = "Upload papers locally to the database"
    upload = root_command.add_parser('upload', description=desc, help=desc)

    upload_group = upload.add_mutually_exclusive_group(required=True)
    upload_group.add_argument('-f', '--file',
                              metavar='<pdf-path>',
                              type=str,
                              help="Path to pdf file to upload and process")
    upload_group.add_argument('-d', '--directory',
                              metavar='<pdf-directory-path>',
                              type=str,
                              help="Path to root directory of pdf files to upload")


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
                        help=f"Path to config file to use")

    # logging flags
    logging = parser.add_argument_group("Logging")
    logging.add_argument("-l", "--log-level",
                         metavar="<log level>",
                         choices=[Level.INFO.name, Level.WARN.name, Level.ERROR.name, Level.DEBUG.name],
                         help=f"Set log level (Default: {DEFAULT_LOG_LEVEL.name}) ({[Level.INFO.name, Level.WARN.name, Level.ERROR.name, Level.DEBUG.name]})",
                         default=DEFAULT_LOG_LEVEL.name)
    logging.add_argument("-s", "--silent",
                         action="store_true",
                         help="Run in silent mode")

    # Create subparsers for different commands
    commands = parser.add_subparsers(dest='command', required=True)

    _add_slr_command(commands)  # slr command - main toolchain
    _add_snowball_command(commands)  # snowball command - perform snowballing without openalex search or llm ranking
    _add_search_command(commands)  # search command - search papers in the database
    _add_inspect_command(commands)  # inspect command - show details about a paper in the database
    _add_rank_command(commands)  # rank command - use llm to rank papers
    _add_upload_command(commands)  # upload command - upload local pdfs to the database

    return parser
