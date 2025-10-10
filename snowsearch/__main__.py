from argparse import ArgumentParser
from typing import Dict, List

import yaml
from dotenv import load_dotenv

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


def main():
    """
    Parse initial arguments and execute commands
    """
    args = _create_parser().parse_args()
    config = _load_config(args.config)


if __name__ == '__main__':
    load_dotenv()
    main()
