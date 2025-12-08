import re

"""
File: config.py
Description: Defaults for OpenAlex

@author Derek Garcia
"""

# https://docs.openalex.org/api-guide-for-llms#rate-limiting-best-practices
# 1 requests per second
DEFAULT_RATE_LIMIT_SLEEP = 1
# 10 requests per second
POLITE_RATE_LIMIT_SLEEP = 0.1
# https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging?q=per_page#basic-paging
MAX_PER_PAGE = 200
# https://docs.openalex.org/api-guide-for-llms#bulk-lookup-by-dois
MAX_DOI_PER_PAGE = 50

NL_TO_QUERY_CONTEXT_FILE = "snowsearch/prompts/nl_to_openalex_query.prompt"
QUERY_JSON_RE = re.compile(r'\{\n.*"query": "(.*?)"')
# attempts to generate OpenAlex query
MAX_RETRIES = 3

OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_PREFIX = "https://openalex.org/"

DEFAULT_WRAP = 100