from dataclasses import dataclass

from ai import ollama
from grobid.config import MAX_GROBID_REQUESTS, MAX_CONCURRENT_DOWNLOADS, MAX_PDF_COUNT
from rank.config import AVG_TOKEN_PER_WORD

"""
File: default.py

Description: Hardcoded defaults to fallback on

@author Derek Garcia
"""


@dataclass(frozen=True)
class AgentDefaults:
    MODEL = "llama3"
    TAG = "latest"
    CONTEXT_WINDOW = 5000  # conservative estimate


@dataclass(frozen=True)
class SnowballDefaults:
    SEED_PAPER_LIMIT = 10
    ROUNDS = 5
    ROUND_QUOTA = 5
    MIN_SIMILARITY_SCORE = 0.4


@dataclass(frozen=True)
class QueryGenerationDefaults:
    MODEL = AgentDefaults.MODEL
    TAG = AgentDefaults.TAG


@dataclass(frozen=True)
class AbstractRankingDefaults:
    MODEL = AgentDefaults.MODEL
    TAG = AgentDefaults.TAG
    CONTEXT_WINDOW = AgentDefaults.CONTEXT_WINDOW
    AVG_TOKEN_PER_WORD = AVG_TOKEN_PER_WORD
    MIN_ABSTRACT_SCORE = 0.6
    TOP_N_PAPERS = 10


@dataclass(frozen=True)
class GrobidDefaults:
    MAX_GROBID_REQUESTS = MAX_GROBID_REQUESTS
    MAX_CONCURRENT_DOWNLOADS = MAX_CONCURRENT_DOWNLOADS
    MAX_LOCAL_PDFS = MAX_PDF_COUNT


@dataclass(frozen=True)
class OllamaDefaults:
    OLLAMA_HOST = ollama.DEFAULT_HOST
    OLLAMA_PORT = ollama.DEFAULT_PORT
