from dataclasses import dataclass
from typing import List

"""
File: dto.py
Description: DTOs for Grobid

@author Derek Garcia
"""


@dataclass
class CitationDTO:
    title: str
    doi: str


@dataclass
class GrobidDTO:
    title: str
    abstract: str
    citations: List[CitationDTO]
