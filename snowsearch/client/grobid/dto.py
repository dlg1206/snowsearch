from dataclasses import dataclass
from typing import List

"""
File: dto.py
Description: DTOs for Grobid

@author Derek Garcia
"""


@dataclass
class CitationDTO:
    id: str
    doi: str
    citation_count: int = None

    def __hash__(self):
        return hash(self.id) + hash(self.doi)


@dataclass
class GrobidDTO:
    id: str
    abstract: str
    citations: List[CitationDTO]
