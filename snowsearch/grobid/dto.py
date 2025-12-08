from dataclasses import dataclass
from typing import List

from dto.paper_dto import PaperDTO

"""
File: dto.py
Description: DTOs for Grobid

@author Derek Garcia
"""


@dataclass
class GrobidDTO:
    id: str
    abstract: str
    citations: List[PaperDTO]
