"""
File: grobid_dto.py
Description: DTO for Grobid

@author Derek Garcia
"""
from dataclasses import dataclass
from typing import List

from dto.paper_dto import PaperDTO


@dataclass
class GrobidDTO:
    """
    DTO for grobid results
    """
    paper: PaperDTO
    citations: List[PaperDTO]
