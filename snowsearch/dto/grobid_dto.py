from dataclasses import dataclass
from typing import List

from dto.paper_dto import PaperDTO

"""
File: grobid_dto.py
Description: DTO for Grobid

@author Derek Garcia
"""


@dataclass
class GrobidDTO:
    paper: PaperDTO
    citations: List[PaperDTO]
