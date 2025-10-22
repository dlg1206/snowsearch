from dataclasses import dataclass

"""
File: dto.py
Description: DTOs for Abstract ranker

@author Derek Garcia
"""


@dataclass
class AbstractDTO:
    id: str
    text: str
    tokens: int

    def __post_init__(self):
        self.text = self.text.strip()
