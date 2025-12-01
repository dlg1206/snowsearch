import hashlib
from dataclasses import dataclass

"""
File: dto.py
Description: DTOs for Abstract ranker

@author Derek Garcia
"""


@dataclass
class AbstractDTO:
    paper_title: str
    content: str
    id: str = None

    def __post_init__(self):
        self.content = self.content.strip()
        self.id = hashlib.md5(self.paper_title.encode("utf-8")).hexdigest()[:5]     # short temp uid to pass to the llm
