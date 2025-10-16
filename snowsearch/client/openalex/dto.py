from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from client.openalex.config import OPENALEX_PREFIX
from db.config import DOI_PREFIX

"""
File: dto.py
Description: DTOs for OpenAlex

@author Derek Garcia
"""


@dataclass
class PaperDTO:
    id: str
    title: str
    doi: str
    is_open_access: bool
    pdf_url: str

    def to_properties(self, with_id: bool = True) -> Dict[str, str | bool | datetime]:
        props = {
            'openalex_id': self.id.removeprefix(OPENALEX_PREFIX) if self.id else None,
            'doi': self.doi.removeprefix(DOI_PREFIX) if self.doi else None,
            'is_open_access': self.is_open_access,
            'pdf_url': self.pdf_url,
            'openalex_status': 200,
            'time_added': datetime.now()
        }
        if with_id:
            props['id'] = self.title
        return props
