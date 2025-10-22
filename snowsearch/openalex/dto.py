from dataclasses import dataclass
from datetime import datetime

from db.config import DOI_PREFIX

"""
File: dto.py
Description: DTOs for OpenAlex

@author Derek Garcia
"""


@dataclass
class PaperDTO:
    id: str
    openalex_id: str = None
    doi: str = None
    abstract_text: str = None
    is_open_access: bool = None
    pdf_url: str = None
    openalex_status: int = None
    download_status: int = None
    download_error_msg: str = None
    grobid_status: int = None
    grobid_error_msg: str = None
    time_grobid_processed: datetime = None
    time_added: datetime = None

    def __post_init__(self):
        if self.doi:
            self.doi = self.doi.removeprefix(DOI_PREFIX)

        if self.abstract_text:
            self.abstract_text = self.abstract_text.strip()
