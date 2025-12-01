from dataclasses import dataclass, fields
from datetime import datetime

from typing import Dict
from db.config import DOI_PREFIX

"""
File: dto.py
Description: DTOs for OpenAlex

@author Derek Garcia
"""


@dataclass
class PaperDTO:
    id: str
    openalex_url: str = None
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

    @classmethod
    def create_dto(cls, data: Dict[str, str | int | datetime]) -> "PaperDTO":
        """
        Create DTO from dict and remove any invalid fields

        :param data: Data to use to create DTO
        :return: PaperDTO
        """
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    def __post_init__(self):
        if self.doi:
            self.doi = self.doi.removeprefix(DOI_PREFIX)

        if self.abstract_text:
            self.abstract_text = self.abstract_text.strip()
