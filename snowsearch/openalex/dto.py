from dataclasses import dataclass, fields
from datetime import datetime
from typing import Dict

from db.config import DOI_PREFIX
from openalex.config import DEFAULT_WRAP

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

    def format_abstract(self, wrap: int = DEFAULT_WRAP) -> str | None:
        """
        Format the abstract to fit within a wrap

        :param wrap: Character limit to wrap at (Default: 100)
        :return: Formatted abstract, None if no abstract to format
        """
        # no text to format
        if not self.abstract_text:
            return None
        # format abstract
        abstract_text = self.abstract_text.split()
        line = "\t"
        abstract_text_formatted = ""
        while abstract_text:
            line += f"{abstract_text.pop(0)} "
            # print if exceed char limit or nothing left to add
            if len(line) > wrap or not abstract_text:
                abstract_text_formatted += f"{line.strip()}\n"
                line = "\t"
        # return results
        return abstract_text_formatted

    def __post_init__(self):
        if self.doi:
            self.doi = self.doi.removeprefix(DOI_PREFIX)

        if self.abstract_text:
            self.abstract_text = self.abstract_text.strip()

    def __hash__(self):
        return hash(self.id) + hash(self.doi)
