"""
File: crossref_dto.py

Description:

@author Derek Garcia
"""
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from pyzotero.zotero import Zotero

from db.zotero import ZoteroClient


@dataclass(frozen=True)
class SimplePersonDTO:
    """
    Simple DTO for author's first and last name
    """
    given: str
    family: str

@dataclass
class SimpleDateDTO:
    year: int = None
    month: int = None
    day: int = None

    def __init__(self, date_parts: List[int]):
        if len(date_parts) == 1:
            self.year = date_parts[0]
        elif len(date_parts) == 2:
            self.year, self.month = date_parts
        elif len(date_parts) == 3:
            self.year, self.month, self.day = date_parts

@dataclass(frozen=True)
class CrossrefDTO(ABC):
    """
    DTO excerpt of a Crossref API response
    """
    type: str
    title: str
    doi: str
    url: str
    authors: List[SimplePersonDTO]
    date: SimpleDateDTO
    abstract: str
    references: List[str]

    @abstractmethod
    def to_zotero_item(self, zotero_client: Zotero) -> Dict[str, Any]:
        pass


@dataclass(frozen=True)
class ThesisDTO(CrossrefDTO):
    university: str
    place: str

    def to_zotero_item(self, zotero_client: Zotero) -> Dict[str, Any]:
        # todo
        pass


@dataclass(frozen=True)
class ConferencePaperDTO(CrossrefDTO):
    proceedings_title: str
    conference_name: str
    publisher: str
    place: str
    pages: str
    isbn: str

    def to_zotero_item(self, zotero_client: Zotero) -> Dict[str, Any]:
        # todo
        pass

@dataclass(frozen=True)
class JournalArticleDTO(CrossrefDTO):
    publication: str
    volume: int
    issue: int
    pages: str
    journal_abbreviation: str
    issn: str

    def to_zotero_item(self, zotero_client: Zotero) -> Dict[str, Any]:
        # todo
        pass

@dataclass(frozen=True)
class BookSectionDTO(CrossrefDTO):
    editors: List[SimplePersonDTO]
    book_title: str
    volume: int
    publisher: str
    places: str
    isbn: str

    def to_zotero_item(self, zotero_client: Zotero) -> Dict[str, Any]:
        # todo
        pass
