"""
File: zotero.py

Description: Client for interacting with a zotero library

@author Derek Garcia
"""
import os
from enum import Enum
from os.path import exists
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Dict, Any, Set

from aiohttp import ClientSession
from pyzotero.zotero import Zotero

from download.exception import NoFileDataError, InvalidFileFormatError, PaperDownloadError
from download.pdf import download_pdf
from dto.paper_dto import PaperDTO
from util.logger import logger


class LibraryType(Enum):
    """
    Enum representing the type of Zotero Library
    """
    USER = "user"
    GROUP = "group"


class ZoteroClient:
    """
    Client for interacting with a single Zotero Collection
    """

    def __init__(self, library_id: str, library_type: LibraryType, collection_key: str = None):
        """
        Create new Zotero Client

        :param library_id: ID of library to connect to
        :param library_type: Type of library
        :param collection_key: Optional collection key to specific collection to work with (Default: All)
        """
        # todo - check for api key
        self._zot = Zotero(library_id, library_type.value, os.getenv('ZOTERO_API_KEY'))
        # todo make list to support multiple collections
        self._collection_key = collection_key

    def _fetch_existing_doi(self) -> Set[str]:
        """
        Fetch list of DOI in a zotero library

        # todo include titles for fallback
        :return: Set of DOI
        """
        if self._collection_key:
            collection = self._zot.collection(self._collection_key)
            logger.info(f"Fetching details from collection '{collection['data']['name']}'")

        # fetch items
        zot_items = self._zot.everything(self._zot.collection_items(self._collection_key)
                                         if self._collection_key else self._zot.items())
        # find identifiers
        existing_doi = set()
        for item in zot_items:
            doi = item['data'].get('DOI')
            if doi:
                existing_doi.add(doi.lower())
        return existing_doi

    async def _create_zotero_item_task(self, session: ClientSession, paper: PaperDTO, work_dir: str) -> Dict[str, Any]:
        """
        Task to create a zotero item and attempt to download pdf if available

        :param session: Client Session to use to download PDF
        :param paper: Paper to create item for
        :param work_dir: Working directory to save pdf to if needed
        :return: Zotero formated item
        """
        # if pdf url and download didn't already fail
        if paper.pdf_url and paper.download_status != 500:
            tmp_pdf = NamedTemporaryFile(dir=work_dir, prefix="zotero-", suffix=".pdf", delete=False)
            try:
                await download_pdf(session, paper.id, paper.pdf_url, tmp_pdf.name)
                # download success, create placeholder pdf value
                template = self._zot.item_template('attachment', 'imported_file')
                template['title'] = 'PDF'
                template['contentType'] = 'application/pdf'
                template['filename'] = 'PDF.pdf'
                return template
            except (NoFileDataError, InvalidFileFormatError, PaperDownloadError) as e:
                logger.error_exp(e)
                # delete file to support windows
                if tmp_pdf and exists(tmp_pdf.name):
                    os.remove(tmp_pdf.name)

        # download either failed or no pdf url, create dummy as next best thing
        template = self._zot.item_template('journalArticle')
        template['title'] = paper.id
        template['DOI'] = paper.doi
        return template

    async def upload_papers(self, papers: List[PaperDTO]) -> None:
        """
        Upload a list of papers to a Zotero collection

        :param papers: List of papers to upload
        """

        existing_doi = self._fetch_existing_doi()
        new_zot_items = []
        with TemporaryDirectory(prefix='zotero-') as work_dir:
            async with ClientSession() as session:
                tasks = []
                # find only new papers
                for p in papers:
                    if not p.doi or p.doi and p.doi not in existing_doi:
                        tasks.append(self._create_zotero_item_task(session, p, work_dir))
                    else:
                        logger.debug_msg(f"Skipping duplicate DOI '{p.doi}'")
                # exit early if nothing new to add
                if not tasks:
                    logger.warn("No new papers to upload, exiting. . .")
                    return
                # else wait for items to finish
                for future in logger.get_data_queue(tasks, "Creating Zotero items", "papers", is_async=True):
                    template = await future
                    if self._collection_key:
                        template['collections'] = [self._collection_key]
                    new_zot_items.append(template)

            # upload items to Zotero
            logger.info("Uploading items. . .")
            response = self._zot.create_items(new_zot_items)
            if response['failed']:
                logger.warn("Some items failed to be created")
            tmp_pdfs = [f.name for f in Path(work_dir).iterdir() if f.is_file()]
            # exit early if nothing to upload
            if not tmp_pdfs:
                logger.info("No PDFs to upload")
                return

            # upload PDF attachments to Zotero
            logger.info("Uploading pdfs. . .")
            attachments = []
            for r in response['successful'].values():
                # ignore any DOI items
                if r['data'].get('contentType') != 'application/pdf':
                    continue
                # else assign a pdf to the new blank item
                attachments.append({
                    'key': r['key'],
                    'filename': tmp_pdfs.pop()
                })
            # upload pdfs
            self._zot.upload_attachments(attachments, basedir=work_dir)
            logger.info("PDFs uploaded")
