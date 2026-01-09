"""
File: zotero.py

Description: Client for interacting with a zotero library

@author Derek Garcia
"""
import os
from enum import Enum
from os.path import exists
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Dict, Any, Set, Tuple

from aiohttp import ClientSession
from pyzotero.zotero import Zotero
from pyzotero.zotero_errors import UserNotAuthorisedError, ResourceNotFoundError

from download.exception import NoFileDataError, InvalidFileFormatError, PaperDownloadError
from download.pdf import download_pdf
from dto.paper_dto import PaperDTO
from util.logger import logger

ZOTERO_API_KEY_ENV = "ZOTERO_API_KEY"


class LibraryType(Enum):
    """
    Enum representing the type of Zotero Library
    """
    USER = "user"
    GROUP = "group"


class InvalidAPIKeyError(Exception):
    """
    Failed to validate Zotero API key
    """

    def __init__(self):
        """
        Failed to validate Zotero API key
        """
        super().__init__("Failed to validate Zotero API key")


class InsufficientPermissionsError(Exception):
    """
    Zotero API key does not have the required permissions to work
    """

    def __init__(self, missing_permissions: List[str], library_id: str = None):
        """
        Zotero API key does not have the required permissions to work

        :param missing_permissions: List of missing permissions
        :param library_id: Optional ID of group library (Default: None)
        """
        msg = "Zotero API key is missing the following permissions to access "
        if library_id:
            msg += f"the group library '{library_id}': "
        else:
            msg += "your personal library: "
        super().__init__(msg + ", ".join(missing_permissions))


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
        # ensure zotero api key is present
        if not os.getenv(ZOTERO_API_KEY_ENV):
            raise EnvironmentError(f"Missing API key in environment variable: {ZOTERO_API_KEY_ENV}")
        self._zot = Zotero(library_id, library_type.value, os.getenv(ZOTERO_API_KEY_ENV))
        self._validate_api_key(library_type, library_id)
        # todo make list to support multiple collections
        if collection_key:
            try:
                self._zot.collection(collection_key)
            except ResourceNotFoundError as e:
                raise ValueError(f"Collection '{collection_key}' does not exist") from e

        self._collection_key = collection_key

    def _validate_api_key(self, library_type: LibraryType, library_id: str) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model

        :raises InvalidAPIKey: If failed to verify key and model
        """

        def __find_missing_perms(obj: Dict[str, bool]) -> List[str]:
            """
            Check for missing params for SnowSearch to work

            :param obj: Permissions object
            :return: List of missing permissions
            """
            m = []
            # value is always true, absence means permission is missing
            if not obj.get('library'):
                m.append('library')
            if not obj.get('write'):
                m.append('write')
            return m

        try:
            permissions = self._zot.key_info()
            logger.debug_msg("Zotero API key is valid")
            # validate personal library perms
            if library_type == LibraryType.USER:
                user_perms = permissions['access'].get('user', {})
                missing_perms = __find_missing_perms(user_perms)
                if missing_perms:
                    raise InsufficientPermissionsError(missing_perms)

                # warn if excessive permissions
                if 'notes' in user_perms:
                    logger.warn(
                        "Zotero API key has notes access but not needed for SnowSearch, considered removing access")
                if 'group' in permissions['access']:
                    logger.warn(
                        "Zotero API key has access to group libraries but configured to use for personal libraries, considered removing access")

            # validate group library params
            else:
                groups_perms = permissions['access'].get('groups', {})

                # attempt to get specific library key, use 'all' key as fallback
                key = library_id if library_id in groups_perms else 'all'
                group_perms = groups_perms.get(key, {})
                missing_perms = __find_missing_perms(group_perms)
                if missing_perms:
                    raise InsufficientPermissionsError(missing_perms, library_id=library_id)

                # warn if excessive permissions
                if key == 'all':
                    logger.warn(
                        f"Using default group permissions, consider defining permissions for only group library '{library_id}'")
                if key == library_id and len(groups_perms) > 1:
                    logger.warn(
                        f"Zotero API key has access to other group libraries but configured to use group library '{library_id}', considered removing access")
                if 'user' in permissions['access']:
                    logger.warn(
                        "Zotero API key has access to personal library but configured to use group libraries, considered removing access")

            logger.debug_msg("Zotero API key has sufficient permissions")
        except UserNotAuthorisedError as e:
            # bad API key
            raise InvalidAPIKeyError() from e

    def _fetch_existing_items(self) -> Tuple[Set[str], Set[str]]:
        """
        Fetch list of items in a zotero library


        :return: Set of DOI and titles of items
        """
        if self._collection_key:
            collection = self._zot.collection(self._collection_key)
            logger.info(f"Fetching details from collection '{collection['data']['name']}'")

        # fetch items
        zot_items = self._zot.everything(self._zot.collection_items(self._collection_key)
                                         if self._collection_key else self._zot.items())
        # return identifiers
        existing_doi, existing_titles = set(), set()
        for i in zot_items:
            doi = i['data'].get('DOI')
            if doi:
                existing_doi.add(doi.lower())

            title = i['data'].get('title')
            if title:
                existing_titles.add(title.lower())

        return existing_doi, existing_titles

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
            tmp_pdf = NamedTemporaryFile(dir=work_dir, suffix=".pdf", delete=False)
            try:
                await download_pdf(session, paper.id, paper.pdf_url, tmp_pdf.name)
                # download success, create placeholder pdf value
                template = self._zot.item_template('attachment', 'imported_file')
                template['title'] = paper.id
                template['contentType'] = 'application/pdf'
                template['filename'] = os.path.basename(tmp_pdf.name)
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

        existing_doi, existing_titles = self._fetch_existing_items()
        new_zot_items = []
        with TemporaryDirectory(prefix='zotero-') as work_dir:
            async with ClientSession() as session:
                tasks = []
                # find only new papers
                for p in papers:
                    if p.doi:
                        # DOI is available and not already added
                        if p.doi.lower() not in existing_doi:
                            tasks.append(self._create_zotero_item_task(session, p, work_dir))
                        else:
                            logger.debug_msg(f"Skipping duplicate DOI '{p.doi}'")
                    else:
                        # DOI is not available and title not already added
                        if p.id.lower() not in existing_titles:
                            tasks.append(self._create_zotero_item_task(session, p, work_dir))
                        else:
                            logger.debug_msg(f"Skipping duplicate title '{p.id}'")

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
            if new_zot_items:
                logger.info("Uploading items. . .")
                response = self._zot.create_items(new_zot_items)
                if response['failed']:
                    logger.warn("Some items failed to be created")

            # exit early if nothing to upload
            if not len(os.listdir(work_dir)):
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
                    'filename': r['data']['filename']
                })
            # upload pdfs
            self._zot.upload_attachments(attachments, basedir=work_dir)
            logger.info("PDFs uploaded")
