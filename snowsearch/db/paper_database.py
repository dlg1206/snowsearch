import os
import warnings
from datetime import datetime
from typing import Dict

from sentence_transformers import SentenceTransformer

from db.database import Neo4jDatabase
from db.entity import Node, NodeType, RelationshipType
from util.logger import logger
from util.timer import Timer

"""
File: paper_database.py

Description: Specialized interface for abstracting Neo4j commands to the database

@author Derek Garcia
"""

# embedding model details
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS = 384
SENTENCE_TRANSFORMER_CACHE = ".cache/huggingface/hub"

DOI_PREFIX = "https://doi.org/"

# suppress cuda warnings
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")


class PaperDatabase(Neo4jDatabase):
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL, model_dimensions: int = DEFAULT_DIMENSIONS):
        """
        Create new instance of the interface

        :param embedding_model: Optional embedding model to use (Default: all-MiniLM-L6-v2)
        :param model_dimensions: Optional dimensions of embedding model. Must match the provided embedding model (Default: 384)
        """
        super().__init__()
        self._model_dimensions = model_dimensions
        # todo - add option in config?
        # use gpu if cuda available
        from torch.cuda import is_available
        if is_available():
            device = "cuda"
            logger.debug_msg("Embedding model utilizing gpu")
        else:
            device = "cpu"
            logger.warn("Using cpu to create abstract embeddings -- this may impact performance")

        # download embedding model if needed
        model_downloaded = _is_model_local(embedding_model)
        timer = None
        if not model_downloaded:
            logger.warn(f"Embedding model '{embedding_model}' not downloaded locally, downloading now")
            timer = Timer()

        self._embedding_model = SentenceTransformer(embedding_model, device=device)
        if timer:
            logger.info(f"Downloaded '{embedding_model}' in {timer.format_time()}s")

    def init(self) -> None:
        """
        Create additional embedding index on init
        """
        init_query = "MATCH (n:DB_Metadata) RETURN n.initialized LIMIT 1"
        with self._driver.session() as session:
            # skip if already initialized
            if session.run(init_query).single():
                logger.debug_msg("Database already initialized, skipping. . .")
                return
        # regular init
        super().init()
        # create embedding index
        index_query = f'''
        CREATE VECTOR INDEX paper_abstract_index
        FOR (p:{NodeType.PAPER.value}) ON (p.abstract_embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {self._model_dimensions},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        '''
        with self._driver.session() as session:
            session.run(index_query)
        # record as initialized
        self.insert_node(Node.create(NodeType.DB_METADATA, {'initialized': datetime.now()}))

    def start_run(self) -> int:
        """
        Start a run in the database

        :return: Run ID
        """
        # get the latest run to +1
        query = "MATCH (n:Run) RETURN n.id ORDER BY toInteger(n.id) DESC LIMIT 1"
        with self._driver.session() as session:
            latest_run = session.run(query).single()
            new_run_id = latest_run['n.id'] + 1 if latest_run else 1  # init at 1 if no previous runs
        # create new run in db
        run_node = Node.create(NodeType.RUN, {'id': new_run_id, 'start': datetime.now()})
        self.insert_node(run_node)
        return new_run_id

    def insert_openalex_query(self, run_id: int, model: str, prompt: str, query: str) -> None:
        """
        Update a run with the openalex prompt and resulting query

        todo add additional OpenAlex filters used
        https://docs.openalex.org/api-entities/works/filter-works#works-attribute-filters

        :param run_id: ID of run
        :param model: Model used to generate OpenAlex query
        :param prompt: Original natural language prompt
        :param query: Resulting OpenAlex query
        """
        run_node = Node.create(NodeType.RUN, {
            'id': run_id,
            'openalex_model': model,
            'openalex_prompt': prompt,
            'openalex_query': query
        })
        self.insert_node(run_node, True)

    def insert_new_paper(self, run_id: int, title: str) -> None:
        """
        Insert a paper into the database

        :param run_id: ID of run paper found
        :param title: Title of paper
        """
        # # add paper
        # abstract_embedding = self._embedding_model.encode(abstract, show_progress_bar=False).tolist()
        paper_node = Node.create(NodeType.PAPER, {
            'id': title,
            'time_added': datetime.now()
        })
        self.insert_node(paper_node)

        # add relationship to current run
        run_node = Node.create(NodeType.RUN, {'id': run_id})
        self.insert_relationship(run_node,
                                 run_node.create_relationship_to(paper_node.type, RelationshipType.ADDED),
                                 paper_node)

    def update_paper(self,
                     title: str,
                     doi: str = None,
                     is_open_access: bool = None,
                     pdf_url: str = None) -> None:
        """
        Update paper fields. Only provided fields will be updated

        :param title: Title of paper
        :param doi: DOI of paper
        :param is_open_access: Is the paper open access?
        :param pdf_url: URL of downloadable PDF
        """
        # set properties
        properties: Dict[str, str | bool | datetime] = {'id': title}
        if doi:
            properties['doi'] = doi.removeprefix(DOI_PREFIX)
        if is_open_access is not None:
            properties['is_open_access'] = is_open_access
        if pdf_url:
            properties['pdf_url'] = pdf_url
        # update node
        self.insert_node(Node.create(NodeType.PAPER, properties), True)


def _is_model_local(embedding_model: str) -> bool:
    """
    Util method to check if the embedding model is downloaded locally

    :param embedding_model: Name of sentence transformer embedding model to use
    :return: True if downloaded, false otherwise
    """
    cache_dir = os.path.expanduser(f"~/{SENTENCE_TRANSFORMER_CACHE}")
    model_path = os.path.join(cache_dir, f"models--sentence-transformers--{embedding_model}")
    return os.path.exists(model_path)
