import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

from sentence_transformers import SentenceTransformer

from db.database import Neo4jDatabase
from db.entity import Node, NodeType, RelationshipType
from util.logger import logger
from util.timer import Timer

"""`
File: paper_database.py

Description: Specialized interface for abstracting Neo4j commands to the database

@author Derek Garcia
`"""

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

    def upsert_paper(self, title: str,
                     run_id: int = None,
                     openalex_id: str = None,
                     doi: str = None,
                     abstract_text: str = None,
                     is_open_access: bool = None,
                     pdf_url: str = None,
                     download_status: int = None,
                     download_error_msg: str = None,
                     grobid_status: int = None,
                     grobid_error_msg: str = None,
                     time_grobid_processed: datetime = None,
                     time_added: datetime = None) -> None:
        """
        Insert paper into database and optional details

        :param title: Title of paper (key)
        :param run_id: ID of run
        :param openalex_id: OpenAlex Work ID
        :param doi: DOI of paper
        :param abstract_text: Abstract of paper
        :param is_open_access: Is paper open access yet
        :param pdf_url: URL of paper pdf
        :param download_status: HTTP status of download
        :param download_error_msg: Error message for download
        :param grobid_status: HTTP status of grobid process
        :param grobid_error_msg: Error message for grobid
        :param time_grobid_processed: Time processed with grobid
        :param time_added: Time initially added
        """
        # add properties
        properties: Dict[str, Any] = {
            'id': title,
            'openalex_id': openalex_id,
            'doi': doi,
            'abstract_text': abstract_text,
            'is_open_access': is_open_access,
            'pdf_url': pdf_url,
            'download_status': download_status,
            'download_error_msg': download_error_msg,
            'grobid_status': grobid_status,
            'grobid_error_msg': grobid_error_msg,
            'time_grobid_processed': time_grobid_processed,
            'time_added': time_added
        }

        # calculate embedding if abstract available
        if abstract_text:
            abstract_embedding = self._embedding_model.encode(abstract_text, show_progress_bar=False).tolist()
            properties['abstract_embedding'] = abstract_embedding

        # insert node
        paper_node = Node.create(NodeType.PAPER, properties)
        is_new_node = self.insert_node(paper_node, True)  # update matches, don't replace existing fields

        # add relationship to current run if new node and run id provided
        if run_id and is_new_node:
            run_node = Node.create(NodeType.RUN, {'id': run_id})
            self.insert_relationship(run_node,
                                     run_node.create_relationship_to(paper_node.type, RelationshipType.ADDED),
                                     paper_node)

    def insert_paper_batch(self, run_id: int, paper_properties: List[Dict[str, str]]) -> None:
        """
        Insert a batch of papers into the database

        todo - cleanup

        :param run_id: ID of run this batch of papers was found in
        :param paper_properties: Node properties
        """
        # ensure open connection
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")

        # convert to nodes
        paper_nodes: List[Node] = [Node.create(NodeType.PAPER, props) for props in paper_properties]

        # add properties
        set_expressions = set()
        if paper_nodes[0].required_properties:
            set_expressions.update([f"n.{k} = paper.{k}" for k in paper_nodes[0].required_properties])
        if paper_nodes[0].properties:
            set_expressions.update([f"n.{k} = paper.{k}" for k in paper_nodes[0].properties])

        set_clause = ", ".join(set_expressions)
        query = f"""
        MERGE (run:{NodeType.RUN.value} {{id: $run_id}})
        WITH run
        UNWIND $papers AS paper
        MERGE (n:{NodeType.PAPER.value} {{match_id: paper.match_id}})
        ON CREATE SET {set_clause} ON MATCH SET {set_clause}
        MERGE (run)-[:{RelationshipType.ADDED.value}]->(n)
        """

        # batch insert
        with self._driver.session() as session:
            session.run(query, run_id=run_id,
                        papers=[{'match_id': node.match_id, **node.required_properties, **node.properties} for node in
                                paper_nodes])

    def insert_citation_papers(self, source_title: str, citations: List[Dict[str, str]]) -> None:
        """
        Batch insert a list of citation papers found by grobid

        todo - cleanup

        :param source_title: Title of paper that cites these papers
        :param citations: List of paper properties
        """
        # ensure open connection
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")

        # convert to nodes
        paper_nodes: List[Node] = [Node.create(NodeType.PAPER, props) for props in citations]

        # add properties
        set_expressions = set()
        all_props = set()
        if paper_nodes[0].required_properties:
            all_props.update(paper_nodes[0].required_properties)
        if paper_nodes[0].properties:
            all_props.update(paper_nodes[0].properties)

        # build the SET expressions
        set_expressions.update([f"n.{k} = coalesce(n.{k}, paper.{k})" for k in all_props])
        set_clause = ", ".join(set_expressions)

        # construct the final query
        query = f"""
        MERGE (source:{NodeType.PAPER.value} {{id: $source_id}})
        WITH source
        UNWIND $papers AS paper
        MERGE (n:{NodeType.PAPER.value} {{match_id: paper.match_id}})
        ON CREATE SET {set_clause} ON MATCH SET {set_clause}
        MERGE (source)-[:{RelationshipType.REFERENCES.value}]->(n)
        """

        # batch insert
        with self._driver.session() as session:
            session.run(query,
                        source_id=source_title,
                        papers=[{'match_id': node.match_id, **node.required_properties, **node.properties} for node in
                                paper_nodes])

    def get_all_unprocessed_pdf_urls(self) -> List[Tuple[str, str]]:
        """
        Get all papers with pdfs that haven't been processed by grobid yet

        :return: List of paper titles and pdf urls
        """
        query = f"""
            MATCH (p:{NodeType.PAPER.value}) 
            WHERE p.pdf_url IS NOT NULL 
            AND p.grobid_status IS NULL 
            AND p.is_open_access 
            RETURN p.id AS id, p.pdf_url AS pdf_url
            """
        # todo - add time filters
        with self._driver.session() as session:
            results = session.run(query)
            return [(r['id'], r['pdf_url']) for r in results]


def _is_model_local(embedding_model: str) -> bool:
    """
    Util method to check if the embedding model is downloaded locally

    :param embedding_model: Name of sentence transformer embedding model to use
    :return: True if downloaded, false otherwise
    """
    cache_dir = os.path.expanduser(f"~/{SENTENCE_TRANSFORMER_CACHE}")
    model_path = os.path.join(cache_dir, f"models--sentence-transformers--{embedding_model}")
    return os.path.exists(model_path)
