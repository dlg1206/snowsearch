import warnings
from datetime import datetime

from sentence_transformers import SentenceTransformer

from db.database import Neo4jDatabase
from db.entity import Node, NodeType, RelationshipType
from util.logger import logger

"""
File: paper_database.py

Description: Specialized interface for abstracting Neo4j commands to the database

@author Derek Garcia
"""

# embedding model details
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS = 384

# suppress cuda warnings
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")


class PaperDatabase(Neo4jDatabase):
    def __init__(self):
        """
        Create new instance of the interface
        """
        super().__init__()
        # todo - add option in config?
        # use gpu if cuda available
        from torch.cuda import is_available
        if is_available():
            device = "cuda"
            logger.debug_msg("Embedding model utilizing gpu")
        else:
            device = "cpu"
            logger.warn("Using cpu to create abstract embeddings -- this may impact performance")
        self._embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, device=device)

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
            `vector.dimensions`: {DEFAULT_DIMENSIONS},
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

    def insert_findpapers_query(self, run_id: int, model: str, prompt: str, query: str) -> None:
        """
        Update a run with the findpapers prompt and resulting query

        :param run_id: ID of run
        :param model: Model used to generate findpapers query
        :param prompt: Original natural language prompt
        :param query: Resulting findpapers query
        """
        run_node = Node.create(NodeType.RUN, {
            'id': run_id,
            'findpapers_model': model,
            'findpapers_prompt': prompt,
            'findpapers_query': query
        })
        self.insert_node(run_node, True)

    def insert_paper(self, run_id: int, title: str, abstract: str) -> None:
        """
        Insert a paper into the database

        :param run_id: ID of run paper found
        :param title: Title of paper
        :param abstract: Abstract of paper
        """
        # todo other fields
        # add paper
        abstract_embedding = self._embedding_model.encode(abstract, show_progress_bar=False).tolist()
        paper_node = Node.create(NodeType.PAPER, {
            'id': title,
            'abstract_text': abstract,
            'abstract_embedding': abstract_embedding,
            'processed': False,
            'time_added': datetime.now()
        })
        self.insert_node(paper_node, True)

        # add relationship to current run
        run_node = Node.create(NodeType.RUN, {'id': run_id})
        self.insert_relationship(run_node,
                                 run_node.create_relationship_to(paper_node.type, RelationshipType.ADDED),
                                 paper_node)
