import os
import warnings
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

from sentence_transformers import SentenceTransformer

from db.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_DIMENSIONS, SENTENCE_TRANSFORMER_CACHE, DOI_PREFIX
from db.database import Neo4jDatabase
from db.entity import Node, NodeType, RelationshipType
from dto.paper_dto import PaperDTO
from util.logger import logger
from util.timer import Timer

"""`
File: paper_database.py

Description: Specialized interface for abstracting Neo4j commands to the database

@author Derek Garcia
`"""

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
            logger.warn("Using cpu to create embeddings -- this may impact performance")

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
        # regular init
        super().init()
        # create embedding index for title and abstract
        queries = [
            f"""
            CREATE VECTOR INDEX paper_title_index IF NOT EXISTS
            FOR (p:{NodeType.PAPER.value}) ON (p.title_embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {self._model_dimensions},
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """,

            f"""
            CREATE VECTOR INDEX paper_abstract_index IF NOT EXISTS
            FOR (p:{NodeType.PAPER.value}) ON (p.abstract_embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {self._model_dimensions},
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """
        ]
        with self._driver.session() as session:
            for q in queries:
                session.run(q)

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

    def insert_openalex_query(self, run_id: int, model: str | None, nl_query: str | None, oa_query: str) -> None:
        """
        Update a run with the openalex prompt and resulting query

        todo add additional OpenAlex filters used
        https://docs.openalex.org/api-entities/works/filter-works#works-attribute-filters

        :param run_id: ID of run
        :param model: Model used to generate OpenAlex query
        :param nl_query: Original natural language prompt
        :param oa_query: Resulting OpenAlex query
        """
        run_node = Node.create(NodeType.RUN, {
            'id': run_id,
            'openalex_model': model,
            'openalex_query_input': nl_query,
            'openalex_query_output': oa_query
        })
        self.insert_node(run_node, True)

    def upsert_paper(self, paper: PaperDTO, run_id: int = None) -> None:
        """
        Insert paper into database and optional details

        :param paper: DTO with paper details
        :param run_id: Optional ID of run (Default: None)
        """
        # add properties
        properties = asdict(paper)

        # ensure just DOI id
        if paper.doi:
            properties['doi'] = paper.doi.removeprefix(DOI_PREFIX)

        # calculate embedding if abstract available
        if paper.abstract_text:
            abstract_embedding = self._embedding_model.encode(paper.abstract_text, show_progress_bar=False).tolist()
            properties['abstract_embedding'] = abstract_embedding

        # calculate embedding if title available
        if paper.id:
            abstract_embedding = self._embedding_model.encode(paper.id, show_progress_bar=False).tolist()
            properties['title_embedding'] = abstract_embedding

        # insert node
        paper_node = Node.create(NodeType.PAPER, properties)
        is_new_node = self.insert_node(paper_node, True)  # update matches, don't replace existing fields

        # add relationship to current run if new node and run id provided
        if run_id and is_new_node:
            run_node = Node.create(NodeType.RUN, {'id': run_id})
            self.insert_relationship(run_node,
                                     run_node.create_relationship_to(paper_node.type, RelationshipType.ADDED),
                                     paper_node)

    def insert_run_paper_batch(self, run_id: int, hits: List[Tuple[PaperDTO, int]]) -> None:
        """
        Insert a batch of papers found by an OpenAlex run and generate title embeddings for new papers

        :param run_id: ID of run this batch of papers was found in
        :param hits: List of papers and their OpenAlex search ranking
        """
        # get match ids
        match_ids, dtos = [], []
        for p, _ in hits:
            match_ids.append(Node.create(NodeType.PAPER, asdict(p)).match_id)
            dtos.append(p)
        # insert nodes
        self._insert_paper_batch(Node.create(NodeType.RUN, {'id': run_id}), RelationshipType.ADDED, dtos)

        # add ranking
        ranked_papers = [
            {
                "match_id": match_ids[i],
                "rank": rank
            }
            for i, (_, rank) in enumerate(hits)
        ]
        query = f"""
        UNWIND $ranked_papers AS item
        MATCH (run:{NodeType.RUN.value} {{id: $run_id}})
        MATCH (paper:{NodeType.PAPER.value} {{match_id: item.match_id}})
        MERGE (run)-[r:{RelationshipType.ADDED.value}]->(paper)
        SET r.rank = item.rank
        """
        with self._driver.session() as session:
            session.run(query, run_id=run_id, ranked_papers=ranked_papers)

        # update title embeddings
        self._update_missing_title_embeddings()

    def _update_missing_title_embeddings(self) -> None:
        """
        Fetch all papers that are missing title embeddings and calculate and set value
        """

        missing_title_embeddings_query = f"""
        MATCH (p:{NodeType.PAPER.value})
        WHERE p.title_embedding IS NULL
        RETURN p.match_id AS match_id, p.id AS id
        """
        # fetch titles with missing embeddings
        with self._driver.session() as session:
            result = session.run(missing_title_embeddings_query)
            missing = [r.data() for r in result]

        # exit early if nothing to update
        if not missing:
            return

        # else generate embeddings and update
        titles = [p['id'] for p in missing]
        title_embeddings = self._embedding_model.encode(titles, show_progress_bar=False).tolist()
        update_query = f"""
        UNWIND $rows AS row
        MATCH (p:{NodeType.PAPER.value} {{match_id: row.match_id}})
        SET p.title_embedding = row.embedding
        """
        rows = [
            {"match_id": paper["match_id"], "embedding": emb}
            for paper, emb in zip(missing, title_embeddings)
        ]
        with self._driver.session() as session:
            session.run(update_query, rows=rows)

    def insert_citation_paper_batch(self, source_title: str, papers: List[PaperDTO]) -> None:
        """
        Insert a batch of papers cited by a source paper

        :param source_title: Title of paper that cites these papers
        :param papers: List of papers
        """
        # wrapper to keep relationship logic internal
        self._insert_paper_batch(Node.create(NodeType.PAPER, {'id': source_title}), RelationshipType.REFERENCES, papers)

    def insert_paper_batch(self, papers: List[PaperDTO]) -> None:
        """
        Batch insert a list of papers

        :param papers: List of papers
        """
        # ensure open connection
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")

        # convert to nodes
        paper_nodes: List[Node] = [Node.create(NodeType.PAPER, asdict(p)) for p in papers]
        query = _format_paper_batch_insert_query(paper_nodes)

        # batch insert
        with self._driver.session() as session:
            session.run(query,
                        papers=[{'match_id': node.match_id, **node.required_properties, **node.properties}
                                for node in paper_nodes])

    def _insert_paper_batch(self,
                            source_node: Node,
                            rel_type: RelationshipType,
                            papers: List[PaperDTO]) -> None:
        """
        Batch insert a list of papers found by a source

        :param source_node: Source node that found this batch of papers
        :param rel_type: Relationship of source node to batch
        :param papers: List of papers
        """
        # ensure open connection
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")

        # convert to nodes
        paper_nodes: List[Node] = [Node.create(NodeType.PAPER, asdict(p)) for p in papers]
        query_body = _format_paper_batch_insert_query(paper_nodes)

        # construct the final query
        query = f"""
        MERGE (source:{source_node.type.value} {{match_id: $match_id}})
        WITH source
        {query_body}
        MERGE (source)-[:{rel_type.value}]->(n)
        """

        # batch insert
        with self._driver.session() as session:
            session.run(query,
                        match_id=source_node.match_id,
                        papers=[{'match_id': node.match_id, **node.required_properties, **node.properties}
                                for node in paper_nodes])

    def get_paper(self, title: str) -> PaperDTO | None:
        """
        Get a paper from the database
        
        :param title: Title of paper
        :return: Paper if found, None otherwise
        """
        # build query
        node = Node.create(NodeType.PAPER, {'id': title})
        query = f"MATCH (p:{node.type.value}) WHERE p.match_id = $match_id RETURN p"

        # exe query
        with self._driver.session() as session:
            record = session.run(query, match_id=node.match_id).single()
            # not found
            if not record:
                return None
            # convert to dto
            return PaperDTO.create_dto(record['p'])

    def get_papers(self, titles: List[str]) -> List[PaperDTO]:
        """
        Get papers from the database

        :param titles: List of paper title to fetch from the database
        :return: List of PaperDTOs
        """
        # build query
        match_ids = [Node.create(NodeType.PAPER, {'id': t}).match_id for t in titles]
        query = f"""
        UNWIND $match_ids AS id
        MATCH (p:{NodeType.PAPER.value} {{match_id: id}})
        RETURN p
        """
        # exe query
        with self._driver.session() as session:
            papers = [r['p'] for r in list(session.run(query, match_ids=match_ids))]
            # convert to dtos
            return [PaperDTO.create_dto(p) for p in papers]

    def get_unprocessed_papers(self, paper_limit: int = None) -> List[PaperDTO]:
        """
        Get unprocessed papers from the database

        :param paper_limit: Limit the max number of papers to return (Default: None)
        :return: List of PaperDTOs
        """
        # build query
        query = f"""
        MATCH (p:{NodeType.PAPER.value})
        WHERE p.pdf_url IS NOT NULL 
        AND p.download_status IS NULL 
        AND p.grobid_status IS NULL
        RETURN p{f' LIMIT {paper_limit}' if paper_limit else ''}
        """
        # exe query
        with self._driver.session() as session:
            papers = [r['p'] for r in list(session.run(query))]
            # convert to dtos
            return [PaperDTO.create_dto(p) for p in papers]

    def search_papers_by_nl_query(self, nl_query: str,
                                  unprocessed: bool = False,
                                  only_open_access: bool = False,
                                  require_abstract: bool = False,
                                  paper_limit: int = 100,
                                  min_score: float = None,
                                  order_by_abstract: bool = False,
                                  include_scores: bool = False) -> List[PaperDTO] | List[Tuple[PaperDTO, float, float]]:
        """
        Get papers that best match the provided query, ranked in order of best title match then abstract
        The similarity score can range from 1 (exact match) and -1 (complete opposite match)

        :param nl_query: Natural language to match papers to
        :param unprocessed: Only get papers that haven't been downloaded or processed with Grobid (Default: False)
        :param only_open_access: Only get papers that have an 'open access' label
        :param require_abstract: Require that paper has an abstract
        :param paper_limit: Limit the max number of papers to return (Default: 100)
        :param min_score: Minimum similarity score of prompt to abstract, must be [-1,1] (Default: None but 0.4 recommended)
        :param order_by_abstract: Return search order by abstract match then title match (Default: False)
        :param include_scores: Include the match score of nl_query (Default: False)
        :raises ValueError: If provided min_score is outside [-1,1] range
        :return: List of top_k papers ids ranked in order of best title match then abstract if available
        """
        # validate min score
        if min_score and (min_score > 1 or min_score < -1):
            raise ValueError("Param 'min_score' must be between -1 and 1")

        nl_query_embedding = self._embedding_model.encode(nl_query, show_progress_bar=False).tolist()

        # build where clause
        conditions = []
        if min_score is not None:
            conditions.append("titleScore >= $minScore AND (abstractScore IS NULL OR abstractScore >= $minScore)")
        if unprocessed:
            conditions.append(
                "node.pdf_url IS NOT NULL AND node.download_status IS NULL AND node.grobid_status IS NULL")
        if only_open_access:
            conditions.append("node.is_open_access")
        if require_abstract:
            conditions.append("node.abstract_embedding")
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        CALL db.index.vector.queryNodes(
          'paper_title_index',
          $topK,
          $embedding
        ) YIELD node AS tnode, score AS titleScore

        OPTIONAL CALL db.index.vector.queryNodes(
          'paper_abstract_index',
          $topK,
          $embedding
        ) YIELD node AS anode, score AS abstractScore
        WHERE anode.id = tnode.id

        WITH
          tnode AS node,
          titleScore,
          abstractScore
        {where_clause}
        RETURN
          node,
          titleScore,
          abstractScore
        ORDER BY {'abstractScore DESC, titleScore DESC' if order_by_abstract else 'titleScore DESC, abstractScore DESC'}
        LIMIT $paper_limit
        """
        # set params
        # topK large to ensure capture enough data to filter and limit
        params: Dict[str, Any] = {'topK': 100 * paper_limit,
                                  'embedding': nl_query_embedding,
                                  'minScore': min_score,
                                  'paper_limit': paper_limit}
        # exe query
        with self._driver.session() as session:
            results = session.run(query, **params)
            if include_scores:
                return [(PaperDTO.create_dto(r["node"]), r["titleScore"], r["abstractScore"]) for r in results]
            else:
                return [PaperDTO.create_dto(r["node"]) for r in results]

    def search_papers_by_title_match(self,
                                  search_term: str,
                                  only_open_access: bool = False,
                                  require_abstract: bool = False,
                                  paper_limit: int = None) -> List[PaperDTO]:
        """
        Get papers that best match the provided query, ranked in order of best title match then abstract
        The similarity score can range from 1 (exact match) and -1 (complete opposite match)

        :param search_term: Title keywords to attempt to match
        :param only_open_access: Only get papers that have an 'open access' label
        :param require_abstract: Require that paper has an abstract
        :param paper_limit: Limit the max number of papers to return (Default: 100)
        :raises ValueError: If provided min_score is outside [-1,1] range
        :return: List of top_k papers ids ranked in order of best title match then abstract if available
        """
        # build where clause
        conditions = [f"toLower(p.id) CONTAINS \"{search_term.lower()}\""]
        if only_open_access:
            conditions.append("p.is_open_access")
        if require_abstract:
            conditions.append("p.abstract_embedding")
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        MATCH (p:{NodeType.PAPER.value})
        {where_clause}
        RETURN p
        {f' LIMIT {paper_limit}' if paper_limit else ''}
        """
        # exe query
        with self._driver.session() as session:
            papers = [r['p'] for r in list(session.run(query))]
            # convert to dtos
            return [PaperDTO.create_dto(p) for p in papers]



    def get_citations(self, source_title: str, unprocessed: bool = False) -> List[PaperDTO]:
        """
        Get citations for a given paper

        :param source_title: Title of paper to get citations for
        :param unprocessed: Only get unprocessed citations (Default: False)
        :return: List of citations
        """
        query = f"""
        MATCH (s:{NodeType.PAPER.value})-[:{RelationshipType.REFERENCES.value}]->(c:{NodeType.PAPER.value})
        WHERE s.id = $source_title
        {'AND c.download_status IS NULL AND c.openalex_status IS NULL' if unprocessed else ''}
        RETURN c
        """
        # exe query
        with self._driver.session() as session:
            citations = [r['c'] for r in list(session.run(query, source_title=source_title))]
            # convert to dtos
            return [PaperDTO.create_dto(c) for c in citations]


def _is_model_local(embedding_model: str) -> bool:
    """
    Util method to check if the embedding model is downloaded locally

    :param embedding_model: Name of sentence transformer embedding model to use
    :return: True if downloaded, false otherwise
    """
    cache_dir = os.path.expanduser(f"~/{SENTENCE_TRANSFORMER_CACHE}")
    model_path = os.path.join(cache_dir, f"models--sentence-transformers--{embedding_model}")
    return os.path.exists(model_path)


def _format_paper_batch_insert_query(paper_nodes: List[Node]) -> str:
    """
    Format a list node paper nodes into cypher unwind query

    :param paper_nodes: List of paper nodes to add
    :return: UNWIND cypher query
    """
    # add properties
    all_props = set()
    if paper_nodes[0].required_properties:
        all_props.update(paper_nodes[0].required_properties)
    if paper_nodes[0].properties:
        all_props.update(paper_nodes[0].properties)

    # build the SET expressions
    set_expressions = {f"n.{k} = coalesce(n.{k}, paper.{k})" for k in all_props}
    set_clause = ", ".join(set_expressions)

    # construct the formatted query
    return f"""
        UNWIND $papers AS paper
        MERGE (n:{NodeType.PAPER.value} {{match_id: paper.match_id}})
        ON CREATE SET {set_clause} 
        ON MATCH SET {set_clause}
        """
