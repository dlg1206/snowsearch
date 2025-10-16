import os
import warnings
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

from sentence_transformers import SentenceTransformer

from client.grobid.dto import CitationDTO
from client.openalex.dto import PaperDTO
from db.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_DIMENSIONS, SENTENCE_TRANSFORMER_CACHE, DOI_PREFIX
from db.database import Neo4jDatabase
from db.entity import Node, NodeType, RelationshipType
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
        Insert a batch of papers found by an OpenAlex run

        :param run_id: ID of run this batch of papers was found in
        :param hits: List of papers and their OpenAlex search ranking
        """
        # wrapper to keep relationship logic internal

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

    def insert_citation_paper_batch(self, source_title: str, papers: List[PaperDTO]) -> None:
        """
        Insert a batch of papers cited by a source paper

        :param source_title: Title of paper that cites these papers
        :param papers: List of papers
        """
        # wrapper to keep relationship logic internal
        self._insert_paper_batch(Node.create(NodeType.PAPER, {'id': source_title}), RelationshipType.REFERENCES, papers)

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
                        papers=[{'match_id': node.match_id, **node.required_properties, **node.properties} for node in
                                paper_nodes])

    def get_unprocessed_pdf_urls(self, run_id: int = None, paper_limit: int = None) -> List[PaperDTO]:
        """
        Get all papers with pdfs that haven't been processed by grobid yet
        If the run_id is provided, the list of papers are returned in order to most to least
        relevant determined by original openalex query

        :param run_id: Optional run the paper was discovered in (Default: None)
        :param paper_limit: Limit the max number of papers to return (Default: None)
        :return: List of paper titles and pdf urls
        """
        # base query
        query = f"""
            WHERE p.pdf_url IS NOT NULL 
            AND p.download_status IS NULL 
            AND p.grobid_status IS NULL 
            AND p.is_open_access 
            RETURN p.id AS id, p.pdf_url AS pdf_url
            """

        # order by run rank if provided
        prefix = f"MATCH (run:{NodeType.RUN.value} {{id: $run_id}})-[r:{RelationshipType.ADDED.value}]->(p:{NodeType.PAPER.value})" if run_id else f"MATCH (p:{NodeType.PAPER.value})"
        query = f"{prefix} {query}"
        if run_id:
            query = f"{query} ORDER BY r.rank"

        # add limit if provided
        if paper_limit:
            query = f"{query} LIMIT {paper_limit}"

        # exe query
        with self._driver.session() as session:
            results = session.run(query, run_id=run_id) if run_id else session.run(query)
            return [PaperDTO(r['id'], pdf_url=r['pdf_url']) for r in results]

    def search_by_prompt_papers(self, prompt: str, paper_limit: int = 100, min_score: float = None) -> List[
        Tuple[PaperDTO, float]]:
        """
        Get papers with abstracts that best match the prompt
        The similarity score can range from 1 (exact match) and -1 (complete opposite match)

        :param prompt: Search prompt
        :param paper_limit: Limit the max number of papers to return (Default: 100)
        :param min_score: Minimum similarity score of prompt to abstract, must be [-1,1] (Default: None but 0.4 recommended)
        :raises ValueError: If provided min_score is outside [-1,1] range
        :return: List of top_k paper titles that best match the given prompt and their score
        """
        # validate min score
        if min_score and (min_score > 1 or min_score < -1):
            raise ValueError("Param 'min_score' must be between -1 and 1")

        prompt_embedding = self._embedding_model.encode(prompt, show_progress_bar=False).tolist()
        query = f"""
        CALL db.index.vector.queryNodes(
          'paper_abstract_index',
          $topK,
          $embedding
        ) YIELD node, score
        WHERE node.grobid_status = 200 {'AND score > $minScore' if min_score else ''}
        RETURN node.id AS id, score AS score
        ORDER BY score DESC
        """
        # set params
        params: Dict[str, Any] = {'topK': paper_limit, 'embedding': prompt_embedding}
        if min_score:
            params['minScore'] = min_score
        # exe query
        with self._driver.session() as session:
            results = session.run(query, **params)
            return [(PaperDTO(r['id']), r['score']) for r in results]

    def get_unprocessed_citations(self, source_title: str, top_k: int = None) -> List[CitationDTO]:
        """
        Get the most referenced citations for a given paper

        :param source_title: Title of paper that cites these papers
        :param top_k: Number of unprocessed citations to get, ranked total number of references (Default: All)
        :return: List of citation title, doi, and current citation count
        """
        query = f"""
        MATCH (s:{NodeType.PAPER.value})-[:{RelationshipType.REFERENCES.value}]->(c:{NodeType.PAPER.value})
        WHERE s.id = $source_title
        WITH c
        MATCH (any_paper:{NodeType.PAPER.value})-[:{RelationshipType.REFERENCES.value}]->(c)
        WHERE c.download_status IS NULL AND c.openalex_status IS NULL
        RETURN c.id AS id, c.doi AS doi, count(any_paper) AS citations
        ORDER BY citations DESC
        {'LIMIT $topK' if top_k else ''}
        """
        # set params
        params: Dict[str, Any] = {'source_title': source_title}
        if top_k:
            params['topK'] = top_k
        # exe query
        with self._driver.session() as session:
            results = session.run(query, **params)
            return [CitationDTO(r['id'], r['doi'], r['citations']) for r in results]


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
