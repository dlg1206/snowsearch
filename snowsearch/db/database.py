import logging
import os
import time
from abc import ABC

from neo4j import GraphDatabase
from neo4j.exceptions import ConstraintError, TransientError

from db import _NODE_SCHEMA
from db.entity import Node, Relationship, NodeType
from util.logger import logger

"""
File: database.py
Description: neo4j database interface for handling threat actor data

@author Derek Garcia
"""

# Mute Neo4j driver logs
logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.backend").setLevel(logging.CRITICAL)

DDL_PATH = f"{os.path.dirname(__file__)}/ddl/schema.yaml"
DEFAULT_USER = "neo4j"
DEFAULT_BOLT_URI = "bolt://localhost"
DEFAULT_BOLT_PORT = "7687"

MAX_ATTEMPTS = 3  # attempt to insert 3 times in case of deadlock


class Neo4jDatabase(ABC):
    """
    Generic interface for accessing a Neo4j Database
    """

    def __init__(self):
        """
        Create interface to neo4j db.

        'BOLT_URI' env variable is used (Default: bolt://localhost)
        'BOLT_PORT' env variable is used (Default: 7687)
        """
        self._uri = f"{os.getenv('BOLT_URI', DEFAULT_BOLT_URI)}:{os.getenv('BOLT_PORT', DEFAULT_BOLT_PORT)}"
        self._driver = None

    def _verify_connection(self) -> None:
        """
        Verify that a connection to the Neo4j database can be established.

        :raises RuntimeError: If connection was never established
        :raises ConnectionError: If failed to connect to the Neo4j database
        """
        logger.debug_msg("Attempting to access database")
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")
        # attempt connection
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            raise ConnectionError(e)
        logger.debug_msg("Connected Successfully")

    def __enter__(self) -> "Neo4jDatabase":
        """
        Open database connection

        :return: Neo4jDatabase with open connection
        """
        username, password = os.getenv('NEO4J_AUTH').split('/', 1)
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(username, password)
        )
        self._verify_connection()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Close database connection

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        """
        if self._driver:
            self._driver.close()
            self._driver = None

    def init(self) -> None:
        """
        Load constraints from the schema yaml file
        """
        logger.debug_msg("Initializing database")
        with self._driver.session() as session:
            # enforce for unique keys
            for node_label in _NODE_SCHEMA.keys():
                # Create the constraint query dynamically
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_label}) REQUIRE n.match_id IS UNIQUE;")
        logger.debug_msg("Database initialized")

    def has(self, node_type: NodeType, node_id: str) -> bool:
        """
        Check if the given node id exists in the database

        :param node_type: Node type to search for
        :param node_id: id to search for
        :raises RuntimeError: If attempt to insert using a closed connection
        :return: True if node exists, False otherwise
        """
        # ensure open connection
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")
        # build query
        query = f"MATCH (n:{node_type.value}) WHERE n.id = $node_id RETURN count(n) > 0 AS exists"
        # exe and return results
        with self._driver.session() as session:
            result = session.run(query, node_id=node_id)
            return result.single()["exists"]

    def insert_node(self, node: Node, update: bool = False, replace: bool = False) -> bool:
        """
        Insert a node into the neo4j database

        :param node: Node to insert into the database
        :param update: Update the data if match is found (Default: False)
        :param replace: If updating, replace existing data (Default: False)
        :raises RuntimeError: If attempt to insert using a closed connection
        :return True if inserted or updated, False if not
        """
        # ensure open connection
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")

        # always use match id
        query = f"{'MERGE' if update else 'CREATE'} (n:{node.type.value} {{match_id: '{node.match_id}'}})"

        # add properties
        set_expressions = set()
        # replace value if set
        statement = "n.{k} = ${k}" if replace else "n.{k} = coalesce(n.{k}, ${k})"
        all_props = set()
        if node.required_properties:
            all_props.update(node.required_properties)
        if node.properties:
            all_props.update(node.properties)

        # build the SET expressions
        set_expressions.update([statement.format(k=k) for k in all_props])
        set_clause = ", ".join(set_expressions)

        # construct the final query
        query = f"{query} ON CREATE SET {set_clause} ON MATCH SET {set_clause}" if update else f"{query} SET {set_clause}"

        # execute query
        for attempt in range(MAX_ATTEMPTS):
            try:
                with self._driver.session() as session:
                    session.run(query, parameters={**node.required_properties, **node.properties})
                return True
            except ConstraintError:
                # already inserted
                return False
            except TransientError as e:
                if "DeadlockDetected" in str(e):
                    time.sleep(0.1 * (attempt + 1))  # backoff
                else:
                    raise  # re-raise other transient errors

    def insert_relationship(self, start_node: Node, relationship: Relationship, end_node: Node) -> None:
        """
        Insert a relationship into the neo4j database. Will insert nodes to ensure they exist

        :param start_node: Start node
        :param relationship: Relationship between nodes
        :param end_node: End node
        :raises RuntimeError: If attempt to insert using a closed connection
        """
        # ensure open connection
        if not self._driver:
            raise RuntimeError("Database driver is not initialized")

        # ensure nodes exist
        self.insert_node(start_node)
        self.insert_node(end_node)

        # match by uids
        start_match = "{" + ", ".join([f"{k}: $start_{k}" for k in start_node.required_properties]) + "}"
        end_match = "{" + ", ".join([f"{k}: $end_{k}" for k in end_node.required_properties]) + "}"

        # add relationship properties
        rel_set_clause = ""
        if relationship.properties:
            rel_set_clause = "SET " + ", ".join([f"r.`{k}` = $rel_{k}" for k in relationship.properties])

        # generate query
        query = f"""
        MATCH (a:{start_node.type.value} {start_match})
        MATCH (b:{end_node.type.value} {end_match})
        MERGE (a)-[r:{relationship.type.value}]->(b)
        {rel_set_clause}
        """

        # generate params
        parameters = {
            **{f"start_{k}": v for k, v in start_node.required_properties.items()},
            **{f"end_{k}": v for k, v in end_node.required_properties.items()},
            **{f"rel_{k}": v for k, v in relationship.properties.items()},
        }

        # execute query
        with self._driver.session() as session:
            session.run(query, parameters)
