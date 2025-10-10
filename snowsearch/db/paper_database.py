from db.database import Neo4jDatabase

"""
File: paper_database.py

Description: Specialized interface for abstracting Neo4j commands to the database

@author Derek Garcia
"""


class PaperDatabase(Neo4jDatabase):
    def __init__(self):
        """
        Create new instance of the interface
        """
        super().__init__()
