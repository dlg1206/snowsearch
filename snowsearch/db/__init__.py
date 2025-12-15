"""
File: __init__.py
Description:

@author Derek Garcia
"""

import os

import yaml

# Load the schema.yaml file when db is imported
# todo - not hardcoded path
_DDL_SCHEMA_PATH = f"{os.path.dirname(__file__)}/ddl/schema.yaml"

with open(_DDL_SCHEMA_PATH, 'r', encoding='utf-8') as f:
    full_schema = yaml.safe_load(f)
# todo - handle no defined nodes or relationships
_NODE_SCHEMA = full_schema['nodes']
_RELATIONSHIP_SCHEMA = full_schema['relationships']
