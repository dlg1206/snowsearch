import hashlib
from abc import ABC
from enum import Enum
from typing import List, Set, Dict, Any

from db import _NODE_SCHEMA, _RELATIONSHIP_SCHEMA

"""
File: entity.py
Description: Node construction and generation from yaml schema

@author Derek Garcia
"""


class NodeType(Enum):
    # todo nodes

    def get_relations_to(self, other_node_type: "NodeType") -> List["RelationshipType"]:
        """
        Get the relation to another node
        # todo - assumes only one type of relation to other node

        :param other_node_type: Other node to get the relation to
        :return: RelationType or None if none exists
        """
        node_schema = _NODE_SCHEMA.get(self.value)
        relationships = node_schema.get('relationships')
        # no relationships
        if not relationships:
            return []
        # check each relationship to see if one exists to the given node
        return [next((RelationshipType(k) for k, v in relationships.items() if other_node_type.value in v), None)]


class RelationshipType(Enum):
    # todo relationships
    pass


class MissingEntityError(ValueError):
    """
    Entity is not found in schema
    """

    def __init__(self, entity: NodeType | RelationshipType):
        """
        Entity is not found in schema

        :param entity: Entity attempted to create
        """
        super().__init__(f"'{entity.value}' is an unsupported entity")


class InvalidPropertyKeyError(KeyError):
    """
    One or more keys are not found in the schema
    """

    def __init__(self, entity: NodeType | RelationshipType, unknown_keys: List[str]):
        """
        One or more keys are not found in the schema

        :param entity: Entity accessed
        :param unknown_keys: List of unknown keys
        """
        # Construct the error message
        if len(unknown_keys) == 1:
            message = f"'{unknown_keys[0]}' is an unsupported key for entity '{entity.value}'"
        else:
            message = f"{', '.join(f"'{k}'" for k in unknown_keys)} are unsupported keys for entity '{entity.value}'"

        super().__init__(message)


class MissingRequiredPropertyKeyError(KeyError):
    """
    One or more required keys are missing
    """

    def __init__(self, entity: NodeType | RelationshipType, missing_keys: List[str]):
        """
        One or more required keys are missing

        :param entity: Entity accessed
        :param missing_keys: List of missing keys
        """
        # Construct the error message
        if len(missing_keys) == 1:
            message = f"'{missing_keys[0]}' required key is missing for entity '{entity.value}'"
        else:
            message = f"{', '.join(f"'{k}'" for k in missing_keys)} are required keys are missing for entity '{entity.value}'"

        super().__init__(message)


class NoRelationshipsError(ValueError):
    """
    Attempt to access a relationship of node with no relationships
    """

    def __init__(self, node: NodeType):
        """
        Attempt to access a relationship of node with no relationships

        :param node: Attempted node
        """
        super().__init__(f"'{node.value}' does not have relationships")


class MissingRelationshipError(KeyError):
    """
    Attempt to access a relationship of node that does not exist
    """

    def __init__(self, node: NodeType, relationship: RelationshipType):
        """
        Attempt to access a relationship of node that does not exist

        :param node: Attempted node
        :param relationship: Attempted relationship
        """
        super().__init__(f"'{node.value}' does not have relationship type of type '{relationship.value}'")


class InvalidRelationshipError(ValueError):
    """
    Attempt to create relationship to node that does not exist
    """

    def __init__(self, start_node: NodeType, relationship: RelationshipType, end_node: NodeType):
        """
        Attempt to create relationship to node that does not exist

        :param start_node: Attempted start node
        :param relationship: Attempted relationship
        :param end_node: Attempted end node
        """
        super().__init__(
            f"'{start_node.value}' does not have relationship type of type '{relationship.value}' to node '{end_node.value}'")


class Entity(ABC):
    def __init__(self, entity_type: NodeType | RelationshipType, properties: Dict[str, Any]):
        if isinstance(entity_type, NodeType):
            entity_schema = _NODE_SCHEMA.get(entity_type.value)
        else:
            entity_schema = _RELATIONSHIP_SCHEMA.get(entity_type.value)
        # entity has no properties - common with relationships
        if not entity_schema:
            entity_schema = {}
        # params are valid
        self._type: NodeType | RelationshipType = entity_type
        # split between required and properties
        required_property_keys = set(entity_schema.get('required_properties', []))
        self._required_properties: Dict[str, str | int] = {k: properties.get(k) for k in required_property_keys}
        self._properties: Dict[str, Any] = {k: v for k, v in properties.items() if k not in required_property_keys}
        # build a semi unique id for matching
        match_id = ""
        for key in sorted(self._required_properties):
            value = self._required_properties[key]
            match_id += str(value)
        self._match_id = hashlib.md5(match_id.lower().encode('utf-8')).hexdigest()

    @classmethod
    def _create(cls, entity_type: NodeType | RelationshipType, properties: Dict[str, Any] = None) -> "Entity":
        """
        Create new instance of an entity with properties

        :param entity_type: Entity that this object is an instance of
        :param properties: Optional Properties of the node
        """
        _validate_entity(entity_type, set(properties.keys()) if properties else set())
        return cls(entity_type, properties)

    @property
    def match_id(self):
        return self._match_id

    @property
    def type(self):
        return self._type

    @property
    def required_properties(self):
        return self._required_properties

    @property
    def properties(self):
        return self._properties


class Relationship(Entity):
    def __init__(self, relationship_type: RelationshipType, properties: Dict[str, Any]):
        super().__init__(relationship_type, properties)
        # save the inverse if one is defined
        entity_schema = _RELATIONSHIP_SCHEMA.get(relationship_type.value, {})
        # get inverse if properties
        if entity_schema:
            inverse_name = entity_schema.get('inverse', None)
            self._inverse_rel_type = RelationshipType(inverse_name) if inverse_name else None
        else:
            self._inverse_rel_type = None

    @classmethod
    def create(cls, start_node_type: NodeType,
               rel_type: RelationshipType,
               end_node_type: NodeType,
               properties: Dict[str, Any] = None) -> "Relationship":
        """
        Create new instance of a relationship with properties

        :param start_node_type: Starting Node type
        :param rel_type: Relationship type
        :param end_node_type: Ending Node type
        :param properties: Properties of the relationship
        """
        _validate_relationship(start_node_type, rel_type, end_node_type)
        return cls._create(rel_type, properties or {})

    @property
    def inverse_rel_type(self):
        return self._inverse_rel_type


class Node(Entity):
    def __init__(self, node_type: NodeType, properties: Dict[str, Any]):
        """
        Create new instance of a node with properties

        :param node_type: Node that this object is an instance of
        :param properties: Properties of the node
        """
        super().__init__(node_type, properties)

    @classmethod
    def create(cls, node_type: NodeType, properties: Dict[str, Any] = None) -> "Node":
        """
        Factory method to create a node using the shared _create method from Entity.

        :param node_type: Node that this object is an instance of
        :param properties: Optional Properties of the node
        :return: A new Node instance
        """
        return cls._create(node_type, properties)

    def create_relationship_to(self,
                               other_node: NodeType,
                               rel_type: RelationshipType,
                               rel_properties: Dict[str, Any] = None) -> Relationship:
        """
        Attempt to create a relationship to another node, will raise an error if invalid

        :param other_node: Other node type to get the relationship to
        :param rel_type: Type of relationship to make
        :param rel_properties: Optional properties of the relationship
        """
        return Relationship.create(self._type, rel_type, other_node, rel_properties)


def _validate_entity(entity_type: NodeType | RelationshipType, property_keys: Set[str]) -> None:
    """
    Validate the entry has data matching the schema

    :param entity_type: Type of entity to attempt to construct
    :param property_keys: set of keys being used
    :raises InvalidEntityError: If the given entity doesn't have a defined schema
    :raises MissingRequiredPropertyKeyError: If one or more required keys are missing
    :raises InvalidPropertyKeyError: If one or more property keys are not found in the schema
    """
    schema_dict = _NODE_SCHEMA if isinstance(entity_type, NodeType) else _RELATIONSHIP_SCHEMA
    # verify the schema exists
    if entity_type.value not in schema_dict:
        raise MissingEntityError(entity_type)

    entity_schema = schema_dict.get(entity_type.value)
    # if no properties to verify, return true
    if not entity_schema:
        return
    # verify all required keys are present
    required_properties = entity_schema.get('required_properties')
    if required_properties:
        missing_keys = list(set(required_properties) - property_keys)
        if missing_keys:
            raise MissingRequiredPropertyKeyError(entity_type, missing_keys)

    # verify provided keys have no unknown keys
    properties = entity_schema.get('properties')
    if properties:
        # remove accepted and required
        unknown_keys = list(property_keys - set(properties) - set(entity_schema.get('required_properties', {})))
        if unknown_keys:
            raise InvalidPropertyKeyError(entity_type, unknown_keys)
    # entity is valid


def _validate_relationship(start_node_type: NodeType, relationship_type: RelationshipType,
                           end_node_type: NodeType) -> None:
    """
    Validate the relationship between two node types

    :param start_node_type: Starting Node type
    :param relationship_type: Relationship type
    :param end_node_type: Ending Node type
    :raises NoRelationshipsError: If the starting node doesn't have any relationships
    :raises MissingRelationshipError: If the starting node doesn't have the requested relationship
    :raises InvalidRelationshipError: If the requested relationship doesn't exist between the start and end node
    """
    # verify the staring node schema exists
    node_start_schema = _NODE_SCHEMA.get(start_node_type.value)
    if not node_start_schema:
        raise MissingEntityError(start_node_type)

    # verify the end schema exists
    if not _NODE_SCHEMA.get(end_node_type.value):
        raise MissingEntityError(end_node_type)

    # verify starting node has relationships
    relationships = node_start_schema.get('relationships')
    if not relationships:
        raise NoRelationshipsError(start_node_type)

    # verify this node has the provided relationship type
    if relationship_type.value not in relationships:
        raise MissingRelationshipError(start_node_type, relationship_type)

    # verify this node has the provided relationship type with the end node
    if end_node_type.value not in relationships.get(relationship_type.value):
        raise InvalidRelationshipError(start_node_type, relationship_type, end_node_type)
    # relationship exists