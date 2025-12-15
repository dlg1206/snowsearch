# Working with Neo4j Schema

Since graph databases are unstructured, I wrote a framework to ensure some consistency.

#### The Schema

I've created a "[schema.yaml](../snowsearch/db/ddl/schema.yaml)" file that is the main framework. The main
entity structure is like so:

```yaml
Entity_Name:
  properties:
    - "prop_1"
    ...
    - "prop_n"
```

- `Entity_Name` (REQUIRED): Unique name of the Node or Relationship in the database
- `properties` (optional): List of additional properties for this entity

There are two types of entities: **Nodes** (objects) and **Relationships** (pointers between objects).
Nodes have a few additional fields:

```yaml
...
required_properties:
  - "prop_1"
  ...
  - "prop_n"
relationships:
  Relationship_Name:
    - "Node_Name_1"
    ...
    - "Node_Name_n"
```

- `required_properties` (REQUIRED): List of required properties for this node. Each must be unique to that node, ie uid.
  This is used for matching later. As of now best practice is just to use 1 key
- `relationships` (optional): List of relationships that this node has with other nodes.
- `Relationship_Name`: Name of the relationship this Node has. Name MUST be defined as a key in the
  overall `relationships` object.
    - The list of nodes are the ending node of this relationship. Name MUST be defined as a key in the overall `nodes`
      object.

> [!WARNING]  
> The following is still supported, but discouraged to prevent redundant relationships

Relationships can have one additional optional keyword: `inverse`

```yaml
inverse: "Relationship_Name"
```

- `inverses` (optional): The inverse relationship to this one. The inverse relationship MUST have been defined.

Finally, the list of Nodes and Relationships are recorded under their respective keys words:

```yaml
nodes:
  ...
relationships:
  ...
```

#### Updating the Schema Workflow

1. Update the "ddl" - make the desired changes to the [schema.yaml](../snowsearch/db/ddl/schema.yaml)
2. If adding any new Nodes or relationships (as opposed to additional properties), update the respective enum
   in [entity.py](../snowsearch/db/entity.py) with the value matching the keyword name in the yaml file

And that's it!

If you do modify or make changes, be kind to future devs and update the docs :)