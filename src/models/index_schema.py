from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# -- Field: Primary key (auto-generated ID)
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True,
)

# -- Field: Embedding (1536-dimensional float vector)
embedding_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=1536
)

# -- Combine fields into a schema
schema = CollectionSchema(
    fields=[id_field, embedding_field],
    description="Storing only embeddings in Milvus"
)

def create_collection(collection_name: str) -> Collection:
    """Create a collection in Milvus."""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
    return collection