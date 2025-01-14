from random import random
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from typing import List

class MilvusHandler:
    def __init__(self, host="127.0.0.1", port="19530"):
        self.host = host
        self.port = port
        self.alias = "default"

    def connect(self):
        """Establish a connection to Milvus."""
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")

    def create_collection(self, collection_name, vector_dim=128):
        """Create a collection in Milvus."""
        field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
        schema = CollectionSchema(fields=[field1, field2], description="Document collection")
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")
        return collection

    def insert_data(self, collection_name, embeddings):
        """Insert embeddings into the collection and handle Milvus response."""

        collection = Collection(name=collection_name)

        # Perform the insertion
        insert_response = collection.insert([embeddings])
        
        # Access the IDs from the MutationResult
        if hasattr(insert_response, "primary_keys"):
            inserted_ids = insert_response.primary_keys
            print(f"Successfully inserted {len(inserted_ids)} records into '{collection_name}'.")
            print(f"Milvus response (IDs of inserted records): {inserted_ids}")
        else:
            print(f"Failed to retrieve IDs from insert response. Raw response: {insert_response}")

        return insert_response

    def search(
        self, 
        collection_name: str, 
        query_vectors: List[List[float]], 
        top_k: int = 3, 
        output_fields: List[str] = None
    ):
        """
        Search for similar vectors, returning any 'output_fields' you want.
        Example: output_fields=["summary"] if you want the text from each doc.
        """
        collection = Collection(collection_name)
        collection.load()  # ensure data is loaded

        if output_fields is None:
            # By default, return only primary key & distance (no extra fields)
            output_fields = []

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )
        return results