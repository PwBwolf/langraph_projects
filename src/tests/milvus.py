from random import random
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

class MilvusHandler:
    def __init__(self, host="10.132.23.198", port="19530"):
        self.host = host
        self.port = port
        self.alias = "default"

    def connect(self):
        """Establish a connection to Milvus."""
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")

    def create_collection(self, collection_name):
        """Create a collection in Milvus."""
        field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        schema = CollectionSchema(fields=[field1, field2], description="Test collection")
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")
        return collection

    def insert_data(self, collection_name, embeddings):
        """Insert embeddings into the collection."""
        collection = Collection(collection_name)
        ids = collection.insert([embeddings])
        print(f"Inserted {len(embeddings)} records into '{collection_name}'.")
        return ids

    def search(self, collection_name, query_vectors, top_k):
        """Search for similar vectors."""
        collection = Collection(collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=None
        )
        return results
    
def test_milvus():
    # Initialize the handler
    handler = MilvusHandler(host="10.132.23.198", port="19530")
    
    # Connect to Milvus
    handler.connect()
    
    # Create a collection
    collection_name = "test_collection"
    handler.create_collection(collection_name)
    
    # Generate dummy data (10 random vectors with dimension 128)
    embedding_data = [[random() for _ in range(128)] for _ in range(10)]
    
    # Insert data
    handler.insert_data(collection_name, embedding_data)
    
    # Search for similar vectors
    query_vector = [random() for _ in range(128)]  # Single query vector
    results = handler.search(collection_name, [query_vector], top_k=3)
    
    # Display results
    print("Search results:")
    for result in results:
        for hit in result:
            print(f"ID: {hit.id}, Distance: {hit.distance}")

if __name__ == "__main__":
    test_milvus()