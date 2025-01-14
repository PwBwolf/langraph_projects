#!/usr/bin/env python3
from dotenv import load_dotenv
import os

# Load environment variables before any other imports
load_dotenv()

from pymilvus import connections, Collection
from src.services.milvus_handler import MilvusHandler
from src.services.embedding_handler import EmbeddingHandler


collection = os.getenv("MILVUS_COLLECTION")
#######################################
# 1. CONNECT TO MILVUS
#######################################
milvus_handler = MilvusHandler(
    host=os.getenv("MILVUS_HOST"),
    port=os.getenv("MILVUS_PORT")
)

embedding_handler = EmbeddingHandler(
    model_name=os.getenv("EMBEDDING_MODEL") or "openai/text-embedding-3-small"
)

question = "How is data stored in milvus?"

# You must pass a *list* of strings.
embedded_question = embedding_handler.generate_embeddings([question])

milvus_handler.connect()

res = milvus_handler.search(collection_name=collection, query_vectors=embedded_question, top_k=3)
 
for hits in res:
    for hit in hits:
        print(hit)