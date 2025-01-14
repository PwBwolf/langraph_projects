# src/index_graph.py
from __future__ import annotations
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from src.index_graph.configuration import IndexConfiguration
from src.index_graph.state import IndexState
# from src.shared.state import reduce_docs  # Assuming you have a reduce_docs utility
from src.services.milvus_handler import MilvusHandler
from src.services.embedding_handler import EmbeddingHandler

async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    print("Indexing documents...")
    # Load configuration
    configuration = IndexConfiguration.from_runnable_config(config)
    print("Loaded configuration:", configuration)
    milvus_handler = MilvusHandler(
        host=configuration.milvus_host,
        port=configuration.milvus_port
    )
    milvus_handler.connect()
    embedding_handler = EmbeddingHandler(model_name=configuration.embedding_model)
    print(state.docs)
      # Generate embeddings for documents
    embeddings = embedding_handler.generate_embeddings(state.docs)
    
    milvus_handler.insert_data(collection_name=configuration.milvus_collection, embeddings=embeddings)
    # Index embeddings in Milvus
    return {"docs": "no_docs_indexed"}

# Now build the graph
workflow = StateGraph(IndexState, config_schema=IndexConfiguration)
workflow.add_node(index_docs)
workflow.add_edge(START, "index_docs")
workflow.add_edge("index_docs", END)

graph = workflow.compile()
graph.name = "IndexGraph"
