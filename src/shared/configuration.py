# base_configuration.py

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Literal, Optional, Type, TypeVar, Any
from langchain_core.runnables import RunnableConfig, ensure_config

T = TypeVar("T", bound="BaseConfiguration")

@dataclass(kw_only=True)
class BaseConfiguration:
    """
    Base configuration class holding common parameters and methods
    that can be inherited by other configuration classes.
    """
        # Fetch sensitive info from environment variables
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "default_collection")
    vector_dim: int = 1536
    
    embedding_model: str = field(
        default="openai/text-embedding-3-small",
        metadata={"description": "Name of the embedding model to use."},
    )
    search_params: dict = field(
        default_factory=lambda: {"metric_type": "L2", "nprobe": 10},
        metadata={"description": "Search parameters for vector database queries."},
    )

    retriever_provider: Annotated[
        Literal["mongodb", "elastic", "pinecone", "other_provider"],
        {"__template_metadata__": {"kind": "retriever"}}
    ] = field(
        default="mongodb",
        metadata={"description": "Default retriever provider used by derived configurations."},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        
        kwargs = {k: v for k, v in configurable.items() if k in _fields}
        return cls(**kwargs)