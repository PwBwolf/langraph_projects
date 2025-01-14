"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, fields, field
from typing import Optional
from typing import Annotated
from langchain_core.runnables import RunnableConfig
from src.agent import prompts
from src.shared.configuration import BaseConfiguration 

@dataclass(kw_only=True)
class Configuration(BaseConfiguration):
    """The configuration for the agent."""
    # models
    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-3.5-turbo-0125",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-3.5-turbo-0125",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    vector_output_fields: Annotated[str, {"__template_metadata__": {"kind": "text"}}] = field(
        default="summary",
        metadata={
            "description": "The fields to retrieve from the Milvus search results."
        },
    )
    # prompts
    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for classifying user questions to route them to the correct node."
        },
    )

    grader_system_prompt: str = field(
        default=prompts.GRADER_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for grading the relevance of a document to a user question."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for generating responses to user questions."
        },
    )
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
