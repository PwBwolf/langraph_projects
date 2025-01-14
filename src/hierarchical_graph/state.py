"""Define the state structures for the agent."""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

@dataclass(kw_only=True)
class InputState:
    messages: Annotated[list[AnyMessage], add_messages]

@dataclass(kw_only=True)
class NodeResponse:
    result: Optional[str] = None  
    status: str = "pending"       
    metadata: Optional[dict] = None  

@dataclass(kw_only=True)
class AgentState(InputState):
    search_response: NodeResponse = field(default_factory=NodeResponse)
    web_scraper_response: NodeResponse = field(default_factory=NodeResponse)
    supervisor_response: Optional[str] = None 

