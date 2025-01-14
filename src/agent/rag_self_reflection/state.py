from typing import List, Literal
from dataclasses import dataclass, field
from typing import Annotated
from langchain_core.documents import Document


@dataclass(kw_only=True)
class QueryState:
    """Private state for the retrieve_documents node in the researcher graph."""
    query: str

@dataclass(kw_only=True)
class Grader:
    type: Literal["yes", "no"]
    logic: str = ""

@dataclass(kw_only=True)
class RewriterResponse:
    rewritten_question: str
    reasoning: str = ""

@dataclass(kw_only=True)
class ResearcherState:    
    question: str
    generation: list[str] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    iteration_count: int = 0  
    max_iterations: int = 5 