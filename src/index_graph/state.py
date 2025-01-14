from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated, List
from langchain_core.documents import Document

@dataclass(kw_only=True)
class IndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed.
    """

    docs: Annotated[List[Document], "A list of documents that the agent can index."]