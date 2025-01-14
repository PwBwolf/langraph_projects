from pydantic import BaseModel
from typing import List, Optional

class DocumentRequest(BaseModel):
    documents: List[str]

class IndexResponse(BaseModel):
    message: str
    documents_indexed: int

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int]

class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    generated_answer: str

class HealthResponse(BaseModel):
    status: str
    vector_db: str
    embedding_model: str