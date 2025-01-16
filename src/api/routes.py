from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import DocumentRequest, IndexResponse, QueryRequest, QueryResponse, HealthResponse
from src.api.dependencies import get_index_graph, get_rag_graph
from src.index_graph.state import IndexState
from langchain_core.runnables import RunnableConfig

router = APIRouter()

@router.post("/index", response_model=IndexResponse)
async def index_documents(
    request: DocumentRequest,
    graph=Depends(get_index_graph),
):
    """API to index documents using the LangGraph workflow."""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    # Initialize the graph state
    state = IndexState(docs=request.documents)

    # Pass runtime configuration as a RunnableConfig
    config = RunnableConfig(configurable={
        "vector_dim": 1536,
        "embedding_model": "openai/text-embedding-3-small",
        "milvus_collection": "simple_embedding"
    })

    # Execute the graph
    result = await graph.ainvoke(state, config=config)

    print(result)
    return IndexResponse(
        message="Indexing complete",
        documents_indexed=len(request.documents)
    )

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    graph=Depends(get_rag_graph),
):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required.")

    # Mocked response
    mocked_results = [
        {
            "document": "Life is a journey, not a destination.",
            "score": 0.95
        },
        {
            "document": "The meaning of life depends on your perspective.",
            "score": 0.89
        }
    ]
    mocked_generated_answer = "The meaning of life is subjective and varies based on personal values and beliefs."

    return QueryResponse(
        query=request.query,
        results=mocked_results,
        generated_answer=mocked_generated_answer
    )