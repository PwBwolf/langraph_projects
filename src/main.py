from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load environment variables before any other imports
load_dotenv()

from src.api.routes import router as api_router

app = FastAPI(title="LangGraph API")

# Include API routes
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=True,
    )