"""Expose the RAG service through a small HTTP API."""

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag_service import RAGService


class ChatRequest(BaseModel):
    """Input accepted by the chat endpoint."""

    question: str = Field(min_length=1)


class SourceResponse(BaseModel):
    """Source metadata returned by ``RAGService``."""

    source: str
    chunk_id: str | int
    score: float


class ChatResponse(BaseModel):
    """Answer and sources returned to the client."""

    answer: str
    sources: list[SourceResponse]


@lru_cache
def get_rag_service() -> RAGService:
    """Create one reusable service instance for the application process."""
    return RAGService()


WEB_DIR = Path(__file__).with_name("web")

app = FastAPI(title="Learning RAG API")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the small browser client."""
    return FileResponse(WEB_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> dict[str, object]:
    """Pass a question to the existing RAG service."""
    return rag_service.ask(request.question)
