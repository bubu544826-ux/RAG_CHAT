"""Coordinate retrieval, prompt building, and answer generation."""

from pathlib import Path

from .generator import Generator
from .prompt_builder import build_prompt
from .retriever import DEFAULT_TOP_K, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "index.json"


class RAGService:
    """Run the three steps of the RAG question-answering flow."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        top_k: int = DEFAULT_TOP_K,
        generator: Generator | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.top_k = top_k
        self.generator = generator if generator is not None else Generator()

    def retrieve(self, question: str) -> list[dict[str, object]]:
        """Retrieve relevant chunks for one question."""
        return retrieve(question, self.index_path, top_k=self.top_k)

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict[str, object]],
    ) -> str:
        """Build the LLM prompt from the question and retrieved chunks."""
        return build_prompt(question, retrieved_chunks)

    def generate(self, prompt: str) -> str:
        """Generate the final answer from the built prompt."""
        return self.generator.generate(prompt)

    def ask(self, question: str) -> dict[str, object]:
        """Return an answer and its retrieved source information."""
        retrieved_chunks = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved_chunks)
        answer = self.generate(prompt)
        sources = [
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ]
        return {"answer": answer, "sources": sources}
