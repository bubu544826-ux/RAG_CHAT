"""Build a prompt from a question and retrieved knowledge chunks."""


SYSTEM_INSTRUCTION = """You are an assistant that answers questions from a knowledge base.
Follow these rules:
1. Write the entire answer in English, even when the question or the context is
   in another language. Translate what you need from the context rather than
   quoting it in its original language.
2. Answer from the provided context first.
3. If the context does not hold enough information to answer, say plainly that
   the provided context does not contain enough information to answer the question.
4. Do not invent information that is not in the context."""

ANSWER_LANGUAGE_REMINDER = "Answer in English."


def build_prompt(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    """Return one prompt containing instructions, context, and the question."""
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if not question.strip():
        raise ValueError("question must not be empty.")
    if not isinstance(retrieved_chunks, list):
        raise TypeError("retrieved_chunks must be a list.")

    context_parts: list[str] = []

    for position, chunk in enumerate(retrieved_chunks, start=1):
        if not isinstance(chunk, dict):
            raise TypeError("Every chunk in retrieved_chunks must be a dictionary.")

        text = chunk.get("text")
        if not isinstance(text, str):
            raise ValueError("Every chunk must contain a string 'text' field.")

        source = chunk.get("source", "unknown source")
        chunk_id = chunk.get("chunk_id", "unknown id")
        context_parts.append(
            f"[Chunk {position}]\n"
            f"source: {source}\n"
            f"chunk_id: {chunk_id}\n"
            f"content: {text}"
        )

    context = (
        "\n\n".join(context_parts)
        if context_parts
        else "(no relevant content retrieved)"
    )

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"context:\n{context}\n\n"
        f"question:\n{question}\n\n"
        f"{ANSWER_LANGUAGE_REMINDER}"
    )
