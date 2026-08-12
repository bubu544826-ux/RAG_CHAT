"""Split a document into overlapping character chunks."""


def chunk_document(
    document: dict[str, str],
    chunk_size: int,
    overlap: int = 0,
) -> list[dict[str, str]]:
    """Split one document and preserve its source on every chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document["text"]
    source = document["source"]
    step = chunk_size - overlap
    chunks = []

    for chunk_index, start in enumerate(range(0, len(text), step)):
        end = start + chunk_size
        chunks.append(
            {
                "text": text[start:end],
                "source": source,
                "chunk_id": f"{source}#chunk-{chunk_index}",
            }
        )

        # The final chunk already contains the end of the document. Stopping here
        # avoids creating a shorter chunk made only from the overlap.
        if end >= len(text):
            break

    return chunks
