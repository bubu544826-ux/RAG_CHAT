"""Load plain-text documents for the RAG pipeline."""

from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(data_directory: str | Path = "data") -> list[dict[str, str]]:
    """Read .txt and .md files and return a common document structure."""
    directory = Path(data_directory)

    if not directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {directory}")

    documents = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(
                {
                    "text": file_path.read_text(encoding="utf-8"),
                    "source": file_path.relative_to(directory).as_posix(),
                }
            )

    return documents
