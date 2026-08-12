"""Small lifecycle helper for Chroma tests on Windows."""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from chromadb.api.shared_system_client import SharedSystemClient


@contextmanager
def temporary_chroma_directory() -> Iterator[Path]:
    """Yield a temp path and close Chroma systems before deleting it."""
    temporary_directory = tempfile.TemporaryDirectory()
    try:
        yield Path(temporary_directory.name)
    finally:
        for system in list(SharedSystemClient._identifier_to_system.values()):
            system.stop()
        SharedSystemClient.clear_system_cache()
        temporary_directory.cleanup()
