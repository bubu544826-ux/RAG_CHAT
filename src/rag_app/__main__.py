"""Allow the application to run with ``python -m src.rag_app``."""

from .app import main


if __name__ == "__main__":
    raise SystemExit(main())
