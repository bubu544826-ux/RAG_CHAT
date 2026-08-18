Here's how to run it on this machine. There's no system python on PATH (the Microsoft Store stub answers instead), but uv 0.11.19 is installed — so every command goes through uv run.

1. Prerequisites (already in place)

- .env exists at the project root with ANTHROPIC_API_KEY and ANTHROPIC_MODEL_NAME.
- A Chroma index exists at data/processed/index.json/ (built Aug 12).

2. Rebuild the index

You added data/raw/RCF.txt after the index was built, so it isn't searchable yet. Re-run ingestion first:

uv run python ingest.py

This runs loader → chunker → embedding and rebuilds the rag_chunks collection from all files in data/raw/. First run downloads the embedding model from Hugging Face, so it takes a while.

3. Start the web app

uv run --env-file .env uvicorn src.rag_app.api:app --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/ for the chat UI, or /docs for the interactive API. The endpoint is POST /chat with {"question": "..."}.