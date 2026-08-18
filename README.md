# RAG project

This is a learning-oriented RAG project. It currently covers document loading, text chunking, embedding, a local Chroma vector store, vector retrieval, prompt building, and answer generation with Anthropic.

## Project structure

```text
.
├── ingest.py             # command entry point for building the local index
├── retrieve.py           # retrieve the Top K chunks from the local index
├── evaluate_retrieval.py # compute Recall, Precision, MRR, and NDCG for the retriever
├── evaluate_rag.py       # end-to-end evaluation of retrieval, answer facts, and refusals
├── RAG_test.json         # 20 retrieval test cases labeled with relevant chunks
├── evaluation_questions.json # legacy source-level retrieval test questions
├── evaluation_cases.json # end-to-end RAG test data
├── src/
│   └── rag_app/          # application source package
│       ├── __init__.py
│       ├── __main__.py   # module entry point for `python -m src.rag_app`
│       ├── app.py        # minimal startup function
│       ├── api.py        # FastAPI /chat endpoint
│       ├── chunker.py    # split documents by character count
│       ├── config.py     # central model configuration
│       ├── document_loader.py # load txt and md files
│       ├── embedding.py  # turn text into vectors
│       ├── indexer.py    # orchestrate and save the local Chroma index
│       ├── retriever.py  # query Chroma and return the Top K chunks
│       ├── prompt_builder.py # build a prompt from the question and retrieved results
│       ├── generator.py  # call Anthropic to generate the answer
│       ├── rag_service.py # orchestrate the full RAG question-answering flow
│       └── web/          # simple web UI (HTML, CSS, JavaScript)
├── data/
│   ├── raw/              # raw data (not committed to Git by default)
│   └── processed/        # processed data (not committed to Git by default)
├── tests/                # automated tests
├── .env.example          # example environment variables, no real secrets
├── .gitignore            # Git ignore rules
├── requirements.txt      # third-party Python dependencies
└── README.md             # project documentation
```

## Requirements

- Python 3.10 or newer

## Install and run

### Using uv (recommended in the current environment)

`uv` is already installed on this machine together with a usable Python 3.12. Creating a persistent `.venv` first avoids re-resolving the `--with-requirements` dependencies on every run:

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m src.rag_app
```

You can also skip the `.venv` and use a temporary environment every time:

```powershell
uv run --no-project --python 3.12 python -m src.rag_app
```

### Using an installed Python

If `python --version` reports Python 3.10 or newer, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.rag_app
```

Expected output:

```text
RAG project initialized successfully.
```

## Inspect the embedding type and dimension

On its first run the test script downloads the configured model from Hugging Face. After you enter a sentence, the script only shows the vector type and dimension — it never prints the full vector:

```powershell
python -m scripts.test_embedding
```

With `uv` and no dependencies installed yet, you can run it directly:

```powershell
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m scripts.test_embedding
```

The default model is configured centrally in `src/rag_app/config.py`, and can be overridden through the `EMBEDDING_MODEL_NAME` environment variable before the process starts.

## Plot embeddings as 3D vectors

An embedding has over a thousand dimensions, so it cannot be inspected directly. The visualization script uses PCA to find the 3 directions of greatest variance, projects the vectors onto those 3 dimensions, and draws them as arrows from the origin:

```powershell
python -m scripts.visualize_embeddings
```

With `uv` and no dependencies installed yet, you can run it directly:

```powershell
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m scripts.visualize_embeddings
```

By default it uses the built-in example sentences (three groups: animals, programming, weather) and saves the image to `data/processed/embeddings_3d.png`. Common options:

| Option | Description |
| --- | --- |
| `--input sentences.txt` | use your own sentences, one per line, at least 3 |
| `--output my_plot.png` | change where the image is saved |
| `--show` | open an interactive window after saving so the plot can be rotated |

The percentage in each axis label is the share of information kept by that principal component. The three percentages usually add up to far less than 100%, which shows the plot is only a projection of a high-dimensional space: distances on the plot are not the true vector distances.

## Build the local index

Put `.txt` or `.md` files into `data/raw/`, then run from the project root:

```powershell
python ingest.py
```

The command runs loader → chunker → embedding in order and produces a persistent Chroma vector store in the `data/processed/index.json` directory. The name `index.json` is kept only for compatibility with the existing Retriever and RAGService call sites; it is no longer a JSON file.

The Chroma collection is named `rag_chunks` and uses cosine distance. Every chunk is stored as:

```json
{
  "id": "rag_notes.md#chunk-0",
  "document": "...",
  "metadata": {
    "source": "rag_notes.md",
    "chunk_id": "rag_notes.md#chunk-0"
  },
  "embedding": [0.01, 0.02]
}
```

On the first migration, if a legacy `index.json` file exists at that path, ingestion renames it to `index.json.legacy` before creating the Chroma directory. Re-running ingestion fully rebuilds the `rag_chunks` collection so that deleted documents do not linger in the index.

When the run finishes, the terminal shows the number of files, chunks, and embeddings. The default chunk size is 500 characters, with 50 characters of overlap between adjacent chunks.

The indexer collects all chunks first and then vectorizes them in batches through a single `embed_texts` call (default `batch_size=32`), instead of calling the model once per chunk. `ingest.py` shows a batch progress bar.

## Run embedding on the GPU

Embedding is the slowest step in ingestion. On Windows, the `torch` that PyPI installs by default is the **CPU-only** build, which will not use an NVIDIA card even when one is present. Check the current state:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the version ends in `+cpu` or `is_available()` is `False`, install a CUDA build. The Quadro P4000 on this machine is a Pascal card (compute capability 6.1), and CUDA 12.8 and later no longer support that architecture, so cu126 has to be requested explicitly:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch
```

### A recent enough display driver is also required

Installing the CUDA build of torch is not enough on its own — **the display driver must support the matching CUDA version too**. Check the `CUDA Version` in the top-right corner of `nvidia-smi`:

```powershell
nvidia-smi
```

This machine originally ran driver 516.40 (CUDA 11.7, a 2022 release), where torch cu126 reports `CUDA initialization: The NVIDIA driver on your system is too old` and `is_available()` stays `False`. CUDA 12.x needs driver 525 or newer.

From <https://www.nvidia.com/Download/index.aspx>, choose
`NVIDIA RTX / Quadro` → `Quadro Series` → `Quadro P4000` → `Windows 10 64-bit`,
download and install the latest driver, then reboot. Pascal support is kept all the way through the R580 driver branch.

After the driver update, torch does not need to be reinstalled; re-run the check command and `is_available()` should be `True`. sentence-transformers picks up the detected GPU automatically, with no code changes.

## Retrieve the Top K chunks

Build the index first, then run from the project root:

```powershell
python retrieve.py "What is RAG?" --top-k 3
```

`--top-k` configures how many results come back (default 3); `--index` points at a different Chroma index directory. The command only runs the question embedding and the Chroma cosine query — it never calls the LLM. The retriever converts the Chroma distance into the cosine similarity score used by the original interface, and the output is still JSON sorted by `score` in descending order:

```json
[
  {
    "text": "...",
    "source": "rag_notes.md",
    "chunk_id": "rag_notes.md#chunk-0",
    "score": 0.82
  }
]
```

## Run the retrieval evaluation

Build the index first, then run:

```powershell
python evaluate_retrieval.py
```

By default the script reads `RAG_test.json`, retrieves once per question, and computes ranking metrics against the explicitly labeled relevant chunks. The default is `K=5`, **and by default it runs exactly the same production retrieval pipeline as `RAGService`** (`vector + BM25 -> RRF -> CrossEncoder rerank -> neighbour expansion`), configured from `config.RETRIEVAL_SETTINGS`, so evaluation and live question answering use one and the same pipeline.

```powershell
python evaluate_retrieval.py --test-set RAG_test.json --index data/processed/index.json --top-k 5

# evaluate only the old vector-only baseline, for comparison with historical numbers
python evaluate_retrieval.py --baseline
```

On the current 20 test cases:

| Pipeline | Recall@5 | MRR@5 | NDCG@5 |
| --- | --- | --- | --- |
| `--baseline` (vector-only) | 69.17% | 61.83% | 59.20% |
| default production pipeline | **84.17%** | **81.67%** | **77.53%** |

The test set is a non-empty JSON array. Every case needs a unique non-empty `id`, a non-empty `question`, and a `relevant_chunk_ids` list of unique non-empty strings; fields such as `ground_truth` and `evidence` are kept as metadata, but the retrieval evaluation never calls the LLM and never uses a similarity threshold:

```json
{
  "id": "rag_quic_001",
  "question": "In QUIC, what is the final size of a stream?",
  "relevant_chunk_ids": ["RCF.txt#chunk-116"],
  "ground_truth": "..."
}
```

All four metrics are computed per question and then macro-averaged over all questions, and reported as a percentage between `0` and `100`:

- `Recall@K = relevant chunks in the top K / all labeled relevant chunks for that question`
- `Precision@K = relevant chunks in the top K / K`. The denominator is `K` rather than the number of labels, so a question with only 1 labeled relevant chunk can reach at most 20% at `K=5`. The current test set averages 1.5 labels per question, which puts the theoretical ceiling of `Precision@5` at 30%; the script prints that ceiling alongside the value so a number near the ceiling is not misread as a failure. `NDCG@K` is normalized and works better as the headline metric.
- `MRR@K = 1 / rank of the first relevant chunk`; `0` when there is no hit in the top K
- `NDCG@K = DCG@K / IDCG@K`, with binary relevance `rel` of `0` or `1`, where `DCG@K = Σ rel(rank) / log2(rank + 1)`

The chunk labels in the current `RAG_test.json` depend on `data/raw/RCF.txt` and on the `chunk_size=500`, `overlap=50` chunking configuration. If the source document or the chunking parameters change, `relevant_chunk_ids` has to be re-checked, otherwise the metrics no longer represent real retrieval quality. `--questions` is still accepted as a compatibility alias for `--test-set`.

### Comparing the full retrieval pipeline

The application defaults to `vector + BM25 -> RRF -> CrossEncoder rerank -> neighbour expansion`. Calling `retriever.retrieve()` directly keeps the original vector-only default behaviour for backwards compatibility with older code; every production entry point (`RAGService`, `retrieve.py`, `evaluate_retrieval.py`) reads the same configuration through `retriever.production_retrieval_options()`, so they cannot drift apart any more.

Run the comparison experiments, which share the same labels and cutoff:

```powershell
python evaluate_retrieval.py --compare --output retrieval_evaluation_report.json

# also evaluate several reranker models (the first run downloads them)
python evaluate_retrieval.py --compare --compare-rerankers
```

#### Neighbour expansion

The chunker cuts hard at 500 characters with only 50 characters of overlap, so one fact often ends up split across two adjacent chunks (8 of the 20 test cases are labeled with an adjacent chunk pair). After reranking, the chunks before and after each hit are therefore pulled into the results, which are then truncated to `top_k`. This is the single highest-impact change so far: `Recall@5 74.17% -> 84.17%`, `NDCG@5 71.45% -> 77.53%`.

`NEIGHBOUR_RADIUS=2` scores higher on this test set (`Recall@5 86.67%`, `NDCG@5 78.85%`), but at `top_k=5` the results degenerate into a contiguous window of "one hit ± 2 neighbours", which sacrifices the ability to gather evidence across sections, so the default stays at `1`.

#### On the choice of reranker model

Three CrossEncoders were compared on the same test set (`--compare-rerankers`, CPU):

| Model | NDCG@5 | median latency |
| --- | --- | --- |
| `cross-encoder/ms-marco-MiniLM-L6-v2` (default) | **77.53%** | **890ms** |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | 74.68% | 1378ms |
| `BAAI/bge-reranker-base` | 78.33% | 2733ms |

L12 is actually worse than L6, and bge-base gains only 0.8 points while being 3x slower, so L6 stays. In other words, the remaining loss in the ranking stage is not something a bigger reranker can solve: the candidate pool already contains 98.33% of the labeled chunks, and the real bottleneck is the chunk fragments produced by the hard 500-character cut (many chunks start mid-word), which a cross-encoder also struggles to score. The genuinely worthwhile next step is to improve the chunking, but that would invalidate every `relevant_chunk_ids` in `RAG_test.json` and requires re-labeling first.

#### On query rewriting

`QUERY_REWRITE_ENABLED` now defaults to `false`. `rule_based_rewrite` prunes keywords rather than rewriting with an LLM, and `is_precise_query()` short-circuits on QUIC questions containing tokens such as `MAX_STREAM_DATA`. In measurements, all four metrics were identical to having it off, while every query cost about 500ms more. The code path and the environment variables are kept so it can be re-enabled and re-measured on a different corpus.

The report contains Recall@1/@3/@5/@10, Precision, MRR, NDCG, and mean/median/P95 retrieval latency. On its first run the CrossEncoder downloads and caches `cross-encoder/ms-marco-MiniLM-L6-v2`; the cold start from loading the model shows up in the mean, while the median is closer to the warmed-up per-request latency.

The environment variables below control retrieval, and their defaults are also listed in `.env.example`:

```dotenv
RETRIEVAL_STRATEGY=hybrid
VECTOR_TOP_K=30
LEXICAL_TOP_K=30
RRF_K=60
RERANKER_ENABLED=true
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L6-v2
RETRIEVAL_CANDIDATE_K=30
FINAL_TOP_K=5
NEIGHBOUR_EXPANSION_ENABLED=true
NEIGHBOUR_RADIUS=1
QUERY_REWRITE_ENABLED=false
QUERY_REWRITE_MODE=multi_query
MAX_QUERIES=3
```

`RETRIEVAL_STRATEGY` accepts `vector_only`, `lexical_only`, and `hybrid`; the rewrite mode accepts `single` and `multi_query`. A failed rewrite falls back to the original question, a failed reranker keeps the RRF order, and if either retrieval backend fails the other one is used. Results keep `document_id`, `chunk_id`, `retrieval_score`, `rerank_score`, `original_rank`, `final_rank`, and `retrieval_source` for debugging and offline evaluation.

## Ask questions with RAGService

Build the local index first, then put the Anthropic API key in a `.env` file at the project root:

```dotenv
ANTHROPIC_API_KEY=your-anthropic-api-key
```

With `uv run`, load that file through `--env-file .env`:

```powershell
uv run --env-file .env --no-project --python 3.12 --with-requirements requirements.txt python -c "from src.rag_app.rag_service import RAGService; print(RAGService().ask('What is RAG?'))"
```

You can then call it from Python:

```python
from src.rag_app.rag_service import RAGService

service = RAGService()
answer = service.ask("What is RAG?")
print(answer)
```

`ask(question)` runs `retrieve -> build_prompt -> generate` in order. The default model is managed centrally in `src/rag_app/config.py` and can be overridden through the `ANTHROPIC_MODEL_NAME` environment variable.

## Run the end-to-end RAG evaluation

Build the index and configure `ANTHROPIC_API_KEY` first, then run:

```powershell
python evaluate_rag.py --top-k 3
```

The script reads `evaluation_cases.json`. Answerable cases use `{question, expected_source, expected_answer_keywords}`; a no-answer case sets `expected_source` to `null` and the keywords to an empty array. Every question goes through the full RAG flow exactly once, and the script computes:

- `retrieval_recall_at_3`: whether the correct source appears in the Top 3, counted for answerable questions only.
- `answer_keyword_pass_rate`: whether the answer contains every key fact for that case.
- `no_answer_refusal_rate`: whether a no-answer question is met with an explicit refusal such as "does not contain enough information".

The summary is printed to the terminal, and the per-question detail is written to `evaluation_report.json` by default. `--cases`, `--index`, `--top-k`, and `--output` change the input, the K value, and the report path.

## Start FastAPI

Build the local index first and configure `ANTHROPIC_API_KEY` in `.env`. To start with `uv`:

```powershell
uv run --env-file .env --no-project --python 3.12 --with-requirements requirements.txt uvicorn src.rag_app.api:app --host 127.0.0.1 --port 8000
```

If the dependencies are installed and the virtual environment is active, run:

```powershell
uvicorn src.rag_app.api:app --host 127.0.0.1 --port 8000
```

Once it is running, open `http://127.0.0.1:8000/` for the web UI; the interactive API documentation is still available at `http://127.0.0.1:8000/docs`.

The endpoint takes a question and returns the answer and source information from `RAGService.ask()` unchanged:

```http
POST /chat
Content-Type: application/json

{"question": "What is RAG?"}
```

## Run the tests

With `uv`:

```powershell
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m unittest discover -s tests -v
```

Or with an installed and activated Python:

```powershell
python -m unittest discover -s tests -v
```
