  The suggestion is to split the MVP into 8 independently verifiable stages, wired together step by step into the complete chain:

  document loading → text chunking → vectorization → indexing → retrieval → context assembly → LLM generation → citation checking

  The first version only supports local Markdown/TXT, single-turn question answering, and simple vector retrieval. No LangChain, LlamaIndex, agents, authentication, or conversation memory.

  ## 1. Requirements analysis

  ### Core functionality

  The system needs to support two flows:

  1. Knowledge base construction

     local documents → reading and normalization → chunk → embedding → vector index

  2. User question answering

     question → query embedding → similarity search → prompt → LLM → answer + citations

  ### MVP inputs and outputs

  Inputs:

  - .md and .txt files
  - the user's natural language question
  - a configurable retrieval count top_k

  Outputs:

  - an answer generated from the retrieved content
  - the cited file names and chunk identifiers
  - an explicit "the knowledge base does not contain enough information" when there is not enough evidence

  ### Non-functional requirements

  - Each module has a single responsibility; avoid one file doing everything.
  - Every stage can be tested independently with fixed inputs.
  - Tests do not depend on the network or a real API key.
  - Embedding and the LLM are isolated behind simple interfaces so fake implementations can be swapped in.
  - Never expose the API key, full document text, or private paths in logs or exceptions.
  - Generated indexes go into data/ and are not committed to Git.

  ### Out of scope for the MVP

  - PDF, Word, web scraping
  - OCR
  - LangChain / LlamaIndex
  - Agents or tool calling
  - Multi-turn conversation and memory
  - Hybrid retrieval, reranking
  - User accounts and access control
  - Web UI
  - Distributed vector databases

  ## 2. Suggested module boundaries

  Suggested split by responsibility inside rag-app/src/:

  - documents: document reading, format validation, text normalization
  - chunking: text splitting and chunk metadata
  - embeddings: the embedding interface and its concrete implementations
  - index: vector storage, saving, loading, and similarity search
  - retrieval: question vectorization, Top-K retrieval, and threshold filtering
  - generation: prompt assembly and LLM calls
  - answers: answer structure, citation assembly, and citation checking
  - pipeline: connects the modules above without carrying low-level implementation
  - cli: the command-line entry point, provided in the final stage

  The key data objects should stay simple:

  - Document: document ID, source, body text, metadata
  - Chunk: chunk ID, document ID, body text, position, metadata
  - SearchResult: chunk, similarity score, rank
  - Answer: answer text, citation list, whether the evidence is sufficient

  This avoids passing unstructured dictionaries between modules.

  ## 3. Staged development plan

  ### Stage 0: project skeleton and testing foundation

  Goal: establish the smallest engineering structure that supports continuous development.

  Scope of work:

  - Create src/, tests/, tests/fixtures/, and data/
  - Prepare the dependency file, the environment variable example, and the README
  - Configure pytest
  - Decide the ignore rules for generated indexes and private files

  Independent verification:

  - Python can import the project modules
  - compileall passes
  - pytest can run one minimal test

  Definition of done: the project installs, imports, and runs its tests, but has no RAG functionality yet.

  ———

  ### Stage 1: document loading

  Goal: turn local .md and .txt files into a uniform Document.

  Scope of work:

  - UTF-8 text reading
  - File type validation
  - Empty file handling
  - Generating a stable document ID
  - Storing metadata such as file name, source, and type

  Independent tests:

  - Markdown and TXT are read correctly
  - Unsupported files are rejected
  - Empty file behaviour is well defined
  - Encoding problems and missing files return an understandable error
  - No unnecessary absolute paths are leaked

  Definition of done: given a file path, a normalized Document comes back reliably.

  ———

  ### Stage 2: chunk splitting

  Goal: turn a Document into retrievable chunks.

  A simple strategy is suggested for the first version:

  - Split on paragraph boundaries first
  - Split over-long paragraphs again with a character window
  - Support a small amount of overlap
  - Do not introduce complex semantic splitting

  Every chunk should keep:

  - chunk_id
  - document_id
  - the text content
  - its order within the document
  - the source file
  - position metadata usable for citations

  Independent tests:

  - A short document yields one chunk
  - A long document yields several chunks
  - Chunk size roughly matches the configuration
  - Overlap behaves correctly
  - Text order is unchanged
  - Metadata traces back to the original document
  - Whitespace-only content does not produce an invalid chunk

  Definition of done: chunk results are stable, traceable, and suitable for embedding.

  ———

  ### Stage 3: embedding abstraction

  Goal: turn chunk text into fixed-dimension vectors.

  Define a simple embedding interface first, then plug in a real provider. Tests use a deterministic fake embedder.

  Behaviour to pin down:

  - Batches of text are supported
  - Every input maps to one vector
  - Vector dimensions are consistent within one implementation
  - Empty input and service errors are handled explicitly
  - The model name and vector dimension can be recorded in the index metadata

  Independent tests:

  - The number of inputs matches the number of outputs
  - Dimensions are consistent
  - Batch processing is correct
  - Fake embedder results are reproducible
  - A dimension mismatch raises promptly
  - Unit tests never touch the real network

  Definition of done: chunks can be turned into vectors, and the layers above do not depend on a specific provider.

  ———

  ### Stage 4: local vector index

  Goal: store chunks together with their vectors and run similarity search.

  The first version can use NumPy and cosine similarity; a vector database is not needed straight away.

  Scope of work:

  - Add vector records
  - Save the index into data/
  - Reload it from disk
  - Validate vector dimensions
  - Return Top-K results for a query vector
  - Map search results back to chunks

  Independent tests:

  - Records can be added and read back
  - Reloading after saving gives identical results
  - Top-K ranking is correct
  - Behaviour is correct when top_k exceeds the number of records
  - An empty index returns empty results
  - A dimension mismatch raises

  Definition of done: given a query vector, similar chunks come back in a stable order.

  ———

  ### Stage 5: the retrieval layer

  Goal: turn a natural language question into usable retrieval results.

  The retriever is responsible for:

  - Embedding the question
  - Calling the vector index
  - Applying top_k
  - Applying a minimum similarity threshold
  - Returning SearchResults carrying the source and the score

  Independent tests:

  - A relevant question returns the relevant chunk first
  - An irrelevant question is filtered out by the threshold
  - Empty question behaviour is well defined
  - An empty index does not crash
  - Results carry the correct source, rank, and score

  Definition of done: a question in, filtered and explainable retrieval results out.

  ———

  ### Stage 6: LLM generation

  Goal: make the LLM answer only from the retrieved context.

  The prompt must explicitly require:

  - Using only the provided context
  - Saying so plainly when the answer is unknown
  - Never dressing outside knowledge up as document content
  - Using chunk identifiers so citations can be built later

  The LLM is likewise isolated behind a simple interface, and unit tests use a fake LLM.

  Independent tests:

  - The prompt contains the question and the selected chunks
  - With no retrieval results, the LLM is not called, or a missing-evidence answer is returned directly
  - Context order matches the retrieval ranking
  - Model exceptions are converted into understandable application errors
  - Tests need no API key

  Definition of done: a predictable answer can be generated from a fixed retrieval context.

  ———

  ### Stage 7: answer and citation assembly

  Goal: produce a final answer whose sources can be verified.

  A citation should carry at least:

  - the file name
  - the chunk ID
  - the chunk's order within the document
  - optionally a summary or snippet of the original text

  What must be prevented:

  - The LLM citing a chunk that does not exist
  - Citations that do not match the retrieval results
  - An answer with no verifiable evidence behind it

  Independent tests:

  - A grounded answer carries valid citations
  - With no evidence, a uniform missing-answer response is returned
  - Non-existent citations are rejected or filtered out
  - Citations trace back to the original file
  - The same input produces a stable citation order

  Definition of done: the answer is not only readable but also traceable and verifiable.

  ———

  ### Stage 8: end-to-end pipeline and CLI

  Goal: connect the already-tested components.

  Two commands are suggested:

  - Build or update the knowledge base index
  - Ask the knowledge base a question

  End-to-end tests use:

  - small fixture documents
  - a fake embedder
  - a fake LLM
  - a temporary index directory

  What to verify:

  - A document passes through every stage intact
  - A question retrieves the expected chunk
  - The final answer carries the correct citations
  - An irrelevant question returns "not enough information"
  - Results stay consistent after a restart that reloads the index

  Definition of done: the whole chain can be tested offline, with the real models as replaceable configuration.

  ## 4. Recommended implementation order

  Complete one stage at a time:

  1. Implement the data structures and modules for the current stage.
  2. Write the unit tests for that stage.
  3. Run the syntax checks and all existing tests.
  4. Update the README to record the stage's behaviour.
  5. Confirm the definition of done before moving to the next stage.

  Wiring up a real embedding provider and a real LLM should not be a prerequisite for early testing. Validating the data flow through fake implementations first, and only then connecting the real APIs, makes it much easier to tell a "business logic error" apart from an "external model service error".

  The next step is to start from stage 0: set up only the project skeleton, the test entry point, and the configuration conventions — no document reading or other RAG functionality yet.
