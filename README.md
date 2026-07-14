# Document Semantic Search

A modular semantic search system over local PDF documents, built with
LangChain, HuggingFace `sentence-transformers`, ChromaDB, and Streamlit.

Drop PDFs into `data/raw/`, build the index, and ask natural-language
questions — you get back the most relevant chunks, ranked by similarity,
with a clear "nothing relevant found" fallback when your documents don't
cover the question.

## Folder structure

```
week-2-semantic-search/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                 # Streamlit entry point
│
├── data/
│   ├── raw/                # put your source PDFs here
│   └── processed/          # persisted ChromaDB files (auto-generated)
│
├── results/                # optional: exported search results
├── tests/                  # pytest unit tests
│
└── src/
    ├── config.py            # all settings (env-driven)
    ├── loaders.py            # PDF -> list[Document]
    ├── splitter.py            # Document -> chunked Document
    ├── embeddings.py           # embedding model provider
    ├── vector_store.py         # ChromaDB: create / save / load / search
    ├── search.py                 # relevance-filtered top-K search
    ├── pipeline.py                # orchestrates load -> split -> embed -> store
    └── utils.py                    # logging, text cleaning, printing
```

## How it works

**Ingestion pipeline** (`src/pipeline.py`):

```
load_pdfs()  ->  split_documents()  ->  embed (lazy)  ->  add_chunks()
```

1. `loaders.py` reads every PDF in `data/raw/` page by page.
2. `splitter.py` breaks pages into ~500-character overlapping chunks and
   tags each with a clean source filename and sequential chunk number.
3. `embeddings.py` loads `BAAI/bge-small-en-v1.5` (normalized embeddings)
   once per process.
4. `vector_store.py` upserts chunks into a persistent ChromaDB collection,
   using a deterministic `source_page_chunk` ID so re-running the pipeline
   never creates duplicate vectors.

**Search** (`src/search.py`):

The vector store is configured with cosine distance. Chroma returns
`distance = 1 - cosine_similarity`; `search.py` converts that back into a
`[0, 1]` similarity score and drops anything below `RELEVANCE_THRESHOLD`
(default `0.5`). If nothing clears the bar, the caller gets an empty list
instead of forced, irrelevant matches — e.g. asking *"Who won the FIFA
World Cup?"* against a corpus of AI papers returns:

> No sufficiently relevant information was found in the indexed documents.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # optional — defaults work out of the box
```

Place one or more PDFs in `data/raw/`.

## Usage

### Build the index

```bash
python -m src.pipeline
```

### Run the app

```bash
streamlit run main.py
```

Use the sidebar to (re)build the index, adjust `Top K` and the relevance
threshold, then type a question and click **Search**.

### Programmatic use

```python
from src.pipeline import run_ingestion_pipeline
from src.search import search

run_ingestion_pipeline()                 # index everything in data/raw/
results = search("What is Artificial Intelligence?", k=3)

for doc, score in results:
    print(score, doc.metadata["source"], doc.page_content[:200])
```

## Configuration

Everything is tunable via `.env` (see `.env.example`) or environment
variables — chunk size/overlap, embedding model, distance metric, top-K,
and the relevance threshold. Defaults live in `src/config.py`.

## Tests

```bash
pytest
```

Tests cover text cleaning/formatting, chunking behavior, and the
relevance-filtering logic in `search.py` (via a mocked vector store, so
no model download is required to run them).
