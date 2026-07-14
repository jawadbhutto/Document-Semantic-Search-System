"""
pipeline.py
-----------
The orchestrator. Wires together loaders -> splitter -> embeddings ->
vector_store in the correct order. This is the only module that should
call all four of those in sequence — everything else uses them
individually.

    load_pdfs -> split_documents -> (embedding model loaded lazily) -> add_chunks
"""

from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.loaders import load_pdfs
from src.splitter import split_documents
from src.utils import get_logger
from src.vector_store import add_chunks, collection_count

logger = get_logger(__name__)


def run_ingestion_pipeline(source_dir: str | Path | None = None) -> dict:
    """
    Run the full ingest pipeline end to end: load raw PDFs, chunk them,
    embed them, and store them in ChromaDB.

    Returns
    -------
    dict
        Summary stats: documents loaded, chunks created, chunks newly
        added, and the total vectors now in the collection.
    """
    settings.ensure_directories()

    logger.info("Pipeline started")

    documents = load_pdfs(source_dir)
    if not documents:
        logger.warning("No documents were loaded; pipeline stopping early")
        return {
            "documents_loaded": 0,
            "chunks_created": 0,
            "chunks_added": 0,
            "total_vectors": collection_count(),
        }

    chunks = split_documents(documents)
    added = add_chunks(chunks)
    total = collection_count()

    summary = {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_added": added,
        "total_vectors": total,
    }

    logger.info("Pipeline finished: %s", summary)
    return summary


if __name__ == "__main__":
    result = run_ingestion_pipeline()
    print(result)
