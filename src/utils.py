"""
utils.py
--------
Small, dependency-light helpers shared across the project:
logging setup, text cleaning, and result formatting/printing.

Nothing here knows about PDFs, embeddings, or Chroma — keep it generic.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Iterable

from langchain_core.documents import Document

from src.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, shared format across modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(settings.log_level.upper())
        logger.propagate = False
    return logger


def clean_text(text: str) -> str:
    """Normalize whitespace and strip odd PDF extraction artifacts."""
    if not text:
        return ""
    # Collapse runs of whitespace (including newlines) into single spaces.
    text = re.sub(r"\s+", " ", text)
    # Remove stray control characters some PDF extractors leave behind.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def truncate(text: str, max_chars: int = 300) -> str:
    """Shorten text for display, adding an ellipsis if truncated."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_result(rank: int, document: Document, score: float) -> str:
    """Render a single search hit as a human-readable block."""
    source = document.metadata.get("source", "unknown")
    page = document.metadata.get("page_label", document.metadata.get("page", "?"))
    lines = [
        "=" * 60,
        f"Result {rank}  |  relevance: {score:.3f}  |  source: {source}  (page {page})",
        "-" * 60,
        truncate(document.page_content, 400),
    ]
    return "\n".join(lines)


def print_results(results: Iterable[tuple[Document, float]]) -> None:
    """Pretty-print an ordered list of (Document, score) tuples to stdout."""
    results = list(results)
    if not results:
        print("No sufficiently relevant information was found in the indexed documents.")
        return
    for rank, (document, score) in enumerate(results, start=1):
        print(format_result(rank, document, score))
