"""
search.py
---------
Single responsibility: semantic search. Turns a raw query into a ranked,
relevance-filtered list of (Document, similarity_score) results.

This is where the "nothing relevant was found" policy lives: if the
top result doesn't clear `settings.relevance_threshold`, we return an
empty list rather than forcing irrelevant chunks on the caller.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import settings
from src.utils import get_logger
from src.vector_store import similarity_search

logger = get_logger(__name__)

NO_RESULTS_MESSAGE = "No sufficiently relevant information was found in the indexed documents."


def _distance_to_similarity(distance: float) -> float:
    """
    Convert Chroma's cosine *distance* into an intuitive [0, 1] *similarity*
    score, where 1.0 means identical and 0.0 means unrelated.

    Chroma (with hnsw:space="cosine") reports distance = 1 - cosine_similarity.
    """
    return max(0.0, 1.0 - distance)


def search(
    query: str,
    k: int | None = None,
    threshold: float | None = None,
) -> list[tuple[Document, float]]:
    """
    Semantic search over the indexed documents.

    Parameters
    ----------
    query : str
        Natural-language search query.
    k : int, optional
        Number of top results to consider (defaults to `settings.top_k`).
    threshold : float, optional
        Minimum similarity score to keep a result (defaults to
        `settings.relevance_threshold`).

    Returns
    -------
    list[tuple[Document, float]]
        Relevant (Document, similarity_score) pairs, ordered best-first.
        Empty list if nothing clears the threshold.
    """
    query = (query or "").strip()
    if not query:
        logger.warning("Empty query passed to search()")
        return []

    k = k or settings.top_k
    threshold = settings.relevance_threshold if threshold is None else threshold

    raw_results = similarity_search(query, k=k)

    scored = [(doc, _distance_to_similarity(dist)) for doc, dist in raw_results]
    relevant = [(doc, score) for doc, score in scored if score >= threshold]

    if not relevant:
        logger.info(
            "No results cleared relevance threshold %.2f for query: %r", threshold, query
        )
        return []

    logger.info("Returning %d relevant result(s) for query: %r", len(relevant), query)
    return relevant
