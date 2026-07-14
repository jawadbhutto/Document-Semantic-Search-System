"""
embeddings.py
-------------
Single responsibility: provide the embedding model used to turn text
into vectors. Everything else (chunking, storage, search) treats this
as an opaque `Embeddings` object.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Return a cached `HuggingFaceEmbeddings` instance.

    Cached with `lru_cache` because loading a sentence-transformers model
    is expensive (downloads/loads weights); we only want to pay that cost
    once per process, no matter how many modules call this function.
    """
    logger.info(
        "Loading embedding model '%s' on device '%s'",
        settings.embedding_model_name,
        settings.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": settings.normalize_embeddings},
    )
