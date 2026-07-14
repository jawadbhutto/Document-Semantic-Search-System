"""
config.py
---------
Single source of truth for every path, model name, and tunable parameter
used across the pipeline. Values can be overridden via environment
variables or a `.env` file (see `.env.example`).

Nothing in this file performs I/O beyond reading environment variables,
so it is safe to import from anywhere without side effects.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --------------------------------------------------------------- #
    # Paths
    # --------------------------------------------------------------- #
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    results_dir: Path = project_root / "results"
    chroma_persist_dir: Path = project_root / "data" / "processed" / "chroma_db"

    # --------------------------------------------------------------- #
    # Document loading / chunking
    # --------------------------------------------------------------- #
    pdf_glob_pattern: str = "*.pdf"
    chunk_size: int = 500
    chunk_overlap: int = 100

    # --------------------------------------------------------------- #
    # Embeddings
    # --------------------------------------------------------------- #
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True

    # --------------------------------------------------------------- #
    # Vector store
    # --------------------------------------------------------------- #
    collection_name: str = "document_semantic_search"
    # Cosine similarity gives a bounded, intuitive [0, 1] relevance score.
    distance_metric: str = "cosine"

    # --------------------------------------------------------------- #
    # Search
    # --------------------------------------------------------------- #
    top_k: int = Field(default=3, ge=1, le=50)
    # Minimum cosine similarity (0-1) a result must clear to be considered
    # relevant. Anything below this is treated as "not found".
    relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # --------------------------------------------------------------- #
    # Logging
    # --------------------------------------------------------------- #
    log_level: str = "INFO"

    def ensure_directories(self) -> None:
        """Create every directory this project writes to, if missing."""
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.results_dir,
            self.chroma_persist_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# A single shared instance, imported everywhere else as `from src.config import settings`.
settings = Settings()
