"""Configuration helpers for the reference RAG pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Config:
    """Light-weight container for runtime configuration.

    The defaults are deliberately conservative so that unit tests can
    instantiate the pipeline without downloading excessively large models.
    """

    pdf_directory: Path = Path("data/pdfs")
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 3

    embedding_models: Dict[str, str] = field(
        default_factory=lambda: {
            "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
            "mpnet": "sentence-transformers/all-mpnet-base-v2",
            "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        }
    )

    language_models: Dict[str, str] = field(
        default_factory=lambda: {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "flan-t5": "google/flan-t5-large",
        }
    )

    default_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    default_language_model: str = "gpt2-medium"

    retriever_options: List[Tuple[str, str]] = field(
        default_factory=lambda: [("FAISS", "faiss"), ("HyDE", "hyde")]
    )

    hyde_settings: Dict[str, object] = field(
        default_factory=lambda: {
            "chunk_size": 512,
            "chunk_overlap": 128,
            "top_k": 3,
            "language_model": "gpt2-medium",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        }
    )

    def ensure_pdf_directory(self) -> None:
        """Create the PDF directory if it is missing."""
        self.pdf_directory.mkdir(parents=True, exist_ok=True)
