"""Utilities for working with embedding models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrap ``SentenceTransformer`` with simple on-disk caching."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        safe_name = model_name.replace("/", "_")
        self.cache_path = Path("model_cache") / safe_name
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        if any(self.cache_path.iterdir()):
            print(f"Loading model from cache: {self.model_name}")
            return SentenceTransformer(str(self.cache_path))
        print(f"Downloading and caching model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        model.save(str(self.cache_path))
        return model

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        """Convert text into embeddings."""
        return self.model.encode(list(texts))
