"""FAISS-based dense retriever."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import faiss
import numpy as np

from ..embeddings import EmbeddingModel


class FaissRetriever:
    """Simple wrapper around a FAISS index."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.index: faiss.Index | None = None
        self.documents: Sequence[str] | None = None

    def build_index(self, documents: Sequence[str]) -> None:
        self.documents = documents
        embeddings = np.asarray(self.embedding_model.encode(documents), dtype=np.float32)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        if self.index is None or self.documents is None:
            raise RuntimeError("The FAISS index has not been built yet.")
        query_embedding = np.asarray(self.embedding_model.encode([query]), dtype=np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        return [
            (self.documents[idx], float(distances[0][i]))
            for i, idx in enumerate(indices[0])
        ]
