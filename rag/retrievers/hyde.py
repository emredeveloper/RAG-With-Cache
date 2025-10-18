"""Implementation of the HyDE retriever without external prompt helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np

from ..data.pdf_loader import PDFLoader
from ..embeddings import EmbeddingModel
from ..language import LanguageModel


class HyDERetriever:
    """Generate hypothetical documents to guide retrieval."""

    def __init__(
        self,
        files_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        language_model_name: str = "gpt2-medium",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        *,
        loader: Optional[PDFLoader] = None,
        language_model: Optional[LanguageModel] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.llm = language_model or LanguageModel(language_model_name)
        self.embeddings = embedding_model or EmbeddingModel(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader = loader or PDFLoader(files_path)
        self.index, self.chunks = self._encode_documents()

    def _encode_documents(self) -> Tuple[faiss.Index, Sequence[str]]:
        documents = self.loader.load_pdfs()
        chunks = self.loader.chunk_text(documents, self.chunk_size, self.chunk_overlap)
        embeddings = np.asarray(self.embeddings.encode(chunks), dtype=np.float32)
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index, chunks

    def _build_prompt(self, query: str) -> str:
        return (
            "Generate a concise academic style passage that answers the question.\n"
            f"Question: {query}\n"
            "Hypothetical Document:"
        )

    def generate_hypothetical_document(self, query: str) -> str:
        prompt = self._build_prompt(query)
        return self.llm.generate(prompt, max_new_tokens=self.chunk_size)

    def retrieve(self, query: str, k: int = 3) -> Tuple[List[Tuple[str, float]], str]:
        hypothetical_doc = self.generate_hypothetical_document(query)
        hypothetical_embedding = np.asarray(
            self.embeddings.encode([hypothetical_doc]), dtype=np.float32
        )
        faiss.normalize_L2(hypothetical_embedding)
        scores, indices = self.index.search(hypothetical_embedding, k)
        cosine_similarities = (scores + 1) / 2
        similar_docs = [
            (self.chunks[idx], float(cosine_similarities[0][i]))
            for i, idx in enumerate(indices[0])
        ]
        return similar_docs, hypothetical_doc
