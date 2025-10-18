"""High-level orchestration for Retrieval-Augmented Generation."""

from __future__ import annotations

from typing import Sequence, Tuple

from .embeddings import EmbeddingModel
from .language import LanguageModel
from .retrievers.faiss import FaissRetriever
from .retrievers.hyde import HyDERetriever


class RAGSystem:
    """Coordinate embedding, retrieval, and generation components."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        retriever: FaissRetriever | HyDERetriever,
        language_model: LanguageModel,
    ) -> None:
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.language_model = language_model

    def answer_question(self, query: str, top_k: int = 2) -> str:
        similar_docs, _ = self.retriever.retrieve(query, top_k)
        prompt = self._create_prompt(query, similar_docs)
        return self.language_model.generate(prompt)

    def _create_prompt(self, query: str, similar_docs: Sequence[Tuple[str, float]]) -> str:
        context = "\n".join(
            f"[Document {idx + 1} - Confidence: {score:.2f}]\n{text}"
            for idx, (text, score) in enumerate(similar_docs)
        )
        return (
            "Answer the following question based on the documents:\n\n"
            f"Question: {query}\n\n"
            "Relevant Documents:\n"
            f"{context}\n\n"
            "Provide a concise and well-structured answer."
        )
