from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from rag.retrievers.hyde import HyDERetriever


@dataclass
class StubLoader:
    documents: List[str]

    def load_pdfs(self) -> List[str]:
        return self.documents

    def chunk_text(self, texts: Iterable[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        return list(texts)


class StubEmbeddings:
    def __init__(self) -> None:
        self._vector = np.ones((1, 4), dtype=np.float32)

    def encode(self, texts: Iterable[str]):
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 4), dtype=np.float32)
        return np.repeat(self._vector, repeats=len(text_list), axis=0)


class StubLanguageModel:
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "stub response"


def test_hyde_retriever_initialization():
    loader = StubLoader(["Document one", "Document two"])
    retriever = HyDERetriever(
        files_path="unused",
        loader=loader,
        language_model=StubLanguageModel(),
        embedding_model=StubEmbeddings(),
    )
    assert retriever is not None


def test_hyde_retriever_retrieve():
    loader = StubLoader(["Document content"])
    retriever = HyDERetriever(
        files_path="unused",
        loader=loader,
        language_model=StubLanguageModel(),
        embedding_model=StubEmbeddings(),
    )
    similar_docs, hypothetical_doc = retriever.retrieve("What is the topic?", k=1)
    assert isinstance(hypothetical_doc, str)
    assert isinstance(similar_docs, list)
