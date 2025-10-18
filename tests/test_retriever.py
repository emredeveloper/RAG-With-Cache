import numpy as np

from rag.retrievers.faiss import FaissRetriever


class StubEmbeddingModel:
    def encode(self, texts):
        return np.arange(len(list(texts)) * 4, dtype=np.float32).reshape(-1, 4)


def test_retriever_initialization():
    retriever = FaissRetriever(StubEmbeddingModel())
    assert retriever is not None


def test_retriever_build_index_and_retrieve():
    retriever = FaissRetriever(StubEmbeddingModel())
    documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    retriever.build_index(documents)
    results = retriever.retrieve("capital of France", top_k=1)
    assert len(results) == 1
