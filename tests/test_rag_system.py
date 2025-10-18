import numpy as np

from rag.system import RAGSystem


class StubEmbeddingModel:
    def encode(self, texts):
        return np.ones((len(list(texts)), 4), dtype=np.float32)


class StubLanguageModel:
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        return "stub answer"


class StubRetriever:
    def retrieve(self, query, top_k):
        return [("Document text", 0.9)], "stub doc"


def test_rag_system_initialization():
    embedding_model = StubEmbeddingModel()
    retriever = StubRetriever()
    language_model = StubLanguageModel()
    rag_system = RAGSystem(embedding_model, retriever, language_model)
    assert rag_system is not None


def test_rag_system_answer_question():
    embedding_model = StubEmbeddingModel()
    retriever = StubRetriever()
    language_model = StubLanguageModel()
    rag_system = RAGSystem(embedding_model, retriever, language_model)
    answer = rag_system.answer_question("What is the capital of France?")
    assert isinstance(answer, str)
    assert len(answer) > 0
