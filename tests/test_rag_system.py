# tests/test_rag_system.py
from model.rag_system import RAGSystem
from model.embedding_model import EmbeddingModel
from model.retriever import Retriever
from model.language_model import LanguageModel

def test_rag_system_initialization():
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(embedding_model)
    language_model = LanguageModel("gpt2")
    rag_system = RAGSystem(embedding_model, retriever, language_model)
    assert rag_system is not None

def test_rag_system_answer_question():
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(embedding_model)
    documents = ["Paris is the capital of France.", "London is the capital of the UK."]
    retriever.build_index(documents)
    language_model = LanguageModel("gpt2")
    rag_system = RAGSystem(embedding_model, retriever, language_model)
    answer = rag_system.answer_question("What is the capital of France?")
    assert isinstance(answer, str)
    assert len(answer) > 0