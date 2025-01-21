import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.retriever import Retriever
from model.langchain_retriever import LangChainRetriever
from model.embedding_model import EmbeddingModel

def test_faiss_retriever():
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(embedding_model)
    documents = ["Test document 1", "Test document 2"]
    retriever.build_index(documents)
    results = retriever.retrieve("test", top_k=1)
    assert len(results) == 1

def test_langchain_retriever():
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    documents = ["Test document 1", "Test document 2"]
    retriever = LangChainRetriever(embedding_model, documents)
    results = retriever.get_relevant_documents("test")
    assert len(results) <= 2