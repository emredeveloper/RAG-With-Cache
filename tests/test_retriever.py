import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.retriever import Retriever
from model.embedding_model import EmbeddingModel


def test_retriever_initialization():
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(embedding_model)
    assert retriever is not None


def test_retriever_build_index():
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(embedding_model)
    documents = ["This is a test document.", "Another test document."]
    retriever.build_index(documents)
    assert retriever.index is not None


def test_retriever_retrieve():
    embedding_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(embedding_model)
    documents = ["This is a test document.", "Another test document."]
    retriever.build_index(documents)
    similar_docs = retriever.retrieve("test", top_k=1)
    assert len(similar_docs) == 1

