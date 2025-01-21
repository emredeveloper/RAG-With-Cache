import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.retriever import Retriever
from model.langchain_retriever import LangChainRetriever
from model.embedding_model import EmbeddingModel

def test_langchain_retriever():
    # Model adını ve belgeleri tanımla
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    documents = ["Test document 1", "Test document 2"]
    
    # LangChainRetriever'ı başlat
    retriever = LangChainRetriever(
        embedding_model_name=embedding_model_name,
        documents=documents
    )
    
    # Sorgu yap ve sonuçları kontrol et
    results = retriever.get_relevant_documents("test")
    assert len(results) <= 2, "Sonuç sayısı beklenenden fazla!"