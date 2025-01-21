from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import numpy as np

class LangChainRetriever(BaseRetriever):
    def __init__(self, embedding_model_name: str, documents: List[str]):
        # HuggingFace embedding modelini direkt yükle
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.documents = documents
        self.index = self._build_index()
        
    def _build_index(self):
        """Metinleri vektörleştir ve indeks oluştur"""
        doc_embeddings = self.embeddings.embed_documents(self.documents)
        return np.array(doc_embeddings)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Sorguya en benzer belgeleri bul"""
        query_embedding = self.embeddings.embed_query(query)
        similarities = np.dot(self.index, query_embedding)
        indices = np.argsort(similarities)[::-1][:5]  # Top 5 sonuç
        return [
            Document(
                page_content=self.documents[i],
                metadata={"similarity": float(similarities[i])}
            ) for i in indices
        ]