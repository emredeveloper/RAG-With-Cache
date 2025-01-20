# retriever.py
import faiss
import numpy as np
from embedding_model import EmbeddingModel

class Retriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = None
    
    def build_index(self, documents):
        """
        Belgeleri FAISS ile indeksler.
        
        Args:
            documents (list): İndekslenecek belgeler listesi.
        """
        self.documents = documents
        embeddings = self.embedding_model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def retrieve(self, query, top_k=2):
        """
        Sorguya en benzer belgeleri FAISS ile bulur.
        
        Args:
            query (str): Kullanıcı sorgusu.
            top_k (int): Kaç belge döneceği.
        
        Returns:
            list: En benzer belgeler ve benzerlik skorları.
        """
        query_embedding = self.embedding_model.encode([query]).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        similar_docs = [(self.documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return similar_docs