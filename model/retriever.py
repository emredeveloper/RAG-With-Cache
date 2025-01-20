# retriever.py
import faiss
import numpy as np
from .embedding_model import EmbeddingModel  # Göreceli içe aktarma

class Retriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = None
    
    def build_index(self, documents):
        self.documents = documents
        embeddings = self.embedding_model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def retrieve(self, query, top_k=2):
        query_embedding = self.embedding_model.encode([query]).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        similar_docs = [(self.documents[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return similar_docs