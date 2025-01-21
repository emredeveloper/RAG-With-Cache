import faiss
import numpy as np
from .embedding_model import EmbeddingModel

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
        self.index.add(embeddings.astype(np.float32))
    
    def retrieve(self, query, top_k=2):
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.documents[i], float(distances[0][j])) 
                for j, i in enumerate(indices[0])]