# rag_system.py
from .embedding_model import EmbeddingModel
from .retriever import Retriever
from .language_model import LanguageModel


class RAGSystem:
    def __init__(self, embedding_model, retriever, language_model):
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.language_model = language_model
    
    def answer_question(self, query, top_k=2):
        similar_docs = self.retriever.retrieve(query, top_k)
        prompt = self._create_prompt(query, similar_docs)
        answer = self.language_model.generate(prompt)
        return answer
    
    def _create_prompt(self, query, similar_docs):
        context = "\n".join([f"Document {i+1}: {doc}" for i, (doc, score) in enumerate(similar_docs)])
        prompt = f"""Answer the following question based on the provided documents:
        
Question: {query}

Documents:
{context}

Answer:"""
        return prompt