from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import numpy as np
from pydantic import Field, model_validator

class LangChainRetriever(BaseRetriever):
    embeddings: HuggingFaceEmbeddings = Field(default=None, exclude=True)
    documents: List[str] = Field(default_factory=list)
    index: np.ndarray = Field(default=None, exclude=True)

    def __init__(self, embedding_model_name: str, documents: List[str]):
        super().__init__()
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.documents = documents
        self.index = self._build_index()

    def _build_index(self):
        doc_embeddings = self.embeddings.embed_documents(self.documents)
        return np.array(doc_embeddings)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)
        similarities = np.dot(self.index, query_embedding)
        indices = np.argsort(similarities)[::-1][:5]
        return [
            Document(
                page_content=self.documents[i],
                metadata={"similarity": float(similarities[i])}
            ) for i in indices
        ]