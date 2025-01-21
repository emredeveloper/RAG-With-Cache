import faiss
import numpy as np
from langchain.prompts import PromptTemplate
from model.language_model import LanguageModel
from model.embedding_model import EmbeddingModel
from data_loader.pdf_loader import PDFLoader
from typing import List, Tuple
import os

class HyDERetriever:
    def __init__(self, 
                 files_path: str,
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 language_model_name: str = "gpt2-medium",
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        
        # Model initialization with configurable parameters
        self.llm = LanguageModel(language_model_name)
        self.embeddings = EmbeddingModel(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Document processing and indexing
        self.index, self.chunks = self._encode_pdfs(files_path)
        
        # Enhanced HyDE prompt template
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Generate a comprehensive hypothetical document that answers the following question. 
            The document should be written in formal academic style and contain exactly {chunk_size} characters.
            
            Question: {query}
            
            Hypothetical Document:"""
        )

    def _encode_pdfs(self, files_path: str) -> Tuple[faiss.Index, List[str]]:
        """Process PDFs and create FAISS index with error handling"""
        if not os.path.exists(files_path):
            raise FileNotFoundError(f"PDF directory not found: {files_path}")
            
        pdf_loader = PDFLoader(files_path)
        documents = pdf_loader.load_pdfs()
        
        # Improved chunking with overlap
        chunks = pdf_loader.chunk_text(documents, self.chunk_size, self.chunk_overlap)
        
        # Enhanced embedding processing
        embeddings = self.embeddings.encode(chunks)
        dimension = embeddings.shape[1]
        
        # FAISS index configuration
        index = faiss.IndexFlatIP(dimension)  # Using inner product for similarity
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        index.add(embeddings.astype(np.float32))
        
        return index, chunks

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document with error handling"""
        try:
            input_variables = {"query": query, "chunk_size": self.chunk_size}
            prompt = self.hyde_prompt.format(**input_variables)
            return self.llm.generate(prompt, max_new_tokens=self.chunk_size)
        except Exception as e:
            raise RuntimeError(f"Hypothetical document generation failed: {str(e)}")

    def retrieve(self, query: str, k: int = 3) -> Tuple[List[Tuple[str, float]], str]:
        """Enhanced retrieval with similarity scoring"""
        hypothetical_doc = self.generate_hypothetical_document(query)
        hypothetical_embedding = self.embeddings.encode([hypothetical_doc])
        faiss.normalize_L2(hypothetical_embedding)
        
        # Perform similarity search
        scores, indices = self.index.search(hypothetical_embedding.astype(np.float32), k)
        
        # Convert to cosine similarity scores
        cosine_similarities = (scores + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        similar_docs = [
            (self.chunks[i], float(cosine_similarities[0][j]))
            for j, i in enumerate(indices[0])
        ]
        
        return similar_docs, hypothetical_doc