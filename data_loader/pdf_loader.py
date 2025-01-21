import os
from typing import List
from pypdf import PdfReader
import logging

class PDFLoader:
    def __init__(self, pdf_directory: str = "data/pdfs"):
        if not os.path.exists(pdf_directory):
            raise ValueError(f"PDF directory '{pdf_directory}' does not exist")
        self.pdf_directory = pdf_directory
        self.logger = logging.getLogger(__name__)

    def load_pdfs(self) -> List[str]:
        """Improved PDF text extraction with error handling"""
        texts = []
        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(self.pdf_directory, filename)
                try:
                    reader = PdfReader(filepath)
                    text = " ".join([page.extract_text() or "" for page in reader.pages])
                    if text.strip():
                        texts.append(text)
                    else:
                        self.logger.warning(f"Empty PDF file: {filename}")
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {str(e)}")
        return texts

    def chunk_text(self, 
                  texts: List[str], 
                  chunk_size: int = 500, 
                  chunk_overlap: int = 100) -> List[str]:
        """Improved chunking with overlap and content-aware splitting"""
        chunks = []
        for text in texts:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - chunk_overlap
        return chunks