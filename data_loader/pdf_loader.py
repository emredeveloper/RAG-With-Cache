import os
from typing import List
from PyPDF2 import PdfReader

class PDFLoader:
    def __init__(self, pdf_directory: str = "data/pdfs"):
        self.pdf_directory = pdf_directory

    def load_pdfs(self) -> List[str]:
        """PDF'lerden metin çıkarır (OCR olmadan)."""
        texts = []
        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(self.pdf_directory, filename)
                try:
                    reader = PdfReader(filepath)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:  # Boş olmayan metinleri ekle
                            texts.append(text)
                except Exception as e:
                    print(f"PDF okuma hatası ({filepath}): {str(e)}")
        return texts

    def chunk_text(self, texts: List[str], chunk_size: int = 500) -> List[str]:
        """Metinleri parçalara böler."""
        chunks = []
        for text in texts:
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunks.append(chunk)
        return chunks