"""Helpers for reading and chunking PDF documents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


class PDFLoader:
    """Load PDFs from a directory and chunk the extracted text."""

    def __init__(self, pdf_directory: Path | str = Path("data/pdfs")) -> None:
        self.pdf_directory = Path(pdf_directory)
        if not self.pdf_directory.exists():
            raise ValueError(f"PDF directory '{self.pdf_directory}' does not exist")
        self.logger = logging.getLogger(__name__)

    def load_pdfs(self) -> List[str]:
        """Read every PDF file in ``pdf_directory`` into memory."""
        texts: List[str] = []
        for path in sorted(self.pdf_directory.glob("*.pdf")):
            try:
                reader = PdfReader(path)
                text = " ".join(page.extract_text() or "" for page in reader.pages)
                if text.strip():
                    texts.append(text)
                else:
                    self.logger.warning("Empty PDF file: %s", path.name)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to read %s: %s", path.name, exc)
        return texts

    def chunk_text(
        self,
        texts: Iterable[str],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> List[str]:
        """Split the provided texts into overlapping chunks."""
        chunks: List[str] = []
        for text in texts:
            start = 0
            text_length = len(text)
            while start < text_length:
                end = min(start + chunk_size, text_length)
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += max(1, chunk_size - chunk_overlap)
        return chunks
