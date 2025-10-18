from pathlib import Path

import pytest
from pypdf import PdfWriter

from rag.data.pdf_loader import PDFLoader


def create_pdf_file(path: Path, content: str) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as fp:
        writer.write(fp)


@pytest.fixture()
def pdf_dir(tmp_path: Path) -> Path:
    pdf_file = tmp_path / "sample.pdf"
    create_pdf_file(pdf_file, "Sample text")
    return tmp_path


def test_pdf_loader_initialization(pdf_dir: Path):
    loader = PDFLoader(pdf_dir)
    assert loader is not None, "PDFLoader could not be initialized!"


def test_pdf_loader_load_pdfs(pdf_dir: Path):
    loader = PDFLoader(pdf_dir)
    texts = loader.load_pdfs()
    assert isinstance(texts, list), "Loaded PDFs should be returned as a list!"


def test_pdf_loader_chunk_text():
    loader = PDFLoader("data/pdfs")
    texts = ["This is a sample text for testing PDF chunking."]
    chunks = loader.chunk_text(texts, chunk_size=10, chunk_overlap=2)
    assert isinstance(chunks, list), "Chunked texts should be returned as a list!"
    assert len(chunks) > 0, "Chunks should not be empty!"
