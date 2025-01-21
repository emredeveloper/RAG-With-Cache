from data_loader.pdf_loader import PDFLoader
import os

def test_pdf_loader_initialization():
    # PDFLoader'ın başlatılmasını test et
    pdf_loader = PDFLoader("data/pdfs")
    assert pdf_loader is not None, "PDFLoader başlatılamadı!"

def test_pdf_loader_load_pdfs():
    # PDF'lerin yüklenmesini test et
    pdf_loader = PDFLoader("data/pdfs")
    texts = pdf_loader.load_pdfs()
    assert isinstance(texts, list), "PDF'ler listeye dönüştürülemedi!"
    assert len(texts) > 0, "PDF'ler yüklenemedi!"

def test_pdf_loader_chunk_text():
    # Metinlerin parçalara ayrılmasını test et
    pdf_loader = PDFLoader("data/pdfs")
    texts = ["This is a long text that needs to be chunked." * 100]
    chunks = pdf_loader.chunk_text(texts, chunk_size=500, chunk_overlap=100)
    assert isinstance(chunks, list), "Parçalar listeye dönüştürülemedi!"
    assert len(chunks) > 1, "Metinler parçalara ayrılamadı!"