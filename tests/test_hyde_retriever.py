from model.hyde_retriever import HyDERetriever
import os

def test_hyde_retriever_initialization():
    # HyDE retriever'ın başlatılmasını test et
    retriever = HyDERetriever(
        files_path="data/pdfs",
        chunk_size=512,
        chunk_overlap=128,
        language_model_name="gpt2-medium",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
    assert retriever is not None, "HyDERetriever başlatılamadı!"

def test_hyde_retriever_generate_hypothetical_document():
    # HyDE ile hipotetik belge oluşturmayı test et
    retriever = HyDERetriever(
        files_path="data/pdfs",
        chunk_size=512,
        chunk_overlap=128,
        language_model_name="gpt2-medium",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
    hypothetical_doc = retriever.generate_hypothetical_document("What is AI?")
    assert isinstance(hypothetical_doc, str), "Hipotetik belge string değil!"
    assert len(hypothetical_doc) > 0, "Hipotetik belge boş!"

def test_hyde_retriever_retrieve():
    # HyDE retriever'ın belge getirmesini test et
    retriever = HyDERetriever(
        files_path="data/pdfs",
        chunk_size=512,
        chunk_overlap=128,
        language_model_name="gpt2-medium",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
    results, hypothetical_doc = retriever.retrieve("What is AI?", k=2)
    assert len(results) == 2, "HyDE retriever sonuç sayısı hatalı!"
    assert isinstance(hypothetical_doc, str), "Hipotetik belge string değil!"