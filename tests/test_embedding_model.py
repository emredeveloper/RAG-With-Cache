# tests/test_embedding_model.py
from model.embedding_model import EmbeddingModel

def test_embedding_model_initialization():
    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    assert model is not None

def test_embedding_model_encode():
    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(["This is a test sentence."])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0  # Embedding vektörünün boyutu kontrolü