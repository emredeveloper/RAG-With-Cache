from model.embedding_model import EmbeddingModel

def test_embedding_model_initialization():
    # Modelin başlatılmasını test et
    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    assert model is not None, "EmbeddingModel başlatılamadı!"

def test_embedding_model_encode():
    # Metinlerin vektörlere dönüştürülmesini test et
    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(["This is a test sentence."])
    assert len(embeddings) == 1, "Embedding boyutu hatalı!"
    assert len(embeddings[0]) == 384, "Embedding vektör boyutu hatalı!"  # all-MiniLM-L6-v2 için 384 boyutlu vektör