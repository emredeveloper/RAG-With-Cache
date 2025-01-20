# config.py
class Config:
    # Embedding model seçenekleri
    EMBEDDING_MODELS = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ("paraphrase-multilingual-MiniLM-L12-v2",
         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2")
    ]

    # Dil modeli seçenekleri
    LANGUAGE_MODELS = [
        ("GPT-2", "gpt2"),
        ("GPT-Neo (125M)", "EleutherAI/gpt-neo-125M"),
        ("FLAN-T5 (Small)", "google/flan-t5-small")
    ]

    # Varsayılan ayarlar
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_LANGUAGE_MODEL = "gpt2"
    TOP_K = 2
    MAX_TOKENS = 50
