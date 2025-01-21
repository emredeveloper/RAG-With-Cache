class Config:
    # Enhanced HyDE configuration
    HYDE_SETTINGS = {
        "chunk_size": 512,
        "chunk_overlap": 128,
        "top_k": 3,
        "language_model": "gpt2-medium",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "similarity_metric": "cosine"
    }
    
    # Model options
    EMBEDDING_MODELS = {
        "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    }
    
    LANGUAGE_MODELS = {
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "flan-t5": "google/flan-t5-large"
    }

    # PDF Ayarları
    PDF_DIRECTORY = "data/pdfs"  # PDF'lerin bulunduğu dizin
    CHUNK_SIZE = 500  # Metin parçalama boyutu

    # Retriever seçenekleri
    RETRIEVER_OPTIONS = [
        ("FAISS", "faiss"),
        ("HyDE", "hyde")
    ]

    # Varsayılan ayarlar
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    DEFAULT_LANGUAGE_MODEL = "gpt2-medium"
    DEFAULT_RETRIEVER = "faiss"  # Varsayılan retriever
    TOP_K = 3  # Benzer belge sayısı

    # HyDE Ayarları
    HYDE_CHUNK_SIZE = HYDE_SETTINGS["chunk_size"]
    HYDE_CHUNK_OVERLAP = HYDE_SETTINGS["chunk_overlap"]