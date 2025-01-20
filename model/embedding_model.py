# embedding_model.py
from sentence_transformers import SentenceTransformer
import os

CACHE_DIR = "model_cache"

class EmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.cache_path = os.path.join(CACHE_DIR, model_name.replace("/", "_"))
        self.model = self._load_model()

    def _load_model(self):
        """
        Modeli önbellekten yükler veya internetten indirip önbelleğe kaydeder.
        """
        if os.path.exists(self.cache_path):
            print(f"Model önbellekten yükleniyor: {self.model_name}")
            return SentenceTransformer(self.cache_path)
        else:
            print(f"Model indiriliyor ve önbelleğe kaydediliyor: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            model.save(self.cache_path)
            return model

    def encode(self, texts):
        """
        Metinleri vektörlere dönüştürür.
        """
        return self.model.encode(texts)