import os

# Cache klasörü yolu
CACHE_DIR = "model_cache"

# Cache klasörünü oluştur (eğer yoksa)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)