import shutil

def clear_cache():
    """
    Cache klasörünü temizler.
    """
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("Cache klasörü temizlendi.")
    else:
        print("Cache klasörü zaten boş.")