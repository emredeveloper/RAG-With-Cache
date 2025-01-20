import os
import shutil


def clear_cache():
    """
    Cache klasörünü temizler.
    """
    if os.path.exists("model_cache"):
        shutil.rmtree("model_cache")
        print("Cache klasörü temizlendi.")
    else:
        print("Cache klasörü zaten boş.")