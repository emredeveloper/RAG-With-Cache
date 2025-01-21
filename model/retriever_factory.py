from .retriever import Retriever
from .embedding_model import EmbeddingModel  # EmbeddingModel sınıfını içe aktar

class RetrieverFactory:
    @staticmethod
    def create_retriever(config, embedding_model_name: str, documents: list):
        if config.DEFAULT_RETRIEVER == "faiss":
            embedding_model = EmbeddingModel(embedding_model_name)  # EmbeddingModel sınıfını kullan
            retriever = Retriever(embedding_model)
            retriever.build_index(documents)
            return retriever
        else:
            raise ValueError(f"Geçersiz retriever: {config.DEFAULT_RETRIEVER}")