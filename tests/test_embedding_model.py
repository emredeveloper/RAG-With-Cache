from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from rag.embeddings import EmbeddingModel


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.saved = False

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saved = True

    def encode(self, texts: Iterable[str]):
        return np.ones((len(list(texts)), 4), dtype=np.float32)


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr("rag.embeddings.SentenceTransformer", DummySentenceTransformer)
    yield


def test_embedding_model_initialization(tmp_path):
    model = EmbeddingModel("test-model")
    assert model is not None


def test_embedding_model_encode(tmp_path):
    model = EmbeddingModel("test-model")
    embeddings = model.encode(["This is a test sentence."])
    assert len(embeddings) == 1
    assert embeddings[0].shape[0] == 4
