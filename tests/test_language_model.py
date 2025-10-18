from pathlib import Path

import pytest
import torch

from rag.language import LanguageModel


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def generate(self, input_ids, attention_mask=None, pad_token_id=None, max_new_tokens=None, num_return_sequences=None):
        return torch.tensor([[1, 2, 3, 4]])


@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    monkeypatch.setattr("rag.language.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("rag.language.AutoModelForCausalLM", DummyModel)
    yield


def test_language_model_initialization():
    model = LanguageModel("gpt2")
    assert model is not None


def test_language_model_generate():
    model = LanguageModel("gpt2")
    prompt = "Hello, how are you?"
    response = model.generate(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
