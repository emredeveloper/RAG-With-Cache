# tests/test_language_model.py
from model.language_model import LanguageModel

def test_language_model_initialization():
    model = LanguageModel("gpt2")
    assert model is not None

def test_language_model_generate():
    model = LanguageModel("gpt2")
    prompt = "What is the capital of France?"
    response = model.generate(prompt, max_new_tokens=10)
    assert isinstance(response, str)
    assert len(response) > 0