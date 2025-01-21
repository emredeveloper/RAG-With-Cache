from model.language_model import LanguageModel

def test_language_model_initialization():
    # Modelin başlatılmasını test et
    model = LanguageModel("gpt2")
    assert model is not None, "LanguageModel başlatılamadı!"

def test_language_model_generate():
    # Dil modelinin metin üretmesini test et
    model = LanguageModel("gpt2")
    prompt = "What is the capital of France?"
    response = model.generate(prompt, max_new_tokens=10)
    assert isinstance(response, str), "Model çıktısı string değil!"
    assert len(response) > 0, "Model çıktısı boş!"