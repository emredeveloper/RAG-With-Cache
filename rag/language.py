"""Language model wrapper with on-disk caching."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModel:
    """Minimal wrapper around HuggingFace causal language models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        safe_name = model_name.replace("/", "_")
        self.cache_path = Path("model_cache") / safe_name
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer, self.model = self._load_model()

    def _load_model(self):
        if any(self.cache_path.iterdir()):
            print(f"Loading model from cache: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.cache_path)
            model = AutoModelForCausalLM.from_pretrained(self.cache_path)
        else:
            print(f"Downloading and caching model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer.save_pretrained(self.cache_path)
            model.save_pretrained(self.cache_path)
        return tokenizer, model

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
