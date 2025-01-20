from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.cache_path = os.path.join("model_cache", model_name.replace("/", "_"))
        self.tokenizer, self.model = self._load_model()

    def _load_model(self):
        """
        Modeli ve tokenizer'ı önbellekten yükler veya internetten indirip önbelleğe kaydeder.
        """
        if os.path.exists(self.cache_path):
            print(f"Model önbellekten yükleniyor: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.cache_path)
            model = AutoModelForCausalLM.from_pretrained(self.cache_path)
        else:
            print(f"Model indiriliyor ve önbelleğe kaydediliyor: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer.save_pretrained(self.cache_path)
            model.save_pretrained(self.cache_path)
        return tokenizer, model

    def generate(self, prompt, max_new_tokens=50):
        """
        Verilen prompt'a göre cevap üretir.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

