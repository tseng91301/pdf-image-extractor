import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class OfflineTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-zh-en", device=None):
        """
        初始化離線翻譯器。
        預設使用 Helsinki-NLP/opus-mt-zh-en 模型。
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                print("\033[93m[WARNING] [Translator] CUDA is not available. Falling back to CPU. This may be slow.\033[0m")
        else:
            self.device = device
            print(f"[Translator] Using specified device: {self.device}")
            
        print(f"[Translator] Loading model {model_name} on device {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("[Translator] Model loaded.")

    def contains_chinese(self, text: str) -> bool:
        """
        檢查字串中是否包含中文字元。
        """
        return bool(re.search(r"[\u4e00-\u9fa5]", text))

    def translate(self, text: str) -> str:
        """
        將中文翻譯成英文。如果輸入不含中文，則直接返回原文字。
        """
        if not text or not text.strip():
            return text
        
        if not self.contains_chinese(text):
            return text
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_tokens = self.model.generate(**inputs)
            
            translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            print(f"[Translator] Translation error: {e}")
            return text

if __name__ == "__main__":
    # 簡單測試
    translator = OfflineTranslator()
    test_text = "許多螞蟻聚集在水管裡的圖片"
    translated = translator.translate(test_text)
    print(f"Original: {test_text}")
    print(f"Translated: {translated}")
    
    test_en = "A group of ants on a pipe"
    print(f"English stay same: {translator.translate(test_en)}")
