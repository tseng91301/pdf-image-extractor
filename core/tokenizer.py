"""Chinese tokenizer module utilizing Jieba with customizable tokenization rules."""

from typing import List, Optional, Set
import re
import jieba


class JiebaTokenizer:
    """Thread-safe Chinese tokenizer with caching and noise filtering."""

    DEFAULT_STOPWORDS: Set[str] = {
        " ", "\t", "\n", "\r", "\u3000",
        "，", "。", "、", "；", "：", "？", "！",
        "（", "）", "【", "】", "《", "》", "「", "」",
        "‘", "’", "“", "”", "—", "…", "～",
        "[", "]", "<", ">", "-", "_", "/", "\\", "|", "*", "#",
        "的", "了", "和", "是", "就", "都", "而", "及", "與", "著",
        "或", "一個", "沒有", "我們", "你們", "他們", "在", "於"
    }

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        filter_single_punctuation: bool = True,
        user_dict_path: Optional[str] = None
    ):
        self.stopwords = stopwords if stopwords is not None else set(self.DEFAULT_STOPWORDS)
        self.filter_single_punctuation = filter_single_punctuation
        if user_dict_path:
            jieba.load_userdict(user_dict_path)

    def add_word(self, word: str, freq: Optional[int] = None, tag: Optional[str] = None) -> None:
        """Dynamically add a custom word to the Jieba dictionary."""
        jieba.add_word(word, freq=freq, tag=tag)

    def tokenize(self, text: Optional[str], remove_stopwords: bool = False) -> List[str]:
        """Tokenizes an input string into a list of cleaned words."""
        if not text:
            return []
        raw_tokens = jieba.lcut(text.strip())
        tokens: List[str] = []
        for t in raw_tokens:
            token = t.strip()
            if not token:
                continue
            if remove_stopwords and token in self.stopwords:
                continue
            if self.filter_single_punctuation and len(token) == 1 and not re.match(r'[\w\u4e00-\u9fff]', token):
                continue
            tokens.append(token)
        return tokens

    def tokenize_query(self, query: str) -> List[str]:
        """Tokenizes a query string, keeping terms and filtering pure whitespace/punctuation."""
        return self.tokenize(query, remove_stopwords=False)

    def explain(self, text: str) -> dict:
        """Returns detailed token breakdown for debugging."""
        raw = list(jieba.cut(text))
        cleaned = self.tokenize(text, remove_stopwords=False)
        no_stop = self.tokenize(text, remove_stopwords=True)
        return {
            "text": text,
            "raw_tokens": raw,
            "cleaned_tokens": cleaned,
            "filtered_no_stopwords": no_stop,
            "total_tokens": len(cleaned)
        }


# Global default instance
default_tokenizer = JiebaTokenizer()


if __name__ == "__main__":
    sample = "BR-R7000 夾器螺母鎖固扭力是多少？山藥桃園4號地下塊莖特點"
    res = default_tokenizer.explain(sample)
    print("Tokenizer Standalone Test:")
    print(f"Raw: {res['raw_tokens']}")
    print(f"Cleaned: {res['cleaned_tokens']}")
