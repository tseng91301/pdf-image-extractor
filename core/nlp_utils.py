import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
import numpy as np
from keybert import KeyBERT

# 定義一個正規表達式：抓取 1~4 個中文字，加上這些農業常見特定結尾
# 例如：炭疽病、斜紋夜蛾、紅蜘蛛蟎... 等等
AGRICULTURAL_PATTERN = re.compile(r'[\u4e00-\u9fa5]{1,4}(?:病|蟲|害|菌|蛾|蟎|蝨|蠅|蝶|農藥|肥料)')

# Unique KeyBERT wrappers cache to avoid wrapping SentenceTransformer multiple times
_kw_models_cache = {}

def get_keybert_model(transformer_model):
    if transformer_model is None:
        return None
    model_id = id(transformer_model)
    if model_id not in _kw_models_cache:
        _kw_models_cache[model_id] = KeyBERT(model=transformer_model)
    return _kw_models_cache[model_id]

def chunk_text_by_sentence(text: str, chunk_size: int = 100) -> list:
    """
    將長文本依據中文標點符號或換行分割成多個小段落（每個段落不超過 chunk_size 字元）。
    """
    if not text:
        return []
    sentences = re.split(r'([。？！\n；;])', text)
    chunks = []
    current_chunk = ""
    for item in sentences:
        if not item:
            continue
        if len(current_chunk) + len(item) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = item
        else:
            current_chunk += item
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def extract_keywords_single_chunk(text: str, topK: int = 8, allowPOS: tuple = ('n', 'nz', 'vn', 'ns'), model=None, device="cpu") -> list:
    """
    對單一小段文本進行關鍵字提取。
    在解析關鍵字之前，先過濾掉贅字與無意義的字詞 (只保留名詞、動詞、形容詞，且長度大於 1)。
    再將乾淨候選詞組合成脫水文本，最後使用 KeyBERT (基於傳入之 SentenceTransformer 模型) 進行語意相關度排名。
    """
    if not text or not text.strip():
        return []
    
    # 1. 詞性過濾 (消滅冗言贅字)，只留名詞、動詞、形容詞，且長度大於 1
    try:
        words = pseg.cut(text)
        allowed_poses = set(allowPOS) | {'nr', 'nt', 'v', 'a'}
        clean_candidates = [
            w.word for w in words if w.flag in allowed_poses and len(w.word) > 1
        ]
    except Exception:
        clean_candidates = [w for w in jieba.lcut(text) if len(w) > 1]

    # 2. 使用正則表達式抓取特定農業名詞
    special_matches = AGRICULTURAL_PATTERN.findall(text)
    
    # 合併為乾淨的候選詞清單
    candidates = list(set(clean_candidates + special_matches))
    candidates = [c for c in candidates if c.strip() and len(c) > 1]
    
    if not candidates:
        return []

    # 3. 建立「脫水文本」
    dehydrated_text = " ".join(candidates)

    # 4. 使用 KeyBERT 進行語意相關度排名
    if model is not None:
        try:
            kw_model = get_keybert_model(model)
            if kw_model is not None:
                keywords_with_scores = kw_model.extract_keywords(
                    dehydrated_text, 
                    candidates=candidates, 
                    use_mmr=True, 
                    diversity=0.5, 
                    top_n=topK
                )
                return [kw for kw, score in keywords_with_scores]
        except Exception as e:
            print(f"[nlp_utils] KeyBERT extraction failed, fallback to TF-IDF: {e}")

    # 5. Fallback/CPU 模式：使用 TF-IDF 粗篩排序
    try:
        jieba_keywords = jieba.analyse.extract_tags(text, topK=topK * 2)
        ranked = [kw for kw in jieba_keywords if kw in candidates]
        for kw in candidates:
            if kw not in ranked:
                ranked.append(kw)
        return ranked[:topK]
    except Exception:
        return candidates[:topK]

def extract_keywords(text: str, topK: int = 8, allowPOS: tuple = ('n', 'nz', 'vn', 'ns'), model=None, device="cpu") -> list:
    """
    統一關鍵字提取邏輯。對長文先做分割處理，分別對每個小段取出 keywords 之後，再把他們結合起來（相同文字不要重複）。
    """
    if not text or not text.strip():
        return []
        
    # 如果長度大於 120，先進行切片
    if len(text) > 120:
        chunks = chunk_text_by_sentence(text, chunk_size=100)
        combined_keywords = []
        for chunk in chunks:
            kws = extract_keywords_single_chunk(chunk, topK=topK, allowPOS=allowPOS, model=model, device=device)
            combined_keywords.extend(kws)
            
        # 移除重複關鍵字並保持順序
        seen = set()
        final_keywords = []
        for kw in combined_keywords:
            if kw not in seen:
                seen.add(kw)
                final_keywords.append(kw)
        return final_keywords
    else:
        return extract_keywords_single_chunk(text, topK=topK, allowPOS=allowPOS, model=model, device=device)

