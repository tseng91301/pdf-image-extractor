import jieba.posseg as pseg
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# 1. 初始化強大的 Qwen3 長文本模型 (用作最終提取)
hf_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
kw_model = KeyBERT(model=hf_model)

def smart_filter_pipeline(text, title=""):
    """
    智慧型長文關鍵字過濾管線
    """
    # ====== 步驟 1：詞性過濾 (消滅冗言贅字) ======
    words = pseg.cut(text)
    allowed_poses = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'a'} # 只留名詞、動詞、形容詞
    
    # 建立「乾淨的候選詞清單」，長度大於 1 的才留下來 (剔除單個字)
    clean_candidates = list(set([
        w.word for w in words if w.flag in allowed_poses and len(w.word) > 1
    ]))
    
    # ====== 步驟 2：段落粗篩 (消滅離題噪音) ======
    # 如果字數真的太多，可以進行段落過濾；若在模型限制內，可直接將過濾後的詞合回
    # 這裡我們直接把詞用空格連起來，做成「脫水文本」
    # 因為 clean_candidates 已經沒有贅字了，合起來的文字語意密度極高！
    dehydrated_text = " ".join(clean_candidates)
    
    # ====== 步驟 3：交給 KeyBERT 進行最終語意比對 ======
    keywords = kw_model.extract_keywords(
        dehydrated_text, 
        candidates=clean_candidates, # 限制模型只能從這些乾淨的詞裡面選
        use_mmr=True, 
        diversity=0.5, 
        top_n=5
    )
    
    return keywords

# 測試用長文本
article = """
各位觀眾大家好，今天我們非常榮幸能夠在這裡跟大家一起聊聊關於人工智慧的未來發展。
其實呢，這個技術在最近這幾年真的是迎來了非常爆發性的成長，而且各大科技巨頭都投了超多錢。
在深度學習的領域中，神經網路架構與自然語言處理（NLP）已經成為了最核心的驅動力。
不過呢，我們在實務上架設系統的時候，往往會遇到很多莫名其妙的硬體效能瓶頸，這點真的非常讓人頭痛。
綜合上述所說的，總之，大語言模型正在改變世界。
"""

result = smart_filter_pipeline(article)
print("🎯 最終精準關鍵字：")
for kw, score in result:
    print(f"- {kw} (分數: {score:.4f})")
