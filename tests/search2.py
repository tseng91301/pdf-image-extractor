import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# 1. 模擬圖文資料庫（包含 Caption 與 Context）
# ==========================================
documents = [
    {
        "id": "IMG_001",
        "caption": "Shimano 105 BR-R7000 公路車煞車夾器特寫",
        "context": "安裝公路車夾器時，建議使用 5mm 內六角扳手以 8-10 Nm 扭力鎖固前叉沉頭螺母，並確認快拆撥把處於定位。"
    },
    {
        "id": "IMG_002",
        "caption": "公路車前後碳纖維輪組與煞車邊保養指南",
        "context": "碳纖維輪框煞車邊若沾染油污會導致煞車制動失靈，需定期使用專用去漬油清潔，並檢視來令片磨耗狀況。"
    },
    {
        "id": "IMG_003",
        "caption": "公路車後變速器 H/L 極限螺栓調整圖解",
        "context": "變速若出現跳齒或異音，先檢視導輪是否對齊飛輪最小齒片，調整張力螺絲前確保變速內線拉緊。"
    },
    {
        "id": "IMG_004",
        "caption": "公路車煞車一般保養與定期檢查項目總覽",
        "context": "本手冊涵蓋公路車煞車日常檢查步驟，注意：本節不包含特定的安裝鎖固扭力規格，僅提供通用外觀檢視流程。"
    }
]

# ==========================================
# 2. 載入模型
# ==========================================
print("載入模型中...")
# 第一階段：粗篩用的雙塔 Embedding 模型
bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
# 第二階段：精排用的 Cross-Encoder Reranker 模型
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

# ==========================================
# 3. 建立索引（BM25 + 向量）
# ==========================================
# 3.1 準備 BM25 語料（Caption 出現兩次以達成加權效果）
def tokenize(text):
    r = list(jieba.cut(text))
    print(f"Tokenize: {text} -> {r}")
    return r

bm25_corpus = [
    tokenize(f"{doc['caption']} {doc['caption']} {doc['context']}")
    for doc in documents
]
bm25 = BM25Okapi(bm25_corpus)

# 3.2 準備向量語意（Caption 與 Context 分開編碼，保留標題高純度）
caption_texts = [doc["caption"] for doc in documents]
context_texts = [doc["context"] for doc in documents]

caption_embeddings = bi_encoder.encode(caption_texts, normalize_embeddings=True)
context_embeddings = bi_encoder.encode(context_texts, normalize_embeddings=True)

# ==========================================
# 4. 兩階段搜尋主邏輯
# ==========================================
def search(query: str, recall_k: int = 3, top_n: int = 2):
    print(f"\n==========================================")
    print(f"使用者查詢：【{query}】")
    print(f"==========================================")

    # ------------------------------------------
    # 【階段一：粗篩召回 (BM25 + 向量加權)】
    # ------------------------------------------
    # A. BM25 打分
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # B. 向量打分（Caption 權重 2.0，Context 權重 1.0）
    query_vec = bi_encoder.encode([query], normalize_embeddings=True)
    sim_captions = np.dot(caption_embeddings, query_vec.T).flatten()
    sim_contexts = np.dot(context_embeddings, query_vec.T).flatten()
    dense_scores = 2.0 * sim_captions + 1.0 * sim_contexts

    # C. 簡單的 RRF 排名融合，撈出前 recall_k 名候選
    rrf_k = 60
    rrf_scores = {i: 0.0 for i in range(len(documents))}
    
    # 依 BM25 名次計分
    for rank, idx in enumerate(np.argsort(bm25_scores)[::-1]):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
    # 依向量名次計分
    for rank, idx in enumerate(np.argsort(dense_scores)[::-1]):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

    # 取得前 recall_k 個候選者
    candidate_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:recall_k]
    
    print(f"\n--- [階段一 粗篩結果] (取前 {recall_k} 名進決賽) ---")
    for idx in candidate_indices:
        doc = documents[idx]
        print(f"[{doc['id']}] {doc['caption']} (BM25: {bm25_scores[idx]:.2f}, 向量: {dense_scores[idx]:.2f})")

    # ------------------------------------------
    # 【階段二：精排終審 (Cross-Encoder)】
    # ------------------------------------------
    # 將 Query 與各候選者的完整內容拼接成配對
    pairs = [
        [query, f"{documents[idx]['caption']}。{documents[idx]['context']}"]
        for idx in candidate_indices
    ]
    
    # Cross-Encoder 一對一交叉注意力評分（輸出關聯度 Logit/機率）
    rerank_scores = cross_encoder.predict(pairs)

    # 結合候選索引與精排分數
    final_candidates = list(zip(candidate_indices, rerank_scores))
    final_candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- [階段二 Cross-Encoder 審查最終結果] (Top {top_n}) ---")
    for rank, (idx, score) in enumerate(final_candidates[:top_n], start=1):
        doc = documents[idx]
        print(f"第 {rank} 名 (相關度分數: {score:.4f})")
        print(f"  編號: {doc['id']}")
        print(f"  標題: {doc['caption']}")
        print(f"  內文: {doc['context']}")


# ==========================================
# 5. 測試驗證
# ==========================================
if __name__ == "__main__":
    # 測試情境：搜尋明確型號與操作細節（帶有高度特定性）
    search("BR-R7000 夾器螺母鎖固扭力是多少", recall_k=3, top_n=2)
    
    # 測試情境：使用模糊口語（同義詞），考驗概念匹配
    search("腳踏車煞車皮沾到油怎麼處理", recall_k=3, top_n=2)