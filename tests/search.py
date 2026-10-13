import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------
# 1. 模擬文件中的圖片元資料（Metadata）
# ----------------------------------------------------
image_records = [
    {
        "image_id": "img_001",
        "file_path": "images/r7000_caliper.jpg",
        "caption": "圖 1：Shimano 105 BR-R7000 公路車雙樞軸煞車夾器特寫",
        "context": "安裝夾器時需確認快拆撥把處於關閉位置，並使用 5mm 內六角扳手以 8-10 Nm 扭力鎖固於前叉沉頭螺母。"
    },
    {
        "image_id": "img_002",
        "file_path": "images/disc_brake_pad.jpg",
        "caption": "圖 2：油壓碟煞來令片磨耗指示與更換示意圖",
        "context": "自行車制動系統失靈往往與來令片過度磨耗或碟盤沾染油污有關，厚度低於 0.5mm 時應即刻更換。"
    },
    {
        "image_id": "img_003",
        "file_path": "images/derailleur_tuning.jpg",
        "caption": "圖 3：後變速器極限螺栓（H/L 螺絲）調整位置圖解",
        "context": "若變速發生跳齒或鏈條掉入輪組輻條之間，應先微調導輪與飛輪最小齒片的對齊基準。"
    }
]

# ----------------------------------------------------
# 2. 建立索引層（BM25 全文索引 + Dense 向量索引）
# ----------------------------------------------------
# 2.1 準備全文檢索語料（結合 Caption 與 Context）
def tokenize(text):
    return list(jieba.cut(text))

corpus_texts = [f"{rec['caption']} {rec['context']}" for rec in image_records]
tokenized_corpus = [tokenize(text) for text in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# 2.2 準備語意向量（使用多語言開源模型，支援繁中/簡中）
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
dense_embeddings = model.encode(corpus_texts, normalize_embeddings=True)

# ----------------------------------------------------
# 3. 混合檢索與 RRF 排名演算法
# ----------------------------------------------------
def search_images(query: str, top_k: int = 2, rrf_k: int = 60):
    # --- 步驟 A: 全文檢索打分 (BM25) ---
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    # 依分數排序，取得 (doc_index, score)
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    
    # --- 步驟 B: 語意向量打分 (Cosine Similarity) ---
    query_embedding = model.encode([query], normalize_embeddings=True)
    # 因已正規化，點積即為餘弦相似度
    semantic_scores = np.dot(dense_embeddings, query_embedding.T).flatten()
    dense_ranked = sorted(enumerate(semantic_scores), key=lambda x: x[1], reverse=True)

    # --- 步驟 C: 倒數排名融合 (Reciprocal Rank Fusion, RRF) ---
    # 公式：Score_RRF = sum(1 / (k + rank))
    rrf_scores = {i: 0.0 for i in range(len(image_records))}

    for rank, (doc_idx, _) in enumerate(bm25_ranked):
        rrf_scores[doc_idx] += 1.0 / (rrf_k + (rank + 1))

    for rank, (doc_idx, _) in enumerate(dense_ranked):
        rrf_scores[doc_idx] += 1.0 / (rrf_k + (rank + 1))

    # --- 步驟 D: 產出最終結果 ---
    final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_idx, score in final_ranked:
        rec = image_records[doc_idx]
        results.append({
            "image_id": rec["image_id"],
            "file_path": rec["file_path"],
            "caption": rec["caption"],
            "rrf_score": score,
            "bm25_raw_score": float(bm25_scores[doc_idx]),
            "semantic_cosine": float(semantic_scores[doc_idx])
        })
    return results

# ----------------------------------------------------
# 4. 測試效果
# ----------------------------------------------------
if __name__ == "__main__":
    # 測試情境 1：專有名詞精準查詢（測試 BM25 優勢）
    print("=== 測試 1: 專有名詞查詢 ===")
    res1 = search_images("BR-R7000 扭力值")
    for r in res1:
        print(f"[{r['image_id']}] {r['caption']} (RRF: {r['rrf_score']:.4f}, BM25: {r['bm25_raw_score']:.2f})")

    # 測試情境 2：完全無相同字眼，全用語意同義詞查詢（測試 Dense 優勢）
    print("\n=== 測試 2: 概念與同義詞查詢 ===")
    res2 = search_images("腳踏車煞車皮磨損怎麼看")
    for r in res2:
        print(f"[{r['image_id']}] {r['caption']} (RRF: {r['rrf_score']:.4f}, Cosine: {r['semantic_cosine']:.4f})")