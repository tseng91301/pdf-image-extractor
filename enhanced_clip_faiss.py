import jieba
import jieba.analyse
from clip_faiss import MultiModalRetriever

class EnhancedMultiModalRetriever(MultiModalRetriever):
    def search(
        self,
        query: str,
        topk=10,
        k_each=50,
        alpha=0.6,
        beta_title=0.7,
        beta_sur=0.3,
        gamma_keyword=0.15 # 關鍵字額外加權的分數佔比 (可根據需要微調)
    ):
        # 1. 先用原本的 FAISS 執行向量檢索 (取更多候選名單來做關鍵字重新排序)
        base_hits = super().search(
            query, 
            topk=max(topk * 3, k_each),  # 加大取樣範圍以利後續重新排序
            k_each=k_each, 
            alpha=alpha, 
            beta_title=beta_title, 
            beta_sur=beta_sur
        )
        
        # 2. 針對使用者的 query 提取關鍵字
        query_keywords = jieba.analyse.extract_tags(query, topK=5)
        if len(query_keywords) == 0:
            # 如果句子太短提不出關鍵字，退回一般的斷詞
            query_keywords = jieba.lcut(query)
            
        q_set = set(query_keywords)
        
        # 3. 對每筆回傳結果加算關鍵字交集配對分數
        for hit in base_hits:
            doc_keywords = hit.get("keywords", [])
            doc_set = set(doc_keywords)
            
            intersection = q_set.intersection(doc_set)
            
            # 計算交集比例
            if len(q_set) > 0:
                keyword_score = len(intersection) / len(q_set)
            else:
                keyword_score = 0.0
                
            hit["s_keyword"] = keyword_score
            hit["matched_keywords"] = list(intersection)
            
            # 公式: 重新分配權重，讓原本的向量分數與關鍵字分數組合
            # 原有的分數會依照 (1 - gamma) 的比例縮放，再疊加關鍵字分數
            hit["score"] = (hit["score"] * (1 - gamma_keyword)) + (keyword_score * gamma_keyword)
            
        # 4. 根據新的綜合分數重新排序
        base_hits.sort(key=lambda x: x["score"], reverse=True)
        return base_hits[:topk]

    @staticmethod
    def load(
        db_dir: str,
        text_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        image_model_name="clip-ViT-B-32",
    ):
        # 建立新的增強版物件
        r = EnhancedMultiModalRetriever(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
        )

        # 把載入底層索引的工作交由舊有方法 (偷吃步寫法，避免複製貼上一大堆 load 的邏輯)
        original = MultiModalRetriever.load(db_dir, text_model_name, image_model_name)
        r.title_index = original.title_index
        r.sur_index = original.sur_index
        r.img_index = original.img_index
        r.text_dim = original.text_dim
        r.image_dim = original.image_dim
        r.v_title = original.v_title
        r.v_img = original.v_img
        r.v_sur_chunks = original.v_sur_chunks
        r.sur_chunks_text = original.sur_chunks_text
        r.meta = original.meta
        
        return r
