import os, json, re
import numpy as np
from PIL import Image
import faiss
import jieba
from sentence_transformers import SentenceTransformer
from .nlp_utils import extract_keywords


# ----------------------------
# Utils
# ----------------------------
def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text, chunk_size=120, overlap=30):
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        start += step
    return chunks


# ----------------------------
# Retriever
# ----------------------------
class MultiModalRetriever:
    text_model_name: str
    image_model_name: str
    def __init__(
        self,
        text_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        image_model_name="clip-ViT-B-32",
    ):
        # models
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_model = SentenceTransformer(text_model_name)
        self.image_model = SentenceTransformer(image_model_name)

        # FAISS indices
        self.title_index = None     # text
        self.sur_index = None       # text (pooled, only for recall)
        self.img_index = None       # image

        # dims
        self.text_dim = None
        self.image_dim = None

        # stored vectors
        self.v_title = None
        self.v_img = None
        self.v_sur_chunks = []      # list[np.ndarray], per image (n_chunks, text_dim)
        self.sur_chunks_text = []   # list[list[str]]

        self.meta = []

    # ----------------------------
    # Index init
    # ----------------------------
    def _ensure_index(self, text_dim: int, image_dim: int):
        if self.title_index is None:
            self.title_index = faiss.IndexFlatIP(text_dim)
            self.sur_index = faiss.IndexFlatIP(text_dim)
            self.text_dim = text_dim

        if self.img_index is None:
            self.img_index = faiss.IndexFlatIP(image_dim)
            self.image_dim = image_dim

    # ----------------------------
    # Add documents
    # ----------------------------
    def add_document(
        self,
        json_path: str,
        images_dir: str,
        n_sur=3,
        doc_name_override=None,
        chunk_size=20,
        overlap=4,
    ):
        data = json.load(open(json_path, "r", encoding="utf-8"))
        imgs = data["imgs"]

        titles = []
        image_paths = []
        sur_chunks_text_list = []
        new_meta = []

        for it in imgs:
            img_name = it["name"]
            png_path = os.path.join(images_dir, f"{img_name}.png")
            if not os.path.exists(png_path):
                continue

            fig_title = normalize_text(it.get("figure_title", ""))

            sur_list = (it.get("surrounding_texts", []) or [])[:n_sur]
            sur_list = [normalize_text(x) for x in sur_list if x.strip()]

            sur_chunks = []
            for sur in sur_list:
                sur_chunks.extend(chunk_text(sur, chunk_size, overlap))

            titles.append(fig_title)
            image_paths.append(png_path)
            sur_chunks_text_list.append(sur_chunks)

            meta_item = {
                "doc_name": doc_name_override or data.get("name"),
                "uid": data.get("uid"),
                "page": it.get("page"),
                "image_name": img_name,
                "image_path": png_path,
                "coordinate": it.get("coordinate"),
                "figure_title": fig_title,
                "sur_text_list": sur_list,
                "sur_chunks_used": sur_chunks,
            }
            
            # --- keyword extraction ---
            text_source = (fig_title + " " + " ".join(sur_list)).strip()
            meta_item["keywords"] = extract_keywords(text_source, topK=8)
            
            new_meta.append(meta_item)

        if not image_paths:
            return 0

        # ---- text embeddings (title) ----
        v_title = self.text_model.encode(
            titles,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        # ---- text embeddings (surrounding chunks) ----
        flat_chunks = []
        chunk_ranges = []
        cur = 0
        for chunks in sur_chunks_text_list:
            flat_chunks.extend(chunks)
            chunk_ranges.append((cur, cur + len(chunks)))
            cur += len(chunks)

        if flat_chunks:
            v_sur_all = self.text_model.encode(
                flat_chunks,
                batch_size=128,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
        else:
            v_sur_all = None

        # ---- image embeddings ----
        pil_imgs = [Image.open(p).convert("RGB") for p in image_paths]
        v_img = self.image_model.encode(
            pil_imgs,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        # ---- ensure indices ----
        self._ensure_index(v_title.shape[1], v_img.shape[1])

        # ---- add to indices ----
        self.title_index.add(v_title)
        self.img_index.add(v_img)

        # pooled sur for recall (mean, only for candidate recall)
        if v_sur_all is not None:
            v_sur_pool = []
            for a, b in chunk_ranges:
                if a == b:
                    v_sur_pool.append(np.zeros((self.text_dim,), dtype="float32"))
                else:
                    v_sur_pool.append(v_sur_all[a:b].mean(axis=0))
            v_sur_pool = np.vstack(v_sur_pool)
        else:
            v_sur_pool = np.zeros((len(image_paths), self.text_dim), dtype="float32")

        self.sur_index.add(v_sur_pool)

        # ---- store vectors ----
        self.v_title = v_title if self.v_title is None else np.vstack([self.v_title, v_title])
        self.v_img = v_img if self.v_img is None else np.vstack([self.v_img, v_img])

        for i, (a, b) in enumerate(chunk_ranges):
            if v_sur_all is None or a == b:
                self.v_sur_chunks.append(np.zeros((0, self.text_dim), dtype="float32"))
                self.sur_chunks_text.append([])
            else:
                self.v_sur_chunks.append(v_sur_all[a:b])
                self.sur_chunks_text.append(sur_chunks_text_list[i])

        self.meta.extend(new_meta)
        return len(new_meta)

    def add_folder(
        self,
        folder_path: str,
        n_sur=3,
        doc_name_override=None,
        chunk_size=20,
        overlap=4,
    ):
        """
        直接新增一個資料夾，結構為:
        {folder_path}/
            metadata.json
            image_0000.png
            image_0001.png
            ...
        """
        return self.add_document(
            os.path.join(folder_path, "metadata.json"),
            folder_path,
            n_sur=n_sur,
            doc_name_override=doc_name_override,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    # ----------------------------
    # Reset + build
    # ----------------------------
    def build(self, json_path: str, images_dir: str, n_sur=3):
        self.title_index = None
        self.sur_index = None
        self.img_index = None
        self.text_dim = None
        self.image_dim = None

        self.v_title = None
        self.v_img = None
        self.v_sur_chunks = []
        self.sur_chunks_text = []
        self.meta = []

        return self.add_document(json_path, images_dir, n_sur=n_sur)

    # ----------------------------
    # Search
    # ----------------------------
    def search(
        self,
        query: str,
        topk=10,
        k_each=50,
        alpha=0.6,
        beta_title=0.7,
        beta_sur=0.3,
        gamma_keyword=0.4, # Weight for keyword matching (0.0 to disable)
    ):
        if self.title_index is None:
            raise RuntimeError("Index not built")

        # normalize weights
        s = beta_title + beta_sur
        beta_title /= s
        beta_sur /= s

        # encode query
        q_text = self.text_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        q_img = self.image_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # recall (if gamma_keyword > 0, recall more for re-ranking)
        k_recall = max(topk * 3, k_each) if gamma_keyword > 0 else k_each
        _, It = self.title_index.search(q_text, k_recall)
        _, Is = self.sur_index.search(q_text, k_recall)
        _, Ii = self.img_index.search(q_img, k_recall)

        cand = set(It[0]) | set(Is[0]) | set(Ii[0])

        results = []
        for idx in cand:
            s_title = float(np.dot(q_text[0], self.v_title[idx]))

            v_chunks = self.v_sur_chunks[idx]
            if v_chunks.shape[0] == 0:
                s_sur = 0.0
                best_chunk = None
            else:
                scores = v_chunks @ q_text[0]
                best_i = int(np.argmax(scores))
                s_sur = float(scores[best_i])
                best_chunk = self.sur_chunks_text[idx][best_i]

            s_img = float(np.dot(q_img[0], self.v_img[idx]))

            s_text = beta_title * s_title + beta_sur * s_sur
            score = alpha * s_text + (1 - alpha) * s_img

            m = self.meta[idx]
            hit = {
                "score": score,
                "s_text": s_text,
                "s_title": s_title,
                "s_sur": s_sur,
                "s_img": s_img,
                "best_sur_chunk": best_chunk,
                **m,
            }
            
            # --- keyword re-ranking ---
            if gamma_keyword > 0:
                query_keywords = extract_keywords(query, topK=5)
                if len(query_keywords) == 0:
                    query_keywords = jieba.lcut(query)
                
                q_set = set(query_keywords)
                doc_keywords = m.get("keywords", [])
                doc_set = set(doc_keywords)
                
                intersection = q_set.intersection(doc_set)
                keyword_score = len(intersection) / len(q_set) if len(q_set) > 0 else 0.0
                
                hit["s_keyword"] = keyword_score
                hit["matched_keywords"] = list(intersection)
                # Combine score
                hit["score"] = (hit["score"] * (1 - gamma_keyword)) + (keyword_score * gamma_keyword)
            
            results.append(hit)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:topk]

    # ----------------------------
    # Save/Load
    # ----------------------------
    @staticmethod
    def load(
        db_dir: str,
        text_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        image_model_name="clip-ViT-B-32",
    ):
        """
        Load a saved MultiModalRetriever database and return a new instance.
        """
        
        cfg_path = os.path.join(db_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("text_model_name") != text_model_name:
                print("[WARN] text_model_name does not match saved DB")
            if cfg.get("image_model_name") != image_model_name:
                print("[WARN] image_model_name does not match saved DB")

        # 1️⃣ 建立新物件（model 會在 __init__ 初始化）
        r = MultiModalRetriever(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
        )

        # 2️⃣ Load FAISS indices
        r.title_index = faiss.read_index(os.path.join(db_dir, "title.faiss"))
        r.sur_index   = faiss.read_index(os.path.join(db_dir, "sur.faiss"))
        r.img_index   = faiss.read_index(os.path.join(db_dir, "img.faiss"))

        # 3️⃣ Load vectors / dims
        data = np.load(os.path.join(db_dir, "vectors.npz"), allow_pickle=True)

        r.v_title = data["v_title"]
        r.v_img = data["v_img"]
        r.v_sur_chunks = list(data["v_sur_chunks"])
        r.sur_chunks_text = list(data["sur_chunks_text"])

        r.text_dim = int(data["text_dim"])
        r.image_dim = int(data["image_dim"])

        # 4️⃣ Load meta
        with open(os.path.join(db_dir, "meta.json"), "r", encoding="utf-8") as f:
            r.meta = json.load(f)

        return r



    def save(self, db_dir: str):
        os.makedirs(db_dir, exist_ok=True)

        # --- FAISS indices ---
        faiss.write_index(self.title_index, os.path.join(db_dir, "title.faiss"))
        faiss.write_index(self.sur_index,   os.path.join(db_dir, "sur.faiss"))
        faiss.write_index(self.img_index,   os.path.join(db_dir, "img.faiss"))

        # --- vectors ---
        np.savez_compressed(
            os.path.join(db_dir, "vectors.npz"),
            v_title=self.v_title,
            v_img=self.v_img,
            v_sur_chunks=np.array(self.v_sur_chunks, dtype=object),
            sur_chunks_text=np.array(self.sur_chunks_text, dtype=object),
            text_dim=self.text_dim,
            image_dim=self.image_dim,
        )

        # --- meta ---
        with open(os.path.join(db_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(db_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "text_model_name": self.text_model_name or "",
                "image_model_name": self.image_model_name or "",
            }, f, indent=2)


if __name__ == "__main__":
    # ===== 用法 =====
    # r = MultiModalRetriever()
    # r.add_folder("output_stored/L1Vin1RByA/image_datas", n_sur=3)
    # r.add_folder("output_stored/43Uk9N1gnY/image_datas", n_sur=3)
    # r = MultiModalRetriever.load("db")
    
    # Check if DB exists, if not, create a placeholder or error
    DB_PATH = "agriculture_db"
    if os.path.exists(DB_PATH):
        r = MultiModalRetriever.load(DB_PATH)
        # 許多螞蟻聚集在水管裡的圖片
        input_text = input("輸入搜索關鍵字: ")
        
        hits = r.search(
            input_text,
            topk=3,
            alpha=0.7,        # text 70%, image 30%
            beta_title=0.7,   # text 裡面：title 70%
            beta_sur=0.3
        )
        
        # print("搜索結果如下(列出信心值前三的相關圖片): ")
        for i, hit in enumerate(hits[:3]):
            print(f"{i+1}. 圖片位置: {hit['image_path']} | 對應文件名稱: {hit['doc_name']} | 信心值: {hit['score']:.2f}")
        
        open("o.json", "w", encoding="utf-8").write(json.dumps(hits, ensure_ascii=False, indent=2))
        print("詳細尋結果已儲存到 o.json 中。")
    else:
        print("資料庫 'db' 不存在，請先運行索引腳本。")


