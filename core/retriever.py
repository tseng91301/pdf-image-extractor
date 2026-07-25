import os
import json
import re
import math
import torch
import numpy as np
from PIL import Image
import faiss
import jieba
from sentence_transformers import SentenceTransformer
from .nlp_utils import extract_keywords
from .translator import OfflineTranslator


def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


class MultiModalRetriever:
    text_model_name: str
    image_model_name: str

    def __init__(
        self,
        text_model_name="BAAI/bge-small-zh-v1.5",
        image_model_name="clip-ViT-B-32",
        use_translation=True,
        device=None,
    ):
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                print("\033[93m[WARNING] [Retriever] CUDA is not available. Falling back to CPU.\033[0m")
        else:
            self.device = device

        # Load models
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_model = SentenceTransformer(text_model_name, device=self.device)
        self.image_model = SentenceTransformer(image_model_name, device=self.device)

        # Translator
        self.translator = OfflineTranslator(device=self.device) if use_translation else None

        # Image index (FAISS)
        self.img_index = None
        self.v_img = None

        # Dimensions
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        self.image_dim = None

        # Unique keyword embeddings cache
        self.keyword_embeddings = {}
        self.meta = []

        # Adjustable surround text keyword weight ranges (TF scaling)
        self.w_surround_min = 0.1
        self.w_surround_max = 0.6

    def _ensure_index(self, image_dim: int):
        if self.img_index is None:
            self.img_index = faiss.IndexFlatIP(image_dim)
            self.image_dim = image_dim

    def add_document(
        self,
        json_path: str,
        images_dir: str,
        n_sur=3,
        doc_name_override=None,
        **kwargs,
    ):
        data = json.load(open(json_path, "r", encoding="utf-8"))
        imgs = data["imgs"]

        image_paths = []
        new_meta = []

        for it in imgs:
            img_name = it["name"]
            png_path = os.path.join(images_dir, f"{img_name}.png")
            if not os.path.exists(png_path):
                continue

            fig_title = normalize_text(it.get("figure_title", ""))
            sur_list = (it.get("surrounding_texts", []) or [])[:n_sur]
            sur_list = [normalize_text(x) for x in sur_list if x.strip()]

            # 1. Extract keywords from Caption (weight = 1.0)
            caption_kws = extract_keywords(fig_title, model=self.text_model, device=self.device)

            # 2. Extract keywords from surrounding text blocks (weight = 0.5)
            text_kws = []
            for sur in sur_list:
                text_kws.extend(extract_keywords(sur, model=self.text_model, device=self.device))
            text_kws = list(set(text_kws))

            # 3. Merge weights: TF-based weights for surrounding text keywords
            combined_sur_text = " ".join(sur_list)
            counts = {}
            for kw in text_kws:
                counts[kw] = combined_sur_text.count(kw)

            min_c = min(counts.values()) if counts else 0
            max_c = max(counts.values()) if counts else 0

            keyword_weights = {}
            for kw, c in counts.items():
                if max_c > min_c:
                    # Min-Max normalize term frequency to [w_surround_min, w_surround_max]
                    w = self.w_surround_min + (c - min_c) / (max_c - min_c) * (self.w_surround_max - self.w_surround_min)
                else:
                    w = self.w_surround_max
                keyword_weights[kw] = float(w)

            # Caption keywords overwrite with 1.0
            for kw in caption_kws:
                keyword_weights[kw] = 1.0

            keywords = list(keyword_weights.keys())
            image_paths.append(png_path)

            meta_item = {
                "doc_name": doc_name_override or data.get("name"),
                "uid": data.get("uid"),
                "page": it.get("page"),
                "image_name": img_name,
                "image_path": png_path,
                "coordinate": it.get("coordinate"),
                "figure_title": fig_title,
                "sur_text_list": sur_list,
                "keywords": keywords,
                "keyword_weights": keyword_weights,
            }
            new_meta.append(meta_item)

        if not image_paths:
            return 0

        # ---- Batch encode new keywords ----
        new_keywords = set()
        for item in new_meta:
            for kw in item["keywords"]:
                if kw not in self.keyword_embeddings:
                    new_keywords.add(kw)
        
        if new_keywords:
            new_keywords_list = list(new_keywords)
            kw_embs = self.text_model.encode(
                new_keywords_list,
                batch_size=128,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            for kw, emb in zip(new_keywords_list, kw_embs):
                self.keyword_embeddings[kw] = emb

        # ---- Image embeddings ----
        pil_imgs = [Image.open(p).convert("RGB") for p in image_paths]
        v_img = self.image_model.encode(
            pil_imgs,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        # ---- Ensure index and add ----
        self._ensure_index(v_img.shape[1])
        self.img_index.add(v_img)
        self.v_img = v_img if self.v_img is None else np.vstack([self.v_img, v_img])

        self.meta.extend(new_meta)
        return len(new_meta)

    def add_folder(self, folder_path: str, n_sur=3, doc_name_override=None, **kwargs):
        return self.add_document(
            os.path.join(folder_path, "metadata.json"),
            folder_path,
            n_sur=n_sur,
            doc_name_override=doc_name_override,
        )

    def extract_query_keywords(self, query: str) -> list:
        query_keywords = extract_keywords(query, model=self.text_model, device=self.device)
        if len(query_keywords) == 0:
            query_keywords = jieba.lcut(query)
        return [kw for kw in query_keywords if kw.strip()]

    def search_path_a(self, query_keywords: list, k_each=100) -> list:
        if not query_keywords or not self.keyword_embeddings:
            return []
        db_keywords = list(self.keyword_embeddings.keys())
        q_text_embs = self.text_model.encode(
            query_keywords,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        db_text_embs = np.array([self.keyword_embeddings[kw] for kw in db_keywords], dtype="float32")
        
        sim_matrix = q_text_embs @ db_text_embs.T  # [len(query_keywords), len(db_keywords)]
        
        matched_db_kws = {}
        for i, q_kw in enumerate(query_keywords):
            matched_db_kws[q_kw] = []
            for j, db_kw in enumerate(db_keywords):
                sim = float(sim_matrix[i, j])
                if sim >= 0.70:
                    matched_db_kws[q_kw].append((db_kw, sim))

        path_a_raw_scores = []
        for idx, img_meta in enumerate(self.meta):
            score_caption = 0.0
            score_surround = 0.0
            matched_pairs = []
            for q_kw in query_keywords:
                for db_kw, sim in matched_db_kws[q_kw]:
                    if db_kw in img_meta.get("keyword_weights", {}):
                        w = img_meta["keyword_weights"][db_kw]
                        term_score = sim * w
                        if w == 1.0:
                            score_caption += term_score
                        else:
                            score_surround += term_score
                        matched_pairs.append((q_kw, db_kw, sim, w, term_score))
            
            # Apply Method 2: Square root scaling for surrounding text score
            score_A = score_caption + math.sqrt(score_surround)
            if score_A > 0.0:
                path_a_raw_scores.append((idx, score_A, matched_pairs))
        
        path_a_raw_scores.sort(key=lambda x: x[1], reverse=True)
        return path_a_raw_scores[:k_each]

    def search_path_b(self, query_keywords: list, k_each=100) -> list:
        if not query_keywords or self.v_img is None:
            return []
        translated_query_keywords = []
        for kw in query_keywords:
            if self.translator:
                translated = self.translator.translate(kw)
                translated_query_keywords.append(translated)
            else:
                translated_query_keywords.append(kw)

        q_img_embs = self.image_model.encode(
            translated_query_keywords,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        sim_matrix_B = q_img_embs @ self.v_img.T  # [len(query_keywords), num_images]

        path_b_raw_scores = []
        for idx in range(len(self.meta)):
            score_B = 0.0
            for i in range(len(query_keywords)):
                sim = float(sim_matrix_B[i, idx])
                if sim >= 0.70:
                    score_B += sim
            if score_B > 0.0:
                path_b_raw_scores.append((idx, score_B))

        path_b_raw_scores.sort(key=lambda x: x[1], reverse=True)
        return path_b_raw_scores[:k_each]

    def search(
        self,
        query: str,
        topk=10,
        k_each=100,
        w_text=0.7,
        w_image=0.3,
        **kwargs,
    ):
        if self.v_img is None:
            raise RuntimeError("Index not built")

        query_keywords = self.extract_query_keywords(query)
        if not query_keywords:
            return {"query_keywords": [], "results": []}

        # 🛑 Path A: Text to Text Search
        top_k1_scores = self.search_path_a(query_keywords, k_each=k_each)

        # 🛑 Path B: Text to Image Search
        top_k2_scores = self.search_path_b(query_keywords, k_each=k_each)

        # 🛑 Merge & Output
        union_indices = set(idx for idx, _, _ in top_k1_scores) | set(idx for idx, _ in top_k2_scores)
        
        # Min-Max Normalization Path A
        scores_A_map = {idx: s for idx, s, _ in top_k1_scores}
        matched_pairs_map = {idx: mp for idx, _, mp in top_k1_scores}
        raw_A_values = list(scores_A_map.values())
        min_A = min(raw_A_values) if raw_A_values else 0.0
        max_A = max(raw_A_values) if raw_A_values else 0.0

        # Min-Max Normalization Path B
        scores_B_map = {idx: s for idx, s in top_k2_scores}
        raw_B_values = list(scores_B_map.values())
        min_B = min(raw_B_values) if raw_B_values else 0.0
        max_B = max(raw_B_values) if raw_B_values else 0.0

        fused_results = []
        for idx in union_indices:
            raw_score_A = scores_A_map.get(idx, 0.0)
            if max_A > min_A:
                norm_score_A = (raw_score_A - min_A) / (max_A - min_A)
            else:
                norm_score_A = 1.0 if raw_score_A > 0.0 else 0.0

            raw_score_B = scores_B_map.get(idx, 0.0)
            if max_B > min_B:
                norm_score_B = (raw_score_B - min_B) / (max_B - min_B)
            else:
                norm_score_B = 1.0 if raw_score_B > 0.0 else 0.0

            # Late fusion combination
            denom = w_text + w_image
            if denom > 0.0:
                alpha_text = w_text / denom
                alpha_image = w_image / denom
            else:
                alpha_text, alpha_image = 0.5, 0.5

            final_score = alpha_text * norm_score_A + alpha_image * norm_score_B

            # Extract matched query keywords
            matched_pairs = matched_pairs_map.get(idx, [])
            matched_kws = list(set(p[0] for p in matched_pairs))
            matched_details = [
                {"q_kw": p[0], "db_kw": p[1], "sim": p[2], "weight": p[3], "score": p[4]}
                for p in matched_pairs
            ]

            meta_item = self.meta[idx]
            fused_results.append({
                "score": final_score,
                "s_text": norm_score_A,
                "s_img": norm_score_B,
                # compatibility properties for frontend
                "s_title": norm_score_A,
                "s_sur": 0.0,
                "s_keyword": norm_score_A,
                "best_sur_chunk": None,
                "matched_keywords": matched_kws,
                "matched_details": matched_details,
                **meta_item,
            })

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query_keywords": query_keywords,
            "results": fused_results[:topk],
        }

    def search_by_image(self, image: Image.Image, topk=10, k_each=100):
        if self.img_index is None:
            raise RuntimeError("Image index not built")

        # Encode input image
        q_img = self.image_model.encode(
            [image],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Recall from image index
        _, Ii = self.img_index.search(q_img, k_each)

        cand = set(Ii[0])
        results = []
        for idx in cand:
            s_img = float(np.dot(q_img[0], self.v_img[idx]))
            m = self.meta[idx]

            results.append({
                "score": s_img,
                "s_img": s_img,
                "s_text": 0.0,
                "s_title": 0.0,
                "s_sur": 0.0,
                "s_keyword": 0.0,
                "best_sur_chunk": None,
                "matched_keywords": [],
                **m,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query_keywords": [],
            "results": results[:topk],
        }

    @staticmethod
    def load(
        db_dir: str,
        text_model_name="BAAI/bge-small-zh-v1.5",
        image_model_name="clip-ViT-B-32",
        use_translation=True,
        device=None,
    ):
        r = MultiModalRetriever(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            use_translation=use_translation,
            device=device,
        )
        r.img_index = faiss.read_index(os.path.join(db_dir, "img.faiss"))

        data = np.load(os.path.join(db_dir, "vectors.npz"), allow_pickle=True)
        r.v_img = data["v_img"]
        r.image_dim = int(data["image_dim"])
        r.text_dim = int(data["text_dim"]) if "text_dim" in data else r.text_model.get_sentence_embedding_dimension()

        r.keyword_embeddings = {}
        if "kw_keys" in data and "kw_values" in data:
            keys = data["kw_keys"]
            values = data["kw_values"]
            for k, v in zip(keys, values):
                r.keyword_embeddings[str(k)] = v

        with open(os.path.join(db_dir, "meta.json"), "r", encoding="utf-8") as f:
            r.meta = json.load(f)

        # Load config and override weight parameters
        config_path = os.path.join(db_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    r.w_surround_min = cfg.get("w_surround_min", 0.1)
                    r.w_surround_max = cfg.get("w_surround_max", 0.6)
            except Exception:
                pass

        return r

    def save(self, db_dir: str):
        os.makedirs(db_dir, exist_ok=True)
        faiss.write_index(self.img_index, os.path.join(db_dir, "img.faiss"))

        np.savez_compressed(
            os.path.join(db_dir, "vectors.npz"),
            v_img=self.v_img,
            image_dim=self.image_dim,
            text_dim=self.text_dim,
            kw_keys=np.array(list(self.keyword_embeddings.keys())),
            kw_values=np.array(list(self.keyword_embeddings.values()), dtype="float32") if self.keyword_embeddings else np.zeros((0, self.text_dim), dtype="float32"),
        )

        with open(os.path.join(db_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

        with open(os.path.join(db_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "text_model_name": self.text_model_name or "",
                "image_model_name": self.image_model_name or "",
                "w_surround_min": self.w_surround_min,
                "w_surround_max": self.w_surround_max,
            }, f, indent=2)
