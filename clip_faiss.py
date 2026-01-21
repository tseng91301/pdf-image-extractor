import os, json, re
import numpy as np
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def build_caption_text(fig_title: str, surrounding_texts: list[str], n_sur=3) -> str:
    fig_title = normalize_text(fig_title)
    sur = "\n".join(normalize_text(x) for x in (surrounding_texts or [])[:n_sur])
    if fig_title and sur:
        return fig_title + "\n" + sur
    return fig_title or sur

def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

class MultiModalRetriever:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.cap_index = None
        self.img_index = None
        self.meta = []  # 每筆對應同一張圖的 metadata
        self.v_cap = None
        self.v_img = None
        self.dim = None
        
    def _ensure_index(self, dim: int):
        if self.cap_index is None:
            self.cap_index = faiss.IndexFlatIP(dim)
            self.img_index = faiss.IndexFlatIP(dim)
            self.dim = dim
    
    def add_document(self, json_path: str, images_dir: str, n_sur=3, doc_name_override=None):
        data = json.load(open(json_path, "r", encoding="utf-8"))
        imgs = data["imgs"]

        captions, image_paths, new_meta = [], [], []

        for it in imgs:
            img_name = it["name"]
            png_path = os.path.join(images_dir, f"{img_name}.png")
            if not os.path.exists(png_path):
                continue

            cap_text = build_caption_text(it.get("figure_title",""), it.get("surrounding_texts", []), n_sur=n_sur)

            captions.append(cap_text)
            image_paths.append(png_path)

            new_meta.append({
                "doc_name": doc_name_override or data.get("name"),
                "uid": data.get("uid"),
                "page": it.get("page"),
                "image_name": img_name,
                "image_path": png_path,
                "coordinate": it.get("coordinate"),
                "figure_title": it.get("figure_title",""),
                "surrounding_texts": it.get("surrounding_texts", []),
                "caption_text_used": cap_text,
            })

        if not captions:
            return 0

        v_cap = self.model.encode(captions, batch_size=64, show_progress_bar=True,
                                  convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        pil_imgs = [Image.open(p).convert("RGB") for p in image_paths]
        v_img = self.model.encode(pil_imgs, batch_size=32, show_progress_bar=True,
                                  convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        self._ensure_index(v_cap.shape[1])

        self.cap_index.add(v_cap)
        self.img_index.add(v_img)

        # 保留 embeddings（若你需要在 search 時自己 dot）
        if self.v_cap is None:
            self.v_cap = v_cap
            self.v_img = v_img
        else:
            self.v_cap = np.vstack([self.v_cap, v_cap])
            self.v_img = np.vstack([self.v_img, v_img])

        self.meta.extend(new_meta)
        return len(new_meta)

    def build(self, json_path: str, images_dir: str, n_sur=3):
        # 兼容舊用法：清空後重建一次
        self.cap_index = None
        self.img_index = None
        self.meta = []
        self.v_cap = None
        self.v_img = None
        self.dim = None
        return self.add_document(json_path, images_dir, n_sur=n_sur)

    def search(self, query: str, topk=10, k_each=50, alpha=0.6):
        if self.cap_index is None or self.img_index is None:
            raise RuntimeError("Index not built. Call build() first.")

        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        # 各取候選
        Dcap, Icap = self.cap_index.search(q, k_each)
        Dimg, Iimg = self.img_index.search(q, k_each)

        # 合併候選集合
        cand = set(Icap[0].tolist()) | set(Iimg[0].tolist())

        results = []
        for idx in cand:
            s_cap = float(np.dot(q[0], self.v_cap[idx]))
            s_img = float(np.dot(q[0], self.v_img[idx]))
            score = alpha * s_cap + (1 - alpha) * s_img
            m = self.meta[idx]
            results.append({
                "score": score,
                "s_cap": s_cap,
                "s_img": s_img,
                "page": m["page"],
                "image_name": m["image_name"],
                "image_path": m["image_path"],
                "figure_title": m["figure_title"],
                "caption_text_used": m["caption_text_used"],
                "coordinate": m["coordinate"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:topk]

# ===== 用法 =====
r = MultiModalRetriever(model_name="clip-ViT-B-32")
r.add_document("output_stored/4WQ9S9yK3M/image_datas/metadata.json", images_dir="output/4WQ9S9yK3M/image_datas", n_sur=3)
hits = r.search("200隻左右 螞蟻", topk=3, alpha=0.6)
for h in hits:
    print(h["score"], h["page"], h["image_name"], h["figure_title"])
