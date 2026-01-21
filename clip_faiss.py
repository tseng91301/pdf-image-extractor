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

    def build(self, json_path: str, images_dir: str, n_sur=3):
        data = json.load(open(json_path, "r", encoding="utf-8"))
        imgs = data["imgs"]

        captions = []
        image_paths = []

        self.meta = []
        for it in imgs:
            img_name = it["name"]
            png_path = os.path.join(images_dir, f"{img_name}.png")
            # 若有些圖可能不是 png，自行在這裡加 fallback（.jpg/.jpeg）
            if not os.path.exists(png_path):
                continue

            cap_text = build_caption_text(it.get("figure_title",""), it.get("surrounding_texts", []), n_sur=n_sur)

            captions.append(cap_text)
            image_paths.append(png_path)

            self.meta.append({
                "doc_name": data.get("name"),
                "uid": data.get("uid"),
                "image_name": img_name,
                "image_path": png_path,
                "page": it.get("page"),
                "coordinate": it.get("coordinate"),
                "figure_title": it.get("figure_title",""),
                "surrounding_texts": it.get("surrounding_texts", []),
                "caption_text_used": cap_text,
            })

        # 1) caption embeddings
        v_cap = self.model.encode(captions, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        # 2) image embeddings
        pil_imgs = [Image.open(p).convert("RGB") for p in image_paths]
        v_img = self.model.encode(pil_imgs, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        self.v_cap = v_cap.astype("float32")
        self.v_img = v_img.astype("float32")

        dim = self.v_cap.shape[1]

        # 用 cosine 相似度：等價於對 L2 normalized 向量做 inner product
        self.cap_index = faiss.IndexFlatIP(dim)
        self.img_index = faiss.IndexFlatIP(dim)
        self.cap_index.add(self.v_cap)
        self.img_index.add(self.v_img)

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
# r = MultiModalRetriever(model_name="clip-ViT-B-32")
# r.build("normal4.json", images_dir=".", n_sur=3)
# hits = r.search("雙疣琉璃蟻 果園 硼酸 餌劑", topk=10, alpha=0.6)
# for h in hits:
#     print(h["score"], h["page"], h["image_name"], h["figure_title"])
