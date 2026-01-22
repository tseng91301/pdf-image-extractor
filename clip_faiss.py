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

def chunk_text(text, chunk_size=120, overlap=30):
    if not text:
        return []

    text = text.strip()
    chunks = []
    start = 0
    L = len(text)

    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if start < 0:
            break

    return chunks


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

        # 3 個 index：title/caption, surrounding, image
        self.title_index = None
        self.sur_index = None
        self.img_index = None

        self.meta = []
        self.v_title = None
        self.v_sur = None
        self.v_img = None
        self.dim = None

    def _ensure_index(self, dim: int):
        if self.title_index is None:
            self.title_index = faiss.IndexFlatIP(dim)
            self.sur_index   = faiss.IndexFlatIP(dim)
            self.img_index   = faiss.IndexFlatIP(dim)
            self.dim = dim

    def add_document(self, json_path: str, images_dir: str, n_sur=3, doc_name_override=None):
        data = json.load(open(json_path, "r", encoding="utf-8"))
        imgs = data["imgs"]

        titles, sur_texts, image_paths, new_meta = [], [], [], []

        for it in imgs:
            img_name = it["name"]
            png_path = os.path.join(images_dir, f"{img_name}.png")
            if not os.path.exists(png_path):
                continue

            fig_title = normalize_text(it.get("figure_title", ""))

            # surrounding texts：取前 n_sur 段，各段 normalize 後用換行串起來
            sur_list = (it.get("surrounding_texts", []) or [])[:n_sur]
            sur_text = "\n".join(normalize_text(x) for x in sur_list).strip()

            titles.append(fig_title)
            sur_texts.append(sur_text)
            image_paths.append(png_path)

            new_meta.append({
                "doc_name": doc_name_override or data.get("name"),
                "uid": data.get("uid"),
                "page": it.get("page"),
                "image_name": img_name,
                "image_path": png_path,
                "coordinate": it.get("coordinate"),
                "figure_title": it.get("figure_title", ""),
                "surrounding_texts": it.get("surrounding_texts", []),
                "title_text_used": fig_title,
                "sur_text_used": sur_text,
            })

        if not image_paths:
            return 0

        # --- embeddings ---
        v_title = self.model.encode(
            titles, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        v_sur = self.model.encode(
            sur_texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        pil_imgs = [Image.open(p).convert("RGB") for p in image_paths]
        v_img = self.model.encode(
            pil_imgs, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        self._ensure_index(v_title.shape[1])

        # --- add to faiss ---
        self.title_index.add(v_title)
        self.sur_index.add(v_sur)
        self.img_index.add(v_img)

        # --- keep vectors for rerank dot ---
        if self.v_title is None:
            self.v_title = v_title
            self.v_sur = v_sur
            self.v_img = v_img
        else:
            self.v_title = np.vstack([self.v_title, v_title])
            self.v_sur   = np.vstack([self.v_sur, v_sur])
            self.v_img   = np.vstack([self.v_img, v_img])

        self.meta.extend(new_meta)
        return len(new_meta)

    def build(self, json_path: str, images_dir: str, n_sur=3):
        self.title_index = None
        self.sur_index = None
        self.img_index = None
        self.meta = []
        self.v_title = None
        self.v_sur = None
        self.v_img = None
        self.dim = None
        return self.add_document(json_path, images_dir, n_sur=n_sur)

    def search(
        self, query: str, topk=10, k_each=50,
        alpha=0.6,          # text vs image
        beta_title=0.7,     # title(caption) vs surrounding (text 內部)
        beta_sur=0.3
    ):
        if self.title_index is None or self.sur_index is None or self.img_index is None:
            raise RuntimeError("Index not built. Call build() or add_document() first.")

        # 確保 text 內部權重總和為 1（避免你手滑）
        s = beta_title + beta_sur
        if s <= 0:
            raise ValueError("beta_title + beta_sur must be > 0")
        beta_title /= s
        beta_sur   /= s

        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        # 各取候選（title / surrounding / image 都取一批）
        Dt, It = self.title_index.search(q, k_each)
        Ds, Is = self.sur_index.search(q, k_each)
        Di, Ii = self.img_index.search(q, k_each)

        cand = set(It[0].tolist()) | set(Is[0].tolist()) | set(Ii[0].tolist())

        results = []
        for idx in cand:
            s_title = float(np.dot(q[0], self.v_title[idx]))
            s_sur   = float(np.dot(q[0], self.v_sur[idx]))
            s_img   = float(np.dot(q[0], self.v_img[idx]))

            s_text = beta_title * s_title + beta_sur * s_sur
            score  = alpha * s_text + (1 - alpha) * s_img

            m = self.meta[idx]
            results.append({
                "score": score,
                "s_text": s_text,
                "s_title": s_title,
                "s_sur": s_sur,
                "s_img": s_img,
                "doc_name": m["doc_name"],
                "uid": m["uid"],
                "page": m["page"],
                "image_name": m["image_name"],
                "image_path": m["image_path"],
                "figure_title": m["figure_title"],
                "title_text_used": m["title_text_used"],
                "sur_text_used": m["sur_text_used"],
                "coordinate": m["coordinate"],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:topk]


# ===== 用法 =====
r = MultiModalRetriever(model_name="clip-ViT-B-32")
r.add_document("output_stored/L1Vin1RByA/image_datas/metadata.json", "output_stored/L1Vin1RByA/image_datas", n_sur=3)
r.add_document("output_stored/43Uk9N1gnY/image_datas/metadata.json", "output_stored/43Uk9N1gnY/image_datas", n_sur=3)

hits = r.search(
    "管內有很多隻螞蟻",
    topk=10,
    alpha=0.6,        # text 60%, image 40%
    beta_title=0.8,   # text 裡面：title 80%
    beta_sur=0.2
)

open("o.json", "w", encoding="utf-8").write(json.dumps(hits, ensure_ascii=False, indent=2))

# for h in hits:
#     print(h["score"], h["page"], h["image_path"], h["figure_title"], h["s_title"], h["s_sur"], h["s_img"])

