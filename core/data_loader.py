"""Data loader module for ingesting image metadata and multi-surrounding texts from image_data/."""

import glob
import json
import logging
import os
from typing import List, Optional, Dict, Any
from core.models import ImageDocument

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads, validates, and normalizes image documents with multiple surrounding texts."""

    def __init__(self, data_dir: str = "image_data"):
        self.data_dir = data_dir

    def scan_metadata_files(self) -> List[str]:
        """Scans data directory for all metadata.json files."""
        pattern1 = os.path.join(self.data_dir, "*", "image_datas", "metadata.json")
        files = glob.glob(pattern1)
        if not files:
            pattern2 = os.path.join(self.data_dir, "**", "metadata.json")
            files = glob.glob(pattern2, recursive=True)
        files.sort()
        return files

    def load_documents(
        self,
        max_documents: Optional[int] = None,
        max_images: Optional[int] = None,
        require_image_exists: bool = False
    ) -> List[ImageDocument]:
        """Parses metadata.json files into ImageDocument records with aggregated surrounding texts."""
        metadata_files = self.scan_metadata_files()
        if max_documents is not None:
            metadata_files = metadata_files[:max_documents]

        documents: List[ImageDocument] = []
        total_files = len(metadata_files)
        logger.info(f"Loading image records from {total_files} metadata files...")

        for f_idx, file_path in enumerate(metadata_files):
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

            doc_name = (data.get("name") or "").strip()
            uid = (data.get("uid") or os.path.basename(os.path.dirname(os.path.dirname(file_path)))).strip()
            imgs = data.get("imgs", [])

            for img_dict in imgs:
                img_name = str(img_dict.get("name", "")).strip()
                if not img_name:
                    continue

                # Support both .png and .jpg image paths
                folder_dir = os.path.dirname(file_path)
                rel_img_path = os.path.join(folder_dir, f"{img_name}.png")
                if not os.path.exists(rel_img_path):
                    alt_path = os.path.join(folder_dir, f"{img_name}.jpg")
                    if os.path.exists(alt_path):
                        rel_img_path = alt_path

                if require_image_exists and not os.path.exists(rel_img_path):
                    continue

                # Process surrounding_texts (ensuring all list items are captured)
                raw_surroundings = img_dict.get("surrounding_texts", [])
                surrounding_texts: List[str] = []
                if isinstance(raw_surroundings, list):
                    for st in raw_surroundings:
                        if isinstance(st, str) and st.strip():
                            surrounding_texts.append(st.strip())
                elif isinstance(raw_surroundings, str) and raw_surroundings.strip():
                    surrounding_texts.append(raw_surroundings.strip())

                image_id = f"{uid}_{img_name}"
                doc = ImageDocument(
                    image_id=image_id,
                    uid=uid,
                    doc_name=doc_name,
                    image_name=img_name,
                    file_path=rel_img_path,
                    page=int(img_dict.get("page", 1)),
                    coordinate=img_dict.get("coordinate"),
                    figure_title=str(img_dict.get("figure_title", "")).strip(),
                    surrounding_texts=surrounding_texts
                )
                doc.build_text_representations()
                documents.append(doc)

                if max_images is not None and len(documents) >= max_images:
                    break

            if max_images is not None and len(documents) >= max_images:
                break

        logger.info(f"Successfully loaded {len(documents)} ImageDocument records from {total_files} files.")
        return documents

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Computes summary statistics on dataset files without loading everything into memory."""
        files = self.scan_metadata_files()
        total_images = 0
        total_surroundings = 0
        
        for f in files[:300]:  # Sample count
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                    imgs = d.get("imgs", [])
                    total_images += len(imgs)
                    for img in imgs:
                        sts = img.get("surrounding_texts", [])
                        total_surroundings += len(sts) if isinstance(sts, list) else (1 if sts else 0)
            except Exception:
                pass

        avg_imgs_per_doc = (total_images / max(1, min(len(files), 300)))
        avg_surroundings = (total_surroundings / max(1, total_images))
        est_total_images = int(len(files) * avg_imgs_per_doc)

        return {
            "total_metadata_files": len(files),
            "sampled_files": min(len(files), 300),
            "avg_images_per_document": round(avg_imgs_per_doc, 2),
            "avg_surrounding_texts_per_image": round(avg_surroundings, 2),
            "estimated_total_images": est_total_images
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    stats = loader.get_dataset_stats()
    print("Dataset Statistics:", stats)
    
    # Test loading small sample
    sample_docs = loader.load_documents(max_documents=5)
    print(f"Loaded sample {len(sample_docs)} docs:")
    for doc in sample_docs[:2]:
        print("-" * 40)
        print(f"ID: {doc.image_id}")
        print(f"Doc Name: {doc.doc_name}")
        print(f"Figure Title: {doc.figure_title}")
        print(f"Surrounding Texts Count: {len(doc.surrounding_texts)}")
        print(f"Caption Text: {doc.caption_text}")
        print(f"Full Text: {doc.full_text[:120]}...")
