# 📄 PDF Image Caption Inference System

## 1. Project Overview

This project aims to build a **robust PDF image extraction and caption inference system** designed primarily for **scanned or image-based PDFs**, where text layers are unreliable or completely absent.

Instead of relying on fragile PDF text extraction, the system treats every PDF as a **visual document**, extracts figures based on layout understanding, and infers image captions using **layout structure, nearby context, and semantic reasoning**.

The final goal is to automatically generate:

* Cropped figure images
* Associated captions (exact, nearby, or inferred)
* Structured metadata suitable for downstream applications (search, QA, datasets, RAG, etc.)

---

## 2. Design Philosophy

### Why Scanned-PDF-First?

* Real-world PDFs are often inconsistent:

  * Mixed text/image pages
  * Broken text layers
  * Embedded rasterized pages
* Treating all PDFs as image-based:

  * Eliminates format branching
  * Unifies the processing pipeline
  * Improves robustness and reproducibility

### Core Principle

> **Do not try to “read text first”;
> understand document structure first, then recover semantics.**

---

## 3. System Architecture (High-Level)

```
PDF
 └── Page Rendering (Image)
      ├── Layout Detection
      │    ├── Figure Blocks
      │    ├── Text Blocks
      │    ├── Caption Blocks
      │    └── Table Blocks
      │
      ├── Block-Level OCR (Selective)
      │
      ├── Figure–Context Association
      │
      └── Caption Inference (LLM-assisted)
```

---

## 4. Processing Pipeline (Step-by-Step)

### Step 1. PDF Page Rendering

* Convert each PDF page into a high-resolution image (200–300 DPI)
* Ignore text layer completely

**Output**

* `page_001.png`, `page_002.png`, ...

---

### Step 2. Document Layout Analysis

Detect and classify layout elements:

* Title
* Text
* Figure
* Caption
* Table

Each block includes:

* Bounding box
* Block type
* Page index

**Output**

```json
{
  "page": 2,
  "blocks": [
    {"type": "figure", "bbox": [...]},
    {"type": "text", "bbox": [...]},
    {"type": "caption", "bbox": [...]}
  ]
}
```

---

### Step 3. Figure Extraction

* Crop figure regions from page images
* Assign unique figure IDs

**Output**

* `fig_page2_01.png`
* `fig_page3_02.png`

---

### Step 4. Caption Candidate Collection

For each figure:

* Collect nearby text blocks based on:

  * Distance
  * Alignment
  * Same column
  * Reading order
* Include:

  * Below / above text
  * Side text (for multi-column layouts)

**Goal**

> Reduce caption search space, not decide yet.

---

### Step 5. Selective OCR

Apply OCR **only** to:

* Caption candidate blocks
* Optional: figure-referencing paragraphs (e.g. “如圖3所示”)

**Why selective OCR?**

* Faster
* Fewer recognition errors
* Easier post-processing

---

### Step 6. Caption Resolution & Inference

Use rules + LLM to classify caption source:

1. **Exact Caption**

   * Explicit “Figure X / 圖 X” patterns
2. **Nearby Caption**

   * Descriptive text close to figure
3. **Inferred Caption**

   * Generated from surrounding context

Each output caption is tagged with its confidence level.

---

## 5. Output Format (Structured & Traceable)

```json
{
  "figure_id": "page2_fig1",
  "page": 2,
  "image_path": "fig_page2_01.png",
  "caption_type": "inferred",
  "caption_text": "Comparison of weed growth between traditional manual weeding and non-woven fabric covering.",
  "evidence": {
    "nearby_text_blocks": [...],
    "layout_relation": "below_figure"
  }
}
```

---

## 6. Key Technologies (Planned)

* PDF Rendering: `pdfium`, `poppler`, or `PyMuPDF`
* Layout Detection: `LayoutParser`, Detectron2 (DocLayNet / PubLayNet)
* OCR: `PaddleOCR` (Traditional Chinese support)
* Caption Reasoning: LLM (prompted with structured evidence)
* Data Storage: JSON / SQLite / Parquet

---

## 7. TODO List (Roadmap)

### Phase 1 – MVP (Core Functionality)

* [ ] PDF to page image rendering
* [ ] Layout detection (figure + text only)
* [ ] Figure cropping and ID assignment
* [ ] Distance-based caption candidate selection
* [ ] Block-level OCR for caption candidates
* [ ] Export per-figure JSON results

### Phase 2 – Layout & Context Refinement

* [ ] Multi-column reading order inference
* [ ] Caption pattern normalization (Fig / 圖 / OCR errors)
* [ ] Cross-page caption matching
* [ ] Table vs figure disambiguation
* [ ] Caption confidence scoring

### Phase 3 – Semantic Inference

* [ ] LLM-based caption reranking
* [ ] Inferred caption generation (context-based)
* [ ] Hallucination control via evidence constraints
* [ ] Distinguish extracted vs inferred captions

### Phase 4 – Usability & Scaling

* [ ] Batch PDF processing
* [ ] CLI / API interface
* [ ] Visualization tool for debugging layout & associations
* [ ] Dataset export for downstream ML / RAG systems

---

## 8. Long-Term Extensions (Optional)

* Document-level summary with figure references
* Multilingual caption normalization
* Scientific paper / technical report specialization
* Human-in-the-loop correction interface

---

## 9. Final Note

This project is **not a PDF parser**.
It is a **document understanding system** focused on **figures as first-class objects**.

LLMs are used **only after structure and evidence are established**, ensuring robustness and explainability.
