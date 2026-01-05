# 📄 PDF 圖片擷取與圖說推論系統

（PDF Image Caption Inference System）

## 一、專案簡介（Project Overview）

本專案旨在建立一套**以掃描型（影像型）PDF 為優先設計的圖片擷取與圖說推論系統**，用於處理現實世界中品質不一、結構不穩定、文字層不可依賴的 PDF 文件。

系統不嘗試直接解析 PDF 內嵌文字，而是將每一份 PDF 視為**視覺文件（Visual Document）**，透過文件版面理解（layout understanding）來擷取圖片，並結合圖片周邊上下文與語意推論，自動產生對應的圖片說明（caption）。

最終輸出為結構化資料，適合用於：

* 文件理解與分析
* 圖片資料集建構
* 知識庫 / RAG 系統
* 搜尋與問答應用

---

## 二、設計理念（Design Philosophy）

### 為什麼以掃描型 PDF 為核心？

實務上，多數 PDF 存在以下問題：

* 文字層缺失或損壞
* 混合掃描頁與文字頁
* 排版不一致、來源不明

因此本專案採取以下策略：

> **無論 PDF 原始格式為何，一律當作影像文件處理**

這樣可以：

* 避免流程分支與例外處理
* 提升系統穩定性與可重現性
* 確保對各類 PDF 的一致行為

---

### 核心原則

> **先理解文件結構，再回推語意；
> 不先讀文字，而是先看版面。**

---

## 三、系統整體架構（High-Level Architecture）

```
PDF
 └─ 頁面影像化（Page Rendering）
      ├─ 文件版面分析（Layout Detection）
      │    ├─ 圖片區塊（Figure）
      │    ├─ 文字區塊（Text）
      │    ├─ 圖說區塊（Caption）
      │    └─ 表格區塊（Table）
      │
      ├─ 區塊級 OCR（Selective OCR）
      │
      ├─ 圖文關聯推論（Figure–Context Association）
      │
      └─ 圖說推論與生成（LLM 輔助）
```

---

## 四、處理流程說明（Processing Pipeline）

### Step 1：PDF 頁面影像化

* 將 PDF 每一頁轉為高解析影像（建議 200–300 DPI）
* **完全忽略 PDF 文字層**

**輸出**

* `page_001.png`, `page_002.png`, …

---

### Step 2：文件版面分析（Layout Analysis）

偵測並分類頁面中的區塊：

* 標題（Title）
* 正文（Text）
* 圖片（Figure）
* 圖說（Caption）
* 表格（Table）

每個區塊包含：

* 區塊類型
* Bounding Box（座標）
* 所屬頁碼

**輸出範例**

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

### Step 3：圖片擷取（Figure Extraction）

* 依 layout 結果裁切圖片區塊
* 為每張圖片指派唯一 ID

**輸出**

* `fig_page2_01.png`
* `fig_page3_02.png`

---

### Step 4：圖說候選區塊蒐集

針對每一張圖片，依據以下條件蒐集可能的圖說文字區塊：

* 與圖片的距離
* 是否位於同一欄位
* 上下／左右相對位置
* 閱讀順序（reading order）

目的不是立刻判斷正確圖說，而是：

> **先縮小搜尋範圍**

---

### Step 5：選擇性 OCR（Selective OCR）

僅對以下區塊執行 OCR：

* 圖說候選文字區塊
* （選擇性）提及圖片的前後文段落（如「如圖 3 所示」）

**優點**

* 提升效率
* 降低 OCR 錯誤
* 簡化後處理

---

### Step 6：圖說判定與語意推論

結合規則與 LLM，將圖說分為三類：

1. **明確圖說（Exact Caption）**

   * 出現「圖 X / Figure X」等明確標記
2. **鄰近圖說（Nearby Caption）**

   * 圖片附近描述性文字
3. **推論圖說（Inferred Caption）**

   * 由上下文語意推論生成

所有圖說皆附上來源與信心標記。

---

## 五、輸出資料格式（Structured Output）

```json
{
  "figure_id": "page2_fig1",
  "page": 2,
  "image_path": "fig_page2_01.png",
  "caption_type": "inferred",
  "caption_text": "比較傳統人工除草與鋪設不織布後之雜草生長情形。",
  "evidence": {
    "nearby_text_blocks": [...],
    "layout_relation": "below_figure"
  }
}
```

---

## 六、預計使用關鍵技術（Key Technologies）

* PDF 影像化：pdfium / poppler / PyMuPDF
* 版面分析：LayoutParser、Detectron2（DocLayNet / PubLayNet）
* OCR：PaddleOCR（支援繁體中文）
* 語意推論：LLM（僅使用結構化證據）
* 資料儲存：JSON / SQLite / Parquet

---

## 七、TODO 清單（開發里程碑）

### Phase 1：最小可行版本（MVP）

* [ ] PDF → 頁面影像轉換
* [ ] 基本版面分析（圖片＋文字）
* [ ] 圖片裁切與 ID 指派
* [ ] 距離與欄位規則的圖說候選蒐集
* [ ] 區塊級 OCR
* [ ] 輸出圖片＋JSON 結果

---

### Phase 2：版面與關聯強化

* [ ] 多欄位閱讀順序推斷
* [ ] 圖說格式正規化（OCR 錯誤修正）
* [ ] 跨頁圖說匹配
* [ ] 表格與圖片區分
* [ ] 圖說信心分數計算

---

### Phase 3：語意推論與生成

* [ ] LLM 圖說重排序（rerank）
* [ ] 上下文推論圖說生成
* [ ] 幻覺控制（僅使用證據）
* [ ] 明確標示「擷取 vs 推論」

---

### Phase 4：實用化與擴充

* [ ] 批次 PDF 處理
* [ ] CLI / API 介面
* [ ] 版面與圖文關係視覺化除錯工具
* [ ] 輸出供 RAG / ML 使用的資料集

---

## 八、長期擴充方向（Optional）

* 文件層級摘要（含圖片引用）
* 多語言圖說正規化
* 論文／技術報告專用模式
* Human-in-the-loop 校正介面

---

## 九、結語

本專案**不是 PDF 文字解析工具**，
而是一套以「圖片為核心」的**文件理解系統**。

LLM 僅在結構與證據確立後介入，用於語意整合與推論，
以確保結果可解釋、可追溯、可擴充。