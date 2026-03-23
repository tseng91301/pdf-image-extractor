# 📄 PDF 圖片擷取與圖說推論系統

**PDF Image Extraction & Caption Inference System**

2026/03/20 22:30:00 開始運行資料建檔

## 📋 目錄

- [專案簡介](#專案簡介)
- [核心特性](#核心特性)
- [系統架構](#系統架構)
- [安裝指南](#安裝指南)
- [快速開始](#快速開始)
- [詳細使用方法](#詳細使用方法)
- [程式運作流程](#程式運作流程)
- [輸出格式](#輸出格式)
- [關鍵技術](#關鍵技術)
- [常見問題](#常見問題)
- [開發進度](#開發進度)

---

## 專案簡介

本專案是一個強大的 **PDF 圖片擷取與圖說推論系統**，專門為**掃描型（影像型）PDF** 設計，用於處理現實世界中品質不一、文字層損壞或缺失的 PDF 文件。

### 為什麼需要這個系統？

- 📄 **現實 PDF 問題複雜**：文字層缺失、混合格式、排版不穩定
- 🔍 **傳統方案失效**：直接提取文字層往往得到亂碼或不完整數據
- 🎯 **我們的解決方案**：將 PDF 視為視覺文件，透過版面分析和 OCR 提取內容

### 適用場景

✅ 掃描文件、電子書、舊版報告  
✅ 文件理解與知識提取  
✅ 圖片資料集建構  
✅ 知識庫 / RAG 系統  
✅ 搜尋與問答應用  

---

## 核心特性

| 功能 | 說明 |
|------|------|
| **PDF 影像化** | 將任何格式的 PDF 統一轉換為高解析度圖片 |
| **版面分析** | 自動偵測圖片、文字、標題、表格等元素 |
| **圖片擷取** | 精確裁切和提取圖片，並指派唯一 ID |
| **智能 OCR** | 選擇性 OCR，只掃描必要區域，速度快精度高 |
| **上下文理解** | 自動蒐集圖片周圍文字作為圖說候選 |
| **結構化輸出** | JSON 格式輸出，包含圖片、圖說、位置等完整訊息 |

---

## 系統架構

```
PDF 輸入
    ↓
[Step 1] 頁面影像化 (Page Rendering)
    ├─ 轉換為 PNG 圖片 (DPI: 100-300)
    └─ 輸出: page_0001.png, page_0002.png, ...
    ↓
[Step 2] 文件版面分析 (Layout Detection)
    ├─ 識別元素: 圖片、文字、標題、表格、圖說
    ├─ 使用模型: PP-DocLayout_plus-L
    └─ 輸出: 帶有座標和類型的區塊訊息
    ↓
[Step 3] 圖片擷取 (Figure Extraction)
    ├─ 根據版面分析結果裁切圖片
    ├─ 生成唯一 ID (UUID)
    └─ 輸出: fig_page2_01.png, fig_page3_02.png, ...
    ↓
[Step 4] 圖說候選蒐集 (Caption Candidate Collection)
    ├─ 基於距離、對齐、欄位等規則
    ├─ 收集圖片周圍的文字區塊
    └─ 輸出: 候選區塊清單
    ↓
[Step 5] 選擇性 OCR (Selective OCR)
    ├─ 只掃描圖說候選區域
    ├─ 自動檢測並處理亂碼
    └─ 輸出: 提取的文字
    ↓
[Step 6] 輸出與推論 (Inference & Export)
    ├─ 整合圖片 + 圖說 + 元數據
    └─ 輸出: JSON + PNG 檔案
```

---

## 安裝指南

### 系統需求

- Python 3.8+
- CUDA 11.0+ (選用，用於 GPU 加速)
- 至少 4GB RAM (建議 8GB+)

### 步驟 1：克隆專案

```bash
git clone https://github.com/tseng91301/pdf-image-extractor.git
cd pdf-image-extractor
```

### 步驟 2：建立虛擬環境（推薦）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 步驟 3：安裝依賴

```bash
# CPU
pip install -r requirements.cpu.txt

# GPU
pip install -r requirements.gpu.txt
```

### 步驟 4：驗證安裝

```bash
python -c "from converter import PdfInfo; print('✅ 安裝成功')"
```

---

## 快速開始

### 最簡單的方式：一鍵提取

```python
from converter import PdfInfo

# 初始化
pdf = PdfInfo('path/to/your/file.pdf')

# 一鍵執行全流程（推薦）
pdf.export_all_images_and_image_descriptions()

print("✅ 提取完成！檢查輸出資料")
```

**輸出：**
- 提取的圖片：`tmp/ImgData/[uuid]/image.png`
- 結構化資料：`output/images_data.json`

---

## 詳細使用方法

### 方法 1：分步驟控制

適合需要自定義各階段參數的進階使用者。

```python
from converter import PdfInfo

# 初始化 PDF
pdf = PdfInfo('sample.pdf', gpu=False)  # 設 gpu=True 啟用 GPU 加速

# Step 1: PDF 轉圖片（DPI 越高品質越好，但速度越慢）
print("▶ 正在轉換 PDF 為圖片...")
pdf.to_images(dpi=100)  # 預設 100 DPI，精度高應用可用 200-300

# Step 2: 版面分析
print("▶ 正在分析文件版面...")
pdf.label_layout(output=False)  # output=True 會顯示偵測結果

# Step 3: 圖片擷取與標記
print("▶ 正在擷取圖片...")
pdf.label_images(
    optimize_resolution=False,  # True: 從原 PDF 高解析度擷取圖片
    optimize_dpi=500,          # 高解析度時的 DPI
    use_xref=True              # 使用 PDF 內嵌圖片參考
)

# Step 4: 提取圖說與周圍文字
print("▶ 正在提取圖說...")
pdf.extract_image_description(
    export=True,  # True: 直接儲存結果
    nl=False      # True: OCR 結果換行顯示
)

# 或者手動匯出
pdf.export_all_image_datas(path='output/images_data.json')
```

### 方法 2：使用 Google Colab（無需本地環境）

直接執行 `colab_entry.ipynb`：

```python
# Colab 會自動克隆並安裝
!git clone https://github.com/tseng91301/pdf-image-extractor
!pip install -r pdf-image-extractor/requirements.txt

from converter import PdfInfo

pdf = PdfInfo('your_file.pdf')
pdf.export_all_images_and_image_descriptions()
```

---

## 程式運作流程

### 核心類：`PdfInfo`

```python
class PdfInfo:
    """管理整個 PDF 處理流程"""
    
    # 主要屬性
    pdf_path: str              # 輸入 PDF 路徑
    pdf_uid: str               # 唯一識別碼 (UUID)
    pdf_name: str              # PDF 檔名
    pdf_doc: fitz.Document     # PyMuPDF 文件物件
    use_gpu: bool              # 是否使用 GPU
    
    # 處理結果
    pdf_img_paths: list        # 頁面圖片路徑
    pdf_layouts: list          # 版面分析結果
    pdf_imgdatas: list         # 提取的圖片物件清單
```

### 核心類：`ImgData`

```python
class ImgData:
    """代表一個提取的圖片及其元數據"""
    
    # 圖片訊息
    uid: str                   # 圖片唯一 ID
    coordinate: list           # 座標 [x1, y1, x2, y2]
    store_path: str            # 儲存路徑
    
    # 圖說訊息
    image_figure_title: str    # 圖片標題
    image_surrounding_texts: list  # 周圍文字清單
    
    # 方法
    get_surroundings()         # 提取周圍文字與上下文
    to_json()                  # 轉換為 JSON
```

### 資料流範例

```
PDF 檔案
    ↓ (to_images)
[page_0001.png, page_0002.png, ...]
    ↓ (label_layout)
{
  "page": 0,
  "boxes": [
    {"type": "image", "coordinate": [100, 200, 500, 600]},
    {"type": "text", "coordinate": [100, 610, 500, 650], "text": "..."},
    ...
  ]
}
    ↓ (label_images)
ImgData(uid="abc123", coordinate=[100, 200, 500, 600], ...)
    ↓ (extract_image_description)
{
  "uid": "abc123",
  "figure_title": "圖 1：架構圖",
  "surrounding_texts": ["上文...", "下文..."],
  "image_path": "tmp/ImgData/abc123/image.png"
}
```

---

## 輸出格式

### JSON 輸出結構

```json
{
  "pdf_name": "report_2024",
  "total_pages": 5,
  "total_images": 3,
  "imgs": [
    {
      "uid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "page": 1,
      "coordinate": [100, 200, 500, 600],
      "figure_title": "圖 1：系統架構",
      "figure_title_box": {
        "coordinate": [100, 610, 500, 630],
        "text": "圖 1：系統架構"
      },
      "surrounding_texts": [
        {
          "text": "本系統採用模組化設計...",
          "box": [100, 50, 500, 100],
          "type": "text"
        }
      ],
      "image_path": "tmp/ImgData/a1b2c3d4-e5f6-7890-abcd-ef1234567890/image.png"
    }
  ]
}
```

### 檔案輸出

```
output/
├── images_data.json           # 完整的元數據
├── extracted_images/
│   ├── page_0001.png
│   ├── page_0002.png
│   └── ...
└── figures/
    ├── a1b2c3d4.../image.png
    ├── b2c3d4e5.../image.png
    └── ...
```

---

## 關鍵技術

| 技術 | 用途 | 備註 |
|------|------|------|
| **PyMuPDF (fitz)** | PDF 渲染與圖片提取 | 支援高 DPI、高速 |
| **PaddleOCR** | 光學字符識別 | 支援繁體中文，精度高 |
| **PP-DocLayout_plus-L** | 文件版面分析 | 支援多種文件類型 |
| **OpenCV** | 圖像處理 | 用於圖片裁切、檢測 |
| **Sentence Transformers** | 語義檢索 | FAISS 向量搜尋 (可選) |
| **FAISS** | 相似度搜尋 | 用於圖片、文字檢索 |

---

## 常見問題

### Q1: 為什麼提取出來是亂碼？

**原因**：PDF 的文字層或字型映射損壞

**解決方案**：
```python
# 系統會自動檢測並使用 OCR
pdf.extract_image_description(nl=False)
```

### Q2: 圖片品質不好怎麼辦？

**提升品質**：

```python
# 使用更高的 DPI
pdf.to_images(dpi=200)  # 預設 100，可提升到 200-300

# 或優化提取解析度
pdf.label_images(
    optimize_resolution=True,
    optimize_dpi=500      # 從原 PDF 高解析度擷取
)
```

### Q3: 處理速度太慢？

**加速方案**：
```python
# 使用 GPU 加速
pdf = PdfInfo('file.pdf', gpu=True)

# 或降低 DPI (犧牲品質換速度)
pdf.to_images(dpi=75)
```

### Q4: 可以自定義輸出位置嗎？

```python
# 自定義輸出路徑
pdf.export_all_image_datas(path='my_custom_output/data.json')
```

### Q5: 如何批量處理多個 PDF？

```python
import os
from converter import PdfInfo

pdf_dir = 'path/to/pdfs'
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith('.pdf'):
        pdf = PdfInfo(os.path.join(pdf_dir, pdf_file))
        pdf.export_all_images_and_image_descriptions()
        print(f"✅ 完成: {pdf_file}")
```

---

## 開發進度

### ✅ 已完成 (Phase 1: MVP)

- [x] PDF → 頁面影像轉換
- [x] 基本版面分析（圖片＋文字）
- [x] 圖片裁切與 ID 指派
- [x] 距離與欄位規則的圖說候選蒐集
- [x] 區塊級 OCR
- [x] 輸出圖片＋JSON 結果

### 🚧 進行中 (Phase 2: 功能擴展)

- [ ] 表格識別與提取
- [ ] 多語言支援優化
- [ ] GPU 加速完全支援
- [ ] 批量處理 UI

### 📋 計畫中 (Phase 3: 高級功能)

- [ ] 使用 LLM 進行智能圖說推論
- [ ] 向量搜尋整合
- [ ] 知識圖譜提取
- [ ] 異步處理與分佈式支援

---

## 依賴列表

```
numpy                          # 數值計算
opencv-python                  # 圖像處理
Pillow                         # 圖片操作
PyMuPDF                        # PDF 處理
paddleocr                      # 光學字符識別
paddlepaddle                   # PaddleOCR 後端
matplotlib                     # 視覺化
sentence-transformers          # 語義 embedding
faiss                          # 向量搜尋
```

---

## 許可証

本專案採用 MIT License

---

## 聯絡與反饋

如有問題或建議，歡迎提出 Issue 或 Pull Request！

**作者**: [@tseng91301](https://github.com/tseng91301)  
**Last Updated**: 2026-01-30
