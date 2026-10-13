已為您在 `pdf-image-extractor-2` 完整重現 PDF 圖片與上下文標註擷取管線，並解決 Paddle 與 Transformers 的 GPU 衝突，建立專屬的資料庫建置腳本 `build_database.sh`。

---

### 一、 核心架構與獨立性設計

這兩個資料夾已實現**完全獨立、互不引用**：

1. **獨立元件重現 (`pdf-image-extractor-2/core/converter`)**：
   - [PdfInfo](file:///home/tseng/external/pdf-image-extractor-2/core/converter/__init__.py)：PDF 頁面高解析度渲染、版面佈局偵測 (`PP-DocLayout_plus-L`)、圖片裁切與抽取、圖片描述提取與匯出。
   - [ImgData](file:///home/tseng/external/pdf-image-extractor-2/core/converter/img_data.py)：圖表座標關聯、圖題（figure title）偵測、周圍文字距離計算與上下文提取。
   - [ImgOcr](file:///home/tseng/external/pdf-image-extractor-2/core/converter/img2text.py)：基於 PaddleOCR 的圖片與文字區塊 OCR。
   - [輔助工具模組](file:///home/tseng/external/pdf-image-extractor-2/core/converter/tools/)：包含 `coordinates.py`（座標轉換）、`random_uid.py`（隨機 UID 生成）與 `text_validation.py`（PDF 亂碼檢測與修復）。
   - **記憶體與暫存最佳化**：加入了完整的暫存檔案清理機制（`cleanup`），避免大量處理 PDF 時塞滿硬碟。

2. **雙 venv 環境配置（解決 Paddle 與 Transformers 之 GPU 衝突）**：
   - **`venv_paddle`**（擷取專用環境）：包含 `paddlepaddle-gpu`、`paddleocr`、`PyMuPDF (fitz)`、`opencv-python`、`Pillow` 等。專責處理版面分析、圖片裁切與 OCR。
   - **`venv`**（檢索與伺服器專用環境）：包含 `torch`、`sentence-transformers` (`BAAI/bge-small-zh-v1.5`、`BAAI/bge-reranker-base`)、`rank-bm25`、`jieba`、`fastapi`、`uvicorn`。專責建立混合檢索索引與伺服器運行。
   - **隔離效益**：擷取階段與檢索階段於各自的行程中執行，釋放全部 CUDA 上下文，彼此在 GPU 上完全互不干擾。

---

### 二、 建置資料庫腳本：`build_database.sh`

已在 `pdf-image-extractor-2/` 建立執行腳本 [build_database.sh](file:///home/tseng/external/pdf-image-extractor-2/build_database.sh) 及 Python 主程式 [build_database.py](file:///home/tseng/external/pdf-image-extractor-2/build_database.py)。

#### 1. 基本用法

```bash
cd /home/tseng/external/pdf-image-extractor-2

# 語法：./build_database.sh <原始_PDF_目錄> [輸出目錄]
./build_database.sh ./test_pdfs
```

- 若未指定輸出目錄，預設會存放在 `./image_data/`，索引存放在 `./index_cache/`。
- 若指定了自訂輸出目錄（例如 `./build_database.sh /path/to/pdfs /path/to/my_output`），腳本會自動建立連結符號，讓 [start_server.sh](file:///home/tseng/external/pdf-image-extractor-2/start_server.sh) 能直接無縫讀取。

#### 2. 常用選項

| 選項 | 說明 | 範例 |
| :--- | :--- | :--- |
| `-s, --source <dir>` | 指定原始 PDF 文件目錄 | `-s ./test_pdfs` |
| `-o, --output <dir>` | 指定輸出目錄（預設: `./image_data`） | `-o ./image_data` |
| `-c, --cache <dir>` | 指定檢索索引快取目錄（預設: `./index_cache`） | `-c ./index_cache` |
| `-l, --limit <n>` | 限制最多處理幾份文件（測試用） | `--limit 5` |
| `--dpi <n>` | PDF 轉圖片 DPI（預設: 300） | `--dpi 300` |
| `--optimize-dpi <n>` | 裁切圖片的最佳化高解析度 DPI（預設: 400） | `--optimize-dpi 400` |
| `--use-xref` | 優先使用 PDF 內部 xref 抽取內嵌點陣圖 | `--use-xref` |
| `--cpu` | 強制以 CPU 執行版面分析與 OCR | `--cpu` |
| `--skip-extract` | 跳過圖片擷取，僅重建檢索索引 | `--skip-extract` |
| `--skip-index` | 僅擷取圖片，跳過檢索索引建立 | `--skip-index` |

#### 3. 斷點續傳機制
- 處理過的 PDF 相對路徑會自動記錄在 `processed_docs.json`（預設儲存於輸出目錄下）。
- 若處理過程中中斷（支援 `Ctrl+C` 優雅儲存），再次執行時會**自動跳過已完成的 PDF 文件**。

---

### 三、 執行流程與與 `start_server.sh` 銜接

`build_database.sh` 執行時分為兩階段：

1. **【階段 1/2】PDF 圖片與文字擷取（使用 `venv_paddle`）**：
   - 遍歷指定目錄及其所有子目錄下的 `.pdf` 檔案。
   - 產出結構符合伺服器規格：
     ```text
     image_data/
     ├── <pdf_uid>/
     │   └── image_datas/
     │       ├── metadata.json
     │       ├── image_0000.png
     │       └── image_0001.png
     └── processed_docs.json
     ```
2. **【階段 2/2】多模態混合檢索索引建置（使用 `venv`）**：
   - 呼叫 `core.index_builder`，為所有圖文建立 BM25 關鍵字索引與 Dense 語意向量矩陣。
   - 產生 `documents.pkl`、`bm25_index.pkl`、`dense_embeddings.npz` 快取於 `index_cache/`。
3. **啟動伺服器**：
   - 建置完成後，直接執行既有的啟動腳本即可：
     ```bash
     ./start_server.sh
     ```
   - 伺服器啟動時會直接從 `index_cache/` 載入建立好的索引，並可直接預覽與搜尋所有抽取出來的圖片。