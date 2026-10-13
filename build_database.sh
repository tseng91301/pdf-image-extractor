#!/usr/bin/env bash
# ==============================================================================
# build_database.sh
# End-to-end PDF processing and Multimodal Search Database Builder
#
# Usage:
#   ./build_database.sh [options] <source_pdf_dir> [output_dir]
#
# Examples:
#   ./build_database.sh ./test_pdfs
#   ./build_database.sh ./test_pdfs ./image_data
#   ./build_database.sh -s ./test_pdfs -o ./image_data --limit 5
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
SOURCE_DIR=""
OUTPUT_DIR="${SCRIPT_DIR}/image_data"
CACHE_DIR="${SCRIPT_DIR}/index_cache"
VENV_PADDLE="${SCRIPT_DIR}/venv_paddle"
VENV_SEARCH="${SCRIPT_DIR}/venv"
DPI=300
OPTIMIZE_DPI=400
LIMIT=""
USE_XREF=""
NO_GPU=""
SKIP_EXTRACT=false
SKIP_INDEX=false

usage() {
    cat << 'HELP'
使用說明:
  ./build_database.sh [選項] <原始_pdf_目錄> [輸出目錄]

參數:
  <原始_pdf_目錄>       存放原始 PDF 文件的目錄 (會遞迴遍歷該目錄下的所有 .pdf 檔案)
  [輸出目錄]            存放擷取出的圖片與 metadata.json 之目錄 (預設: ./image_data)

選項:
  -s, --source <dir>    指定原始 PDF 目錄 (亦可直接使用第一個位置參數)
  -o, --output <dir>    指定輸出目錄 (預設: ./image_data)
  -c, --cache <dir>     指定索引快取目錄 (預設: ./index_cache)
  -l, --limit <n>       最多處理 n 個 PDF 檔案 (測試用)
  --dpi <n>             PDF 轉影像解析度 DPI (預設: 300)
  --optimize-dpi <n>    高解析度裁切圖片 DPI (預設: 400)
  --use-xref            優先使用 PDF xref 提取內嵌圖片 (若找不到則用渲染裁切)
  --cpu                 強制使用 CPU 進行 Layout 偵測與 OCR (預設: 優先使用 GPU)
  --skip-extract        跳過 PDF 擷取步驟，僅對現有輸出目錄建立搜尋索引
  --skip-index          跳過搜尋索引建立步驟，僅執行 PDF 圖片擷取
  -h, --help            顯示此說明訊息

範例:
  # 處理 test_pdfs 目錄並輸出至預設 image_data，然後建立索引:
  ./build_database.sh ./test_pdfs

  # 指定輸出目錄:
  ./build_database.sh ./test_pdfs ./my_image_data

  # 僅處理 5 篇文件進行測試:
  ./build_database.sh ./test_pdfs ./image_data --limit 5
HELP
    exit 0
}

# Parse Arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -s|--source|--source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -o|--output|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--cache|--cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --dpi)
            DPI="$2"
            shift 2
            ;;
        --optimize-dpi)
            OPTIMIZE_DPI="$2"
            shift 2
            ;;
        --use-xref)
            USE_XREF="--use-xref"
            shift
            ;;
        --cpu)
            NO_GPU="--no-gpu"
            shift
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --skip-index)
            SKIP_INDEX=true
            shift
            ;;
        -*)
            echo "[ERROR] 未知的選項: $1"
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Handle positional arguments
if [[ -z "$SOURCE_DIR" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    SOURCE_DIR="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    OUTPUT_DIR="${POSITIONAL_ARGS[1]}"
fi

# Validation
if [[ "$SKIP_EXTRACT" = false ]]; then
    if [[ -z "$SOURCE_DIR" ]]; then
        echo "[ERROR] 請指定要建立資料庫的原始 PDF 目錄！"
        echo "例如: ./build_database.sh ./test_pdfs"
        echo "使用 ./build_database.sh --help 查看詳細說明。"
        exit 1
    fi
    if [[ ! -d "$SOURCE_DIR" ]]; then
        echo "[ERROR] 找不到指定的 PDF 目錄: $SOURCE_DIR"
        exit 1
    fi
fi

# Resolve absolute paths
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
CACHE_DIR="$(mkdir -p "$CACHE_DIR" && cd "$CACHE_DIR" && pwd)"
if [[ -n "$SOURCE_DIR" ]]; then
    SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
fi

echo "======================================================================"
echo "🚀 啟動 PDF 資料庫建置程序 (PDF Image Extractor 2)"
echo "======================================================================"
echo "原始 PDF 目錄: $SOURCE_DIR"
echo "結果輸出目錄: $OUTPUT_DIR"
echo "索引快取目錄: $CACHE_DIR"
echo "======================================================================"

# Check Python Environments
if [[ "$SKIP_EXTRACT" = false ]]; then
    if [[ ! -x "${VENV_PADDLE}/bin/python" ]]; then
        echo "[ERROR] 找不到 Paddle 專用虛擬環境: ${VENV_PADDLE}/bin/python"
        echo "請確認 ${VENV_PADDLE} 是否存在。"
        exit 1
    fi
fi

if [[ "$SKIP_INDEX" = false ]]; then
    if [[ ! -x "${VENV_SEARCH}/bin/python" ]]; then
        echo "[ERROR] 找不到搜尋引擎專用虛擬環境: ${VENV_SEARCH}/bin/python"
        echo "請確認 ${VENV_SEARCH} 是否存在。"
        exit 1
    fi
fi

# ----------------------------------------------------------------------
# 階段 1: PDF 圖片與上下文標註擷取 (使用 Paddle venv，獨立行程，無 GPU 衝突)
# ----------------------------------------------------------------------
if [[ "$SKIP_EXTRACT" = false ]]; then
    echo ""
    echo "----------------------------------------------------------------------"
    echo "【階段 1/2】擷取 PDF 圖片與周圍說明文字 (使用 Paddle venv)"
    echo "----------------------------------------------------------------------"
    
    EXTRACTION_CMD=(
        "${VENV_PADDLE}/bin/python" "build_database.py"
        --source-dir "$SOURCE_DIR"
        --output-dir "$OUTPUT_DIR"
        --dpi "$DPI"
        --optimize-dpi "$OPTIMIZE_DPI"
    )

    if [[ -n "$LIMIT" ]]; then
        EXTRACTION_CMD+=(--limit "$LIMIT")
    fi
    if [[ -n "$USE_XREF" ]]; then
        EXTRACTION_CMD+=($USE_XREF)
    fi
    if [[ -n "$NO_GPU" ]]; then
        EXTRACTION_CMD+=($NO_GPU)
    fi

    # 執行擷取，並加速 PaddleX 啟動
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True "${EXTRACTION_CMD[@]}"
    echo "✅ 【階段 1/2】PDF 圖片擷取完成！結果儲存於: $OUTPUT_DIR"
else
    echo "ℹ️  跳過【階段 1/2】PDF 圖片擷取。"
fi

# ----------------------------------------------------------------------
# 階段 2: 建立 BM25 與 Dense 向量檢索索引 (使用 PyTorch/Transformers venv)
# ----------------------------------------------------------------------
if [[ "$SKIP_INDEX" = false ]]; then
    echo ""
    echo "----------------------------------------------------------------------"
    echo "【階段 2/2】建立多模態混合檢索索引 (使用 Search venv)"
    echo "----------------------------------------------------------------------"

    "${VENV_SEARCH}/bin/python" -m core.index_builder \
        --data-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --force

    echo "✅ 【階段 2/2】檢索索引建置完成！快取儲存於: $CACHE_DIR"
else
    echo "ℹ️  跳過【階段 2/2】檢索索引建置。"
fi

# ----------------------------------------------------------------------
# 階段 3: 確保 start_server.sh 可直接使用
# ----------------------------------------------------------------------
DEFAULT_IMAGE_DATA="${SCRIPT_DIR}/image_data"
DEFAULT_CACHE="${SCRIPT_DIR}/index_cache"

# 若自訂輸出目錄不是預設的 ./image_data，建立/更新符號連結以利 start_server.sh 直接取用
if [[ "$OUTPUT_DIR" != "$DEFAULT_IMAGE_DATA" ]]; then
    if [[ ! -e "$DEFAULT_IMAGE_DATA" || -L "$DEFAULT_IMAGE_DATA" ]]; then
        rm -f "$DEFAULT_IMAGE_DATA"
        ln -s "$OUTPUT_DIR" "$DEFAULT_IMAGE_DATA"
        echo "🔗 已建立符號連結: ./image_data -> $OUTPUT_DIR"
    else
        echo "ℹ️  注意: 輸出目錄為自訂路徑 ($OUTPUT_DIR)。"
        echo "    啟動伺服器時可使用: IMAGE_DATA_DIR=$OUTPUT_DIR ./start_server.sh"
    fi
fi

if [[ "$CACHE_DIR" != "$DEFAULT_CACHE" ]]; then
    if [[ ! -e "$DEFAULT_CACHE" || -L "$DEFAULT_CACHE" ]]; then
        rm -f "$DEFAULT_CACHE"
        ln -s "$CACHE_DIR" "$DEFAULT_CACHE"
        echo "�� 已建立符號連結: ./index_cache -> $CACHE_DIR"
    fi
fi

echo ""
echo "======================================================================"
echo "🎉 資料庫建置完成！"
echo "======================================================================"
echo "處理好的結果目錄: $OUTPUT_DIR"
echo "已建置索引快取:   $CACHE_DIR"
echo ""
echo "現在您可以直接啟動伺服器:"
echo "  ./start_server.sh"
echo "======================================================================"
