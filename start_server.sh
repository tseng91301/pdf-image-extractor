#!/bin/bash

# 取得目前腳本 (start_server.sh) 所在的絕對路徑
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

SESSION="pdf-image-extractor-2"

# 檢查 tmux 是否已安裝
if ! command -v tmux &> /dev/null; then
    echo "錯誤: 未安裝 tmux。請先安裝再執行此腳本。"
    exit 1
fi

# 如果 session 已存在，則先殺掉並釋放埠號
if tmux has-session -t $SESSION 2>/dev/null; then
    tmux kill-session -t $SESSION 2>/dev/null
    sleep 1
fi
fuser -k 25643/tcp 2>/dev/null || true

# 設定環境變數預設值
export IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-${SCRIPT_DIR}/image_data}"
export CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/index_cache}"

# 建立新的 tmux session，直接啟動 Backend，並指定工作目錄
tmux new-session -d -s $SESSION -n "FastAPI-Backend" -c "$SCRIPT_DIR"

# 在第一個視窗設定環境變數並啟動 Backend
tmux send-keys -t $SESSION:0 "export IMAGE_DATA_DIR=\"$IMAGE_DATA_DIR\" && export CACHE_DIR=\"$CACHE_DIR\" && source venv/bin/activate && python -m uvicorn server.app:app --host 0.0.0.0 --port 25643 --reload" C-m

echo "已在 tmux 會話 '$SESSION' 中啟動服務。"
echo "FastAPI Backend: 視窗口 0 (http://localhost:25643)"
