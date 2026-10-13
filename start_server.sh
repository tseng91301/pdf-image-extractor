#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-${SCRIPT_DIR}/image_data}"
export CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/index_cache}"

exec "${SCRIPT_DIR}/venv/bin/python" -m uvicorn server.app:app --host 0.0.0.0 --port 25643 --reload
