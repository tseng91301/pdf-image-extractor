#!/bin/bash

# 取得目前腳本所在的絕對路徑
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Define session name
SESSION="pdf_extractor"

# Check if session already exists
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Killing existing session: $SESSION"
    tmux kill-session -t $SESSION
fi

# Create new session, detached
tmux new-session -d -s $SESSION

# Send commands: cd to script directory, activate venv and run app.py
tmux send-keys -t $SESSION "cd $SCRIPT_DIR" C-m
tmux send-keys -t $SESSION "source venv/bin/activate" C-m
tmux send-keys -t $SESSION "python app.py" C-m

echo "Tmux session '$SESSION' restarted and app.py started."
