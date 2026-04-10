#!/bin/bash

# Define session name
SESSION="pdf_extractor"

# Check if session already exists
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Killing existing session: $SESSION"
    tmux kill-session -t $SESSION
fi

# Create new session, detached
tmux new-session -d -s $SESSION

# Send commands: activate venv and run app.py
tmux send-keys -t $SESSION "source venv/bin/activate" C-m
tmux send-keys -t $SESSION "python app.py" C-m

echo "Tmux session '$SESSION' restarted and app.py started."

# Attach to the session
tmux attach-session -t $SESSION
