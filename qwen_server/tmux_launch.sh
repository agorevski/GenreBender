#!/bin/bash
# Launch tmux monitoring session for QWEN server
# Creates a single window with 3 panes: command (top), GPU (bottom-left), CPU (bottom-right)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION_NAME="qwen-server"

# Check for required tools
check_dependencies() {
    local missing=()
    
    if ! command -v tmux &> /dev/null; then
        missing+=("tmux")
    fi
    if ! command -v nvtop &> /dev/null; then
        missing+=("nvtop")
    fi
    if ! command -v htop &> /dev/null; then
        missing+=("htop")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo "ERROR: Missing required tools: ${missing[*]}"
        echo "Install with: sudo apt install ${missing[*]}"
        exit 1
    fi
}

# Main execution
check_dependencies

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
fi

echo "Creating tmux session: $SESSION_NAME"
echo ""

# Create new tmux session with main window and set terminal size
tmux new-session -d -s "$SESSION_NAME" -n "monitor" -c "$SCRIPT_DIR" -x 275 -y 75

# Layout: 
# +---------------------------+
# |         command           |
# |        (top pane)         |
# +-------------+-------------+
# |    nvtop    |    htop     |
# | (GPU, left) | (CPU, right)|
# +-------------+-------------+

# Split horizontally (creates top and bottom)
tmux split-window -t "$SESSION_NAME:monitor" -v -c "$SCRIPT_DIR"

# Split the bottom pane vertically (creates left and right on bottom)
tmux split-window -t "$SESSION_NAME:monitor.1" -h -c "$SCRIPT_DIR"

# Now we have:
# Pane 0: top (command)
# Pane 1: bottom-left (nvtop)
# Pane 2: bottom-right (htop)

# Set pane sizes:
# - Top pane (command): 30 lines (75 - 45 = 30)
# - Bottom panes: 45 lines, split 50/50 horizontally
tmux resize-pane -t "$SESSION_NAME:monitor.0" -y 30
tmux resize-pane -t "$SESSION_NAME:monitor.1" -x 137

# Start nvtop in bottom-left pane
tmux send-keys -t "$SESSION_NAME:monitor.1" "nvtop" C-m

# Start htop in bottom-right pane
tmux send-keys -t "$SESSION_NAME:monitor.2" "htop" C-m

# Run the server startup script in the top (command) pane
tmux send-keys -t "$SESSION_NAME:monitor.0" "./start_server.sh" C-m

# Select the top pane (command) as active
tmux select-pane -t "$SESSION_NAME:monitor.0"

# Give user instructions before attaching
echo "=========================================="
echo "QWEN Server tmux Session"
echo "=========================================="
echo ""
echo "Layout:"
echo "  +---------------------------+"
echo "  |         command           |"
echo "  |   (server startup/shell)  |"
echo "  +-------------+-------------+"
echo "  |    nvtop    |    htop     |"
echo "  |  (GPU mon)  |  (CPU mon)  |"
echo "  +-------------+-------------+"
echo ""
echo "Navigation:"
echo "  Ctrl+b ↑/↓/←/→  - Switch between panes"
echo "  Ctrl+b z        - Zoom/unzoom current pane"
echo "  Ctrl+b d        - Detach from session"
echo ""
echo "To reattach later: tmux attach -t $SESSION_NAME"
echo "=========================================="
echo ""
echo "Attaching to tmux session..."
sleep 2

# Attach to the session
tmux attach -t "$SESSION_NAME"
