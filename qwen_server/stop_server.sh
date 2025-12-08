#!/bin/bash
# Stop QWEN server(s)

echo "Stopping QWEN server(s)..."

# PID file location
PID_FILE=".server_pids"

if [ -f "$PID_FILE" ]; then
    # Multi-server mode: Kill all PIDs from file
    echo "Found PID file with multiple servers"
    
    while read PID; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping server (PID: $PID)..."
            kill $PID
        else
            echo "Process $PID not running"
        fi
    done < "$PID_FILE"
    
    # Remove PID file
    rm -f "$PID_FILE"
    echo "All servers stopped"
    
else
    # Single-server mode: Find process by name
    PIDS=$(ps aux | grep '[p]ython3 server.py' | awk '{print $2}')
    
    if [ -z "$PIDS" ]; then
        echo "No server processes found"
    else
        for PID in $PIDS; do
            echo "Stopping server (PID: $PID)..."
            kill -9 $PID
        done
        echo "Server stopped"
    fi
fi

# Clean up log files (optional - comment out if you want to keep logs)
# rm -f server_*.log

echo "Done"
