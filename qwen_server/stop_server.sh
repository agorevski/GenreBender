#!/bin/bash
# Stop QWEN server

echo "Stopping QWEN server..."

# Find and kill the server process
PID=$(ps aux | grep '[p]ython3 server.py' | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "Server is not running"
else
    kill $PID
    echo "Server stopped (PID: $PID)"
fi
