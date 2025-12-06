#!/bin/bash
# Start QWEN server

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Get configuration
PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['server']['port'])")
HOST=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['server']['host'])")

echo "=========================================="
echo "Starting QWEN Server"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "API Key: helloagorevski"
echo ""
echo "Endpoints:"
echo "  Health: http://localhost:$PORT/health"
echo "  Analyze: http://localhost:$PORT/analyze"
echo "  Batch: http://localhost:$PORT/analyze_batch"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start server
python3 server.py
