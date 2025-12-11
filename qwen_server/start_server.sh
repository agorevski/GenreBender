#!/bin/bash
# Start QWEN server (single or multi-server mode)

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -d "venv_qwen_server" ]; then
    source venv_qwen_server/bin/activate
    echo "Virtual environment activated"
fi

# Get configuration
MULTI_SERVER_MODE=$(python -c "import yaml; config=yaml.safe_load(open('config.yaml')); print(str(config['server'].get('multi_server_mode', False)).lower())")
BASE_PORT=$(python -c "import yaml; config=yaml.safe_load(open('config.yaml')); print(config['server'].get('multi_server_base_port', 8000))")
HOST=$(python -c "import yaml; config=yaml.safe_load(open('config.yaml')); print(config['server']['host'])")
API_KEY=$(python -c "import yaml; config=yaml.safe_load(open('config.yaml')); print(config['server']['api_key'])")

echo "=========================================="
echo "Starting QWEN Server"
echo "=========================================="
echo "Host: $HOST"
echo "API Key: $API_KEY"
echo ""

# PID file for tracking server processes
PID_FILE=".server_pids"
rm -f "$PID_FILE"

if [ "$MULTI_SERVER_MODE" = "true" ]; then
    # Multi-server mode: Start N servers (one per GPU)
    echo "Mode: Multi-Server (True Multi-GPU Parallelism)"
    
    # Get GPU list
    GPUS=$(python -c "import yaml; config=yaml.safe_load(open('config.yaml')); print(' '.join(map(str, config['server'].get('multi_server_gpus', [0, 1, 2, 3]))))")
    GPU_ARRAY=($GPUS)
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    echo "GPUs: ${GPU_ARRAY[*]}"
    echo "Ports: $BASE_PORT-$(($BASE_PORT + $NUM_GPUS - 1))"
    echo ""
    
    # Start one server per GPU
    for i in "${!GPU_ARRAY[@]}"; do
        GPU_ID="${GPU_ARRAY[$i]}"
        PORT=$(($BASE_PORT + $i))
        
        echo "Starting Server $((i+1))/$NUM_GPUS on GPU $GPU_ID, Port $PORT..."
        
        # Start server in background with specific GPU
        CUDA_VISIBLE_DEVICES=$GPU_ID python server.py --port $PORT > "server_${PORT}.log" 2>&1 &
        SERVER_PID=$!
        echo $SERVER_PID >> "$PID_FILE"
        
        echo "  → PID: $SERVER_PID, Log: server_${PORT}.log"
        
        # Brief pause to stagger startup
        sleep 1
    done
    
    echo ""
    echo "All servers started. Waiting for health checks..."
    sleep 10
    
    # Health check all servers
    echo ""
    FAILED_SERVERS=0
    for i in "${!GPU_ARRAY[@]}"; do
        GPU_ID="${GPU_ARRAY[$i]}"
        PORT=$(($BASE_PORT + $i))
        
        if curl -s -f "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "✓ Server $((i+1)) (GPU $GPU_ID, Port $PORT): Healthy"
        else
            echo "✗ Server $((i+1)) (GPU $GPU_ID, Port $PORT): Not responding"
            FAILED_SERVERS=$((FAILED_SERVERS + 1))
        fi
    done
    
    echo ""
    if [ $FAILED_SERVERS -eq 0 ]; then
        echo "=========================================="
        echo "✓ All $NUM_GPUS servers are running!"
        echo "=========================================="
        echo ""
        echo "Endpoints:"
        for i in "${!GPU_ARRAY[@]}"; do
            PORT=$(($BASE_PORT + $i))
            echo "  Server $((i+1)): http://localhost:${PORT}"
        done
        echo ""
        echo "Configure pipeline with:"
        echo "  server_urls: ["
        for i in "${!GPU_ARRAY[@]}"; do
            PORT=$(($BASE_PORT + $i))
            if [ $i -eq $((NUM_GPUS - 1)) ]; then
                echo "    \"http://localhost:${PORT}\""
            else
                echo "    \"http://localhost:${PORT}\","
            fi
        done
        echo "  ]"
        echo ""
        echo "PIDs saved to: $PID_FILE"
        echo "Logs: server_${BASE_PORT}.log, server_$(($BASE_PORT + 1)).log, ..."
        echo ""
        echo "To stop all servers: ./stop_server.sh"
        echo "=========================================="
    else
        echo "=========================================="
        echo "⚠ Warning: $FAILED_SERVERS server(s) failed to start"
        echo "Check logs: server_*.log"
        echo "=========================================="
    fi
    
else
    # Single-server mode (original behavior)
    echo "Mode: Single-Server"
    PORT=$BASE_PORT
    echo "Port: $PORT"
    echo ""
    echo "Endpoints:"
    echo "  Health: http://localhost:$PORT/health"
    echo "  Analyze: http://localhost:$PORT/analyze"
    echo "  Batch: http://localhost:$PORT/analyze_batch"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "=========================================="
    echo ""
    
    # Start single server (foreground)
    python server.py
fi
