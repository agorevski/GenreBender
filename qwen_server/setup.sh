#!/bin/bash
# Setup script for QWEN server
# Downloads model, installs dependencies, and verifies installation

set -e  # Exit on error

echo "=========================================="
echo "QWEN Server Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "Error: Python 3.9+ required"
    exit 1
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: NVIDIA GPU not detected. Server will run on CPU (slower)."
fi

# Create virtual environment (optional but recommended)
echo ""
read -p "Create virtual environment? (recommended) [Y/n]: " create_venv
create_venv=${create_venv:-Y}

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_qwen_server
    source venv_qwen_server/bin/activate
    echo "Virtual environment activated"
fi


# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (with CUDA support if available)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision
fi

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download model
echo ""
echo "=========================================="
echo "Downloading Qwen VL Model"
echo "=========================================="
echo "This will download the model weights..."
echo "The model will be cached in ~/.cache/huggingface/"
echo ""

python3 << 'PYTHON'
import yaml
import logging
from model_loader import download_model, detect_model_family, MODEL_FAMILY_AUTO

logging.basicConfig(level=logging.INFO)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
cache_dir = config['model']['cache_dir']
model_family = config['model'].get('model_family', MODEL_FAMILY_AUTO)

# Detect model family for display
if model_family == MODEL_FAMILY_AUTO:
    detected_family = detect_model_family(model_name)
    print(f"Model: {model_name}")
    print(f"Detected family: {detected_family}")
else:
    print(f"Model: {model_name}")
    print(f"Configured family: {model_family}")

print(f"\nDownloading {model_name}...")
success = download_model(model_name, cache_dir, model_family)

if success:
    print("\n✓ Model downloaded successfully!")
else:
    print("\n✗ Model download failed")
    exit(1)
PYTHON

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

python3 << 'PYTHON'
import yaml
import logging
import torch
from model_loader import ModelLoader, detect_model_family, MODEL_FAMILY_AUTO

logging.basicConfig(level=logging.INFO)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Show model family info
model_name = config['model']['name']
model_family = config['model'].get('model_family', MODEL_FAMILY_AUTO)
if model_family == MODEL_FAMILY_AUTO:
    detected_family = detect_model_family(model_name)
    print(f"\nModel: {model_name}")
    print(f"Detected family: {detected_family}")
else:
    print(f"\nModel: {model_name}")
    print(f"Configured family: {model_family}")

# Try loading model
print("\nAttempting to load model...")
loader = ModelLoader(config)
success = loader.verify_installation()

if success:
    print("\n✓ Model verification successful!")
    print("\nModel info:")
    info = loader.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
else:
    print("\n✗ Model verification failed")
    exit(1)
PYTHON

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start the server, run:"
echo "  ./start_server.sh"
echo ""
echo "API key: helloagorevski"
echo ""
