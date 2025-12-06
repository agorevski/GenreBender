# QWEN Server - Multimodal Shot Analysis

FastAPI server for analyzing video shots using Qwen2-VL with multi-frame temporal analysis and audio feature integration.

## Features

- **Qwen2-VL Integration**: Automatic model download from HuggingFace
- **Multi-Frame Analysis**: Processes 5 frames per shot for temporal understanding
- **Audio-Aware**: Integrates MFCC and spectral audio features
- **GPU Acceleration**: Automatic multi-GPU support (uses all 4 GPUs)
- **API Authentication**: Bearer token authentication with hardcoded key
- **Batch Processing**: Efficient batch endpoint for multiple shots
- **Health Monitoring**: Health check endpoint with GPU status

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended, supports 4x GPUs)
- ~4GB disk space for model
- ~8GB GPU memory for Qwen2-VL-2B

## Quick Start

### 1. Setup (First Time Only)

```bash
cd qwen_server
chmod +x *.sh
./setup.sh
```

This will:
- Check Python and CUDA availability
- Create virtual environment (optional)
- Install PyTorch with CUDA support
- Install dependencies
- Download Qwen2-VL-2B model (~4GB)
- Verify installation

### 2. Start Server

```bash
./start_server.sh
```

Server will start on `http://0.0.0.0:8000`

### 3. Test Server

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model": "Qwen/Qwen2-VL-2B-Instruct",
#   "device": "cuda",
#   "gpu_count": 4,
#   "gpu_memory_total": "80.0 GB"
# }
```

### 4. Stop Server

```bash
./stop_server.sh
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
model:
  name: "Qwen/Qwen2-VL-2B-Instruct"  # Switch to 7B if needed
  device: "cuda"                      # cuda or cpu
  dtype: "float16"                    # float16 or float32
  
server:
  host: "0.0.0.0"
  port: 8000
  api_key: "helloagorevski"           # Authentication key
  
processing:
  max_batch_size: 10
  enable_audio_fusion: true
  temporal_weight: 0.3
  audio_weight: 0.2
```

## API Endpoints

### Health Check

**GET** `/health`

No authentication required.

```bash
curl http://localhost:8000/health
```

### Single Shot Analysis

**POST** `/analyze`

Requires: `Authorization: Bearer helloagorevski`

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Authorization: Bearer helloagorevski" \
  -H "Content-Type: application/json" \
  -d '{
    "shot_id": 1,
    "images": ["base64_image_1", "base64_image_2", ...],
    "audio_features": {
      "mfcc_mean": [...],
      "spectral_centroid_mean": 2500.5,
      "rms_energy_mean": 0.045,
      ...
    },
    "start_time": 10.5,
    "end_time": 12.8,
    "duration": 2.3
  }'
```

**Response:**
```json
{
  "caption": "A person walking in a dark hallway",
  "attributes": {
    "suspense": 0.78,
    "darkness": 0.65,
    "ambiguity": 0.71,
    "emotional_tension": 0.59,
    "intensity": 0.82,
    "motion": 0.45
  }
}
```

### Batch Analysis

**POST** `/analyze_batch`

Requires: `Authorization: Bearer helloagorevski`

```bash
curl -X POST http://localhost:8000/analyze_batch \
  -H "Authorization: Bearer helloagorevski" \
  -H "Content-Type: application/json" \
  -d '{
    "shots": [
      {
        "shot_id": 1,
        "images": [...],
        "audio_features": {...},
        "start_time": 10.5,
        "end_time": 12.8,
        "duration": 2.3
      },
      ...
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "shot_id": 1,
      "caption": "...",
      "attributes": {...}
    },
    ...
  ]
}
```

## Authentication

All analysis endpoints require Bearer token authentication:

```
Authorization: Bearer helloagorevski
```

Invalid or missing tokens will return `401 Unauthorized`.

## Multi-Frame Processing

The server processes temporal sequences:

1. Receives 5 evenly-spaced frames per shot
2. Analyzes key frames (first, middle, last)
3. Aggregates temporal information
4. Detects motion from frame variance
5. Weights middle frames more heavily

## Audio Integration

When audio features are provided:

- **Energy-based classification**: Silent, dialog, music, ambiance
- **Attribute enhancement**: Boosts suspense, darkness, or intensity
- **Multimodal fusion**: Combines visual and audio cues

Audio features include:
- MFCC coefficients (13 mean + std)
- Spectral features (centroid, rolloff, bandwidth)
- Temporal features (zero-crossing rate, RMS energy)
- Chroma features (12 dimensions)
- Tempo (when available)

## GPU Usage

The server automatically:
- Detects all available GPUs
- Uses `device_map="auto"` for model sharding
- Distributes computation across GPUs
- Falls back to CPU if no GPUs available

Monitor GPU usage:
```bash
nvidia-smi -l 1  # Update every second
```

## Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Check logs
cat qwen_server.log
```

### Out of Memory

Edit `config.yaml`:
```yaml
model:
  dtype: "float32"  # Use full precision (uses more memory but may help)
```

Or reduce batch size:
```yaml
processing:
  max_batch_size: 5  # Reduce from 10
```

### CUDA Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Download Failed

```bash
# Manually download with Python
python3 -c "
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', cache_dir='~/.cache/huggingface')
Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', cache_dir='~/.cache/huggingface')
"
```

## Performance

Expected performance with 4x GPUs:

- Model loading: ~30 seconds
- Single shot analysis: ~1-2 seconds
- Batch (10 shots): ~8-15 seconds
- Typical movie (500-1000 shots): ~8-20 minutes

## Switching Models

To use the larger 7B model, edit `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2-VL-7B-Instruct"  # ~14GB model
```

Then re-run setup to download:
```bash
./setup.sh
```

## Development

Run in development mode with auto-reload:

```yaml
# config.yaml
server:
  reload: true
```

View interactive API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Files

```
qwen_server/
├── server.py          # FastAPI server
├── model_loader.py    # Model download/loading
├── analyzer.py        # Multimodal analysis
├── config.yaml        # Configuration
├── requirements.txt   # Python dependencies
├── setup.sh          # Setup script
├── start_server.sh   # Start server
├── stop_server.sh    # Stop server
└── README.md         # This file
```

## License

Part of the RuinedMedia trailer generator project.

## Support

For issues or questions, refer to:
- Main project README: `../README.md`
- API changes doc: `../QWEN_SERVER_API_CHANGES.md`
- Qwen2-VL documentation: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
