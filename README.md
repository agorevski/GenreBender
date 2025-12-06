# Genre Bener

A sophisticated, AI-powered system for automatically generating cinematic trailers from full-length movies, but with a twist -- the genre of the cinematic trailers is not the original!

Built with modular architecture supporting multiple genres and configurable processing pipelines.

## 🎬 Features

- **Intelligent Shot Detection**: Streaming shot detection with overlap handling for long movies (1-2 hours)
- **Multi-Frame Analysis**: Extracts 5 sample frames per shot for temporal understanding
- **Audio-Aware Processing**: MFCC and spectral feature extraction for audio analysis
- **Multimodal Analysis**: Remote Qwen2-VL integration for visual, temporal, and audio analysis
- **Genre-Configurable**: Support for thriller, action, drama, horror, sci-fi, comedy, and romance
- **LLM-Powered Narrative**: Azure OpenAI GPT-4 generates compelling trailer structures
- **Smart Caching**: Analysis result caching to avoid redundant processing
- **Batch Processing**: Memory-efficient processing of full-length movies
- **Chunked Generation**: Token-aware LLM processing for long shot lists

## 🏗️ Architecture

```text
trailer_generator/
├── ingest/              # Shot detection & keyframe extraction
├── analysis/            # Multimodal analysis & scoring
├── narrative/           # LLM-based timeline generation
├── editing/             # Video assembly & color grading (TODO)
├── audio/               # Music & SFX mixing (TODO)
├── config/              # YAML configuration files
└── utils/               # Shared utilities
```

## 📋 Prerequisites

- Python 3.9+
- FFmpeg installed and in PATH
- Access to Qwen2-VL server (or compatible multimodal API)
- Azure OpenAI API key (GPT-4)

## 🚀 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd trailer_generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

```bash
# Download from https://ffmpeg.org/download.html
# Add to PATH
```

### 4. Setup PySceneDetect

```bash
pip install scenedetect[opencv]
```

### 5. Configure Environment

Create `.env` file:
```bash
AZURE_OPENAI_KEY=your_api_key_here
```

### 6. Update Configuration

Edit `trailer_generator/config/settings.yaml`:
- Set your Qwen2-VL server URL
- Set your Azure OpenAI endpoint
- Adjust processing parameters

## 📖 Usage

### Basic Usage

```bash
python main.py input_movie.mp4 --genre thriller
```

### Advanced Options

```bash
python main.py input_movie.mp4 \
  --genre action \
  --output my_trailer.mp4 \
  --config custom_settings.yaml \
  --skip-analysis  # Use cached results
```

### Available Genres

- `thriller` - Suspenseful, building tension
- `action` - Fast-paced, high energy
- `drama` - Emotional, character-driven
- `horror` - Atmospheric, frightening
- `scifi` - Futuristic, wonder-filled
- `comedy` - Upbeat, humorous
- `romance` - Warm, emotional connection

### Command-Line Arguments

```
positional arguments:
  input                 Path to input video file

optional arguments:
  --genre {thriller,action,drama,horror,scifi,comedy,romance}
                        Trailer genre/style (default: thriller)
  --output OUTPUT       Output trailer path (default: output/trailer.mp4)
  --config CONFIG       Configuration file path
  --test                Run in test mode with sample validation
  --skip-analysis       Skip remote analysis (use cached results)
  --no-cache            Disable analysis caching
```

## 🔧 Configuration

### Main Settings (`settings.yaml`)

```yaml
processing:
  target_trailer_length: 90  # seconds
  shot_candidate_count: 60
  batch_size: 50

remote_analysis:
  qwen_server_url: "http://your-server:8000"
  cache_enabled: true

azure_openai:
  endpoint: "https://your-resource.openai.azure.com/"
  deployment_name: "gpt-4"
```

### Genre Profiles (`genre_profiles.yaml`)

Each genre defines:
- **Scoring weights**: Attribute importance (suspense, intensity, etc.)
- **Color grading**: FFmpeg filter chains
- **Music tags**: Recommended music styles
- **Pacing**: Editing rhythm
- **Text overlay style**: Font and animation

## 🎯 Pipeline Overview

### 1. Shot Detection
- Detects scene boundaries using PySceneDetect
- Processes in 30-second chunks with 5-second overlap
- Deduplicates shots across chunk boundaries

### 2. Multi-Frame Keyframe Extraction
- Extracts 5 evenly-spaced frames per shot for temporal analysis
- Provides QWEN with shot progression and motion context
- Configurable frame count and quality

### 3. Audio Feature Extraction
- Extracts audio segment for each shot time range
- Computes MFCC (Mel-Frequency Cepstral Coefficients) features
- Extracts spectral features (centroid, rolloff, bandwidth)
- Computes temporal features (zero-crossing rate, RMS energy)
- Extracts chroma features for musical content
- Optional tempo detection for longer segments
- Uses librosa for professional-grade audio analysis

### 4. Multimodal Analysis
- Sends multiple frames + audio features to Qwen2-VL server
- Analyzes temporal progression within each shot
- Integrates audio context for audio-aware scene understanding
- Receives visual description and genre attributes
- Caches results to avoid redundant API calls

### 5. Genre Scoring
- Computes weighted scores based on genre profile
- Sorts shots by relevance to target genre

### 6. Shot Selection
- Selects top N candidates
- Optional temporal diversity filtering

### 7. Narrative Generation
- Azure OpenAI GPT-4 creates trailer structure
- Chunked processing for long shot lists
- Returns timeline with shot order and durations

### 8. Video Assembly (TODO)
- Concatenate shots according to timeline
- Apply genre-specific color grading
- Add text overlays and transitions

### 9. Audio Mixing (TODO)
- Select music from library or generate AI audio
- Mix sound effects
- Sync with video timeline

## 📊 Output Files

```
output/
  ├── trailer.mp4           # Final rendered trailer (when complete)
  └── timeline.json         # Timeline structure

shots/
  ├── shot_0001.mp4         # Individual shot segments
  ├── shot_0002.mp4
  └── shot_metadata.json    # Shot timing and metadata

keyframes/
  ├── kf_0001_1.jpg         # Extracted keyframes (5 per shot)
  ├── kf_0001_2.jpg
  ├── kf_0001_3.jpg
  ├── kf_0001_4.jpg
  ├── kf_0001_5.jpg
  └── kf_0002_1.jpg

cache/
  └── analysis_cache.json   # Cached analysis results

temp/
  └── partial_analysis.json # Resume-capable partial results
```

## 🧪 Testing

Start with short test clips (5-10 minutes) before processing full movies:

```bash
python main.py test_clip.mp4 --genre thriller --test
```

## 🔌 Qwen2-VL Server Setup

Your Qwen2-VL server should expose these endpoints:

### Health Check
```
GET /health
```

### Single Analysis
```
POST /analyze
{
  "shot_id": 1,
  "images": [
    "base64_encoded_image_1",
    "base64_encoded_image_2",
    "base64_encoded_image_3",
    "base64_encoded_image_4",
    "base64_encoded_image_5"
  ],
  "audio_features": {
    "mfcc_mean": [13 MFCC coefficients],
    "mfcc_std": [13 MFCC std deviations],
    "spectral_centroid_mean": 2500.5,
    "spectral_rolloff_mean": 4800.2,
    "zero_crossing_rate_mean": 0.12,
    "rms_energy_mean": 0.045,
    "chroma_mean": [12 chroma features],
    "tempo": 120.5
  },
  "start_time": 10.5,
  "end_time": 12.8,
  "duration": 2.3
}
```

### Batch Analysis
```
POST /analyze_batch
{
  "shots": [
    {
      "shot_id": 1,
      "images": ["base64_img1", "base64_img2", ...],
      "audio_features": { ... },
      "start_time": 10.5,
      "end_time": 12.8,
      "duration": 2.3
    },
    ...
  ]
}
```

**Note**: The server should process the temporal sequence of frames and integrate audio features into its multimodal analysis. Audio features can help identify:
- Music cues and intensity
- Dialogue vs silence
- Sound effects and ambiance
- Overall audio energy and mood

**Expected Response:**
```json
{
  "caption": "A person walking in a dark hallway",
  "attributes": {
    "suspense": 0.78,
    "darkness": 0.65,
    "ambiguity": 0.71,
    "emotional_tension": 0.59
  }
}
```

## 🐛 Troubleshooting

### FFmpeg Not Found
```bash
# Verify FFmpeg installation
ffmpeg -version

# Add to PATH if needed (Windows)
setx PATH "%PATH%;C:\path\to\ffmpeg\bin"
```

### Librosa Installation Issues
```bash
# Install audio processing dependencies
pip install librosa soundfile

# On Windows, you may need Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# On Linux/Mac, you may need system audio libraries
# Ubuntu/Debian: sudo apt-get install libsndfile1
# macOS: brew install libsndfile
```

### Audio Extraction Disabled
If librosa is not installed, the system will skip audio extraction and continue with visual-only analysis. Install librosa to enable audio-aware processing.

### PySceneDetect Installation Issues
```bash
# Install with OpenCV backend
pip install scenedetect[opencv] --upgrade
```

### Qwen2-VL Connection Failed
- Verify server URL in `settings.yaml`
- Check server is running: `curl http://your-server:8000/health`
- Review firewall/network settings

### Azure OpenAI Authentication Failed
- Verify API key in `.env` file
- Check endpoint URL format
- Confirm deployment name matches your Azure resource

### Out of Memory
- Reduce `batch_size` in settings
- Process shorter test clips first
- Ensure streaming mode is enabled

## 📝 Current Implementation Status

### ✅ Completed
- Shot detection with streaming and overlap
- Multi-frame keyframe extraction (5 frames per shot)
- Audio feature extraction (MFCC, spectral, temporal)
- Remote multimodal analysis client with multi-frame + audio support
- Analysis caching system
- Genre-configurable scoring
- Azure OpenAI integration
- LLM-powered timeline generation
- Main orchestrator with CLI

### 🚧 In Progress
- Video assembly and editing pipeline
- Color grading implementation
- Audio mixing system
- Text overlay rendering

### 📅 Planned
- Web UI dashboard
- Real-time progress monitoring
- Advanced shot transition effects
- AI-powered audio generation
- Multi-language support

## 🤝 Contributing

This is a modular system designed for extension. Key integration points:

1. **New Genres**: Add to `genre_profiles.yaml`
2. **Analysis Backends**: Implement `RemoteAnalyzer` interface
3. **LLM Providers**: Extend `AzureOpenAIClient` base
4. **Video Effects**: Add to editing pipeline modules

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- PySceneDetect for shot detection
- OpenCV for video processing
- librosa for audio analysis
- Azure OpenAI for narrative generation
- MoviePy for video editing capabilities

## 📧 Contact

[Your Contact Information]

---

**Note**: Video assembly and audio mixing modules are under development. The current implementation successfully generates trailer timelines that can be used with external video editing tools.
