# GenreBender

A sophisticated, AI-powered system for automatically generating cinematic trailers from full-length movies in **multiple genres simultaneously**!

Built with modular architecture supporting 12 genres and parallel processing pipelines.

## 🎬 Overview

GenreBender transforms full-length movies into compelling trailers by:
1. Analyzing visual, temporal, and audio content using multimodal AI
2. Generating semantic story graphs and genre-specific beat sheets
3. Matching scenes to trailer beats using embedding similarity
4. Assembling video with genre-specific color grading and transitions
5. Mixing audio with music and effects

The system processes movies through a multi-phase pipeline that can generate **multiple genre versions in parallel**.

## 🏗️ Pipeline Architecture

### Phase 1: Genre-Agnostic (Run Once)
| Stage | Function | Technology |
|-------|----------|------------|
| 1. Shot Detection | Identifies scene boundaries | PySceneDetect |
| 2. Keyframe Extraction | Extracts 5 frames per shot | FFmpeg |
| 3. Audio Extraction | Analyzes MFCC, spectral features | librosa |
| 4. Subtitle Management | Parses SRT files | pysrt |
| 5. Multimodal Analysis | Visual + audio understanding | Qwen2-VL |
| 11. Story Graph | Semantic narrative structure | Azure OpenAI GPT-4 |

### Phase 2: Genre-Dependent (Run Per Genre, Parallelizable)
| Stage | Function | Technology |
|-------|----------|------------|
| 12. Beat Sheet | Genre-specific trailer beats | GPT-4 |
| 13. Embeddings | Scene & beat vector embeddings | Azure OpenAI |
| 14. Scene Retrieval | Semantic beat-to-scene matching | FAISS |
| 15. Timeline | Deterministic shot timeline | Custom algorithm |
| 9. Video Assembly | Color grading & transitions | FFmpeg |
| 10. Audio Mixing | Music with ducking | FFmpeg |

## 🎭 Supported Genres (12)

| Genre | Description |
|-------|-------------|
| `comedy` | Upbeat, humorous |
| `horror` | Atmospheric, frightening |
| `thriller` | Suspenseful, building tension |
| `parody` | Over-the-top, comedic exaggeration |
| `mockumentary` | Documentary-style, deadpan humor |
| `crime` | Noir-inspired, gritty investigation |
| `drama` | Emotional, character-driven |
| `experimental` | Surrealist, unconventional |
| `fantasy` | Magical, epic adventure |
| `romance` | Warm, emotional connection |
| `scifi` | Futuristic, wonder-filled |
| `action` | Fast-paced, high energy |

## 📋 Prerequisites

- Python 3.9+
- FFmpeg (in PATH)
- Qwen2-VL server (or compatible multimodal API)
- Azure OpenAI API key (GPT-4)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```bash
AZURE_OPENAI_KEY=your_api_key_here
```

Update `trailer_generator/config/settings.yaml`:
```yaml
remote_analysis:
  qwen_server_url: "http://your-server:8000"

azure_openai:
  endpoint: "https://your-resource.openai.azure.com/"
  deployment_name: "gpt-4"
```

### 3. Configure Movie in config.yaml

```yaml
movies:
  hitch:
    movie_name: "Hitch"
    video: "./test_files/hitch.mp4"
    subtitles: "./test_files/hitch.srt"
    synopsis: "./test_files/hitch_synopsis.txt"
    genres:
      - comedy
      - thriller
      - romance
      - drama
```

### 4. Run Multi-Genre Pipeline

```bash
python run_multi_genre_pipeline.py hitch
```

## 📖 Usage

### Multi-Genre Pipeline (Recommended)

```bash
# Generate trailers for all genres defined in config.yaml
python run_multi_genre_pipeline.py hitch

# Specify parallel workers (default: 4)
python run_multi_genre_pipeline.py hitch --parallel-workers 4

# Skip Phase 1 if genre-agnostic stages already complete
python run_multi_genre_pipeline.py hitch --skip-phase1

# Override genres from config.yaml
python run_multi_genre_pipeline.py hitch --genres comedy,thriller,horror

# Force re-run all stages
python run_multi_genre_pipeline.py hitch --force
```

### Stage-by-Stage Execution

**Genre-Agnostic Stages (no --genre required):**
```bash
python 1_shot_detection.py --input movie.mp4
python 2_keyframe_extraction.py --input movie.mp4
python 3_audio_extraction.py --input movie.mp4
python 4_subtitle_management.py --input movie.mp4 --srt-file movie.srt
python 5_remote_analysis.py --input movie.mp4
python 11_story_graph_generator.py --movie-name "Movie" --synopsis synopsis.txt --srt-file movie.srt
```

**Genre-Dependent Stages (--genre required):**
```bash
python 12_beat_sheet_generator.py --movie-name "Movie" --genre thriller
python 13_embedding_generator.py --input movie.mp4 --genre thriller --movie-name "Movie"
python 14_scene_retrieval.py --input movie.mp4 --genre thriller --movie-name "Movie"
python 15_timeline_constructor.py --input movie.mp4 --genre thriller
python 9_video_assembly.py --input movie.mp4 --genre thriller
python 10_audio_mixing.py --input movie.mp4 --genre thriller
```

### Key Options

```bash
--input PATH              Input video file (required for stage scripts)
--genre GENRE            Target trailer genre (required for genre-dependent stages)
--force                  Force re-run even if completed
--test                   Process only first 5 shots for testing
--parallel-workers N     Number of parallel genre workers (default: 4)
--skip-phase1           Skip genre-agnostic stages
```

## 📊 Output Structure

```
outputs/<sanitized_filename>/
├── shots/
│   ├── shot_0001.mp4
│   └── shot_metadata.json
├── keyframes/
│   ├── kf_0001_1.jpg (5 frames per shot)
│   └── ...
├── cache/
│   └── analysis_cache.json
├── embeddings/
│   ├── scene_embeddings.pkl
│   └── beat_embeddings.pkl
├── trailers/
│   ├── comedy/
│   │   ├── trailer_comedy_assembled.mp4
│   │   └── trailer_comedy_final.mp4      # FINAL
│   ├── thriller/
│   │   └── trailer_thriller_final.mp4    # FINAL
│   └── <other_genres>/
├── checkpoint.json          # Per-genre progress tracking
└── trailer_generator.log

outputs/story_graphs/<movie_name>/
├── story_graph.json         # Semantic narrative structure
├── beats.json              # Genre-specific beat sheets
└── genre_rewrite.json      # Genre transformation
```

## 🔧 Configuration

### config.yaml (Pipeline Configuration)

```yaml
settings:
  parallel_workers: 4  # Number of genres to process in parallel

movies:
  hitch:
    movie_name: "Hitch"
    video: "./test_files/hitch.mp4"
    subtitles: "./test_files/hitch.srt"
    synopsis: "./test_files/hitch_synopsis.txt"
    genres:
      - comedy
      - thriller
      - horror
      - romance
```

### Genre Profiles (`genre_profiles.yaml`)

Each genre defines:
- **Scoring weights**: Attribute importance (suspense, intensity, etc.)
- **Color grading**: FFmpeg filter specifications
- **Music tags**: Recommended audio styles
- **Pacing**: Shot timing preferences

### Settings (`settings.yaml`)

```yaml
processing:
  target_trailer_length: 90
  shot_candidate_count: 60

remote_analysis:
  qwen_server_url: "http://..."
  batch_size: 10

azure_openai:
  temperature: 0.7
  max_completion_tokens: 4000
```

## 🔌 Qwen2-VL Server

### Setup

```bash
cd qwen_server
./setup.sh        # First-time setup
./start_server.sh # Start server
./stop_server.sh  # Stop server
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🐛 Troubleshooting

### Common Issues

1. **"Qwen2-VL server not responding"**: 
   - Check server URL in settings.yaml
   - Verify server: `curl http://localhost:8000/health`

2. **"FFmpeg not found"**: Ensure FFmpeg is in PATH

3. **Out of memory**: Reduce `batch_size` in settings

4. **Genre stage already completed**: Use `--force` flag

### Debug Commands

```bash
# Check pipeline state
cat outputs/<filename>/checkpoint.json | jq

# View logs
tail -n 50 outputs/<filename>/trailer_generator.log

# Check Qwen server
curl http://localhost:8000/health
```

## 📝 Checkpoint System

The checkpoint system tracks progress separately for:
- **Genre-agnostic stages**: Run once, shared across genres
- **Genre-dependent stages**: Tracked per-genre for parallel processing

```json
{
  "completed_stages": ["shot_detection", "keyframe_extraction", ...],
  "genre_stages": {
    "comedy": ["beat_sheet_generation", "video_assembly", ...],
    "thriller": ["beat_sheet_generation", ...]
  }
}
```

## 🎵 Audio Assets

Place music files in `audio_assets/music/` with genre keywords:
```
thriller_suspense_01.mp3
action_epic_music.wav
horror_atmospheric.mp3
```

See `audio_assets/README.md` for detailed setup.

## 🤝 Contributing

Key extension points:
1. **New Genres**: Add profiles to `genre_profiles.yaml`
2. **Analysis Backends**: Implement `RemoteAnalyzer` interface
3. **LLM Providers**: Extend `AzureOpenAIClient`

## 📚 Additional Documentation

- `audio_assets/README.md` - Music library setup
- `utilities/README.md` - OMDB API utilities
- `qwen_server/README.md` - Qwen2-VL server setup

---

**Final Output**: The system generates broadcast-ready trailers for each genre at `outputs/<video>/trailers/<genre>/trailer_<genre>_final.mp4`
