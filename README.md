# GenreBender

An AI-assisted command-line tool for creating genre-reimagined trailer drafts from full-length movies in **multiple genres simultaneously**.

Built with modular architecture supporting **27 genres** and parallel processing pipelines.

## 🎬 Overview

GenreBender transforms full-length movies into compelling trailers by:
1. Analyzing visual, temporal, and audio content using multimodal AI
2. Generating semantic story graphs and genre-specific beat sheets
3. Matching scenes to trailer beats using embedding similarity and FAISS
4. Assembling video with genre-specific color grading and transitions
5. Mixing audio with music and effects

The system processes movies through a multi-phase pipeline that can generate **multiple genre versions in parallel**.

GenreBender is a local Python pipeline, not a hosted editor. You supply a movie,
matching SRT subtitles, a synopsis, and access to the configured AI services.
Generated trailers should be reviewed for pacing, dialogue, spoilers, and
appropriate use of the source material before publication. Only process media
you have permission to use; the configured analysis and Azure OpenAI services
receive movie-derived frames, audio features, subtitles, or story text.

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
| 13. Embeddings | Scene & beat vector embeddings | Azure OpenAI text-embedding-ada-002 |
| 14. Scene Retrieval | Semantic beat-to-scene matching | FAISS |
| 15. Timeline | Deterministic shot timeline | Custom algorithm |
| 9. Video Assembly | Color grading & transitions | FFmpeg |
| 10. Audio Mixing | Music with ducking | FFmpeg |

## 🎭 Supported Genres (27)

### Original Genres (12)
| Genre | Description |
|-------|-------------|
| `thriller` | Suspenseful, building tension |
| `action` | Fast-paced, high energy |
| `drama` | Emotional, character-driven |
| `horror` | Atmospheric, frightening |
| `scifi` | Futuristic, wonder-filled |
| `comedy` | Upbeat, humorous |
| `romance` | Warm, emotional connection |
| `parody` | Over-the-top, comedic exaggeration |
| `mockumentary` | Documentary-style, deadpan humor |
| `crime` | Noir-inspired, gritty investigation |
| `experimental` | Surrealist, unconventional |
| `fantasy` | Magical, epic adventure |

### Major Traditional Genres (8)
| Genre | Description |
|-------|-------------|
| `western` | Rugged, expansive landscapes |
| `war` | Intense, heroic sacrifice |
| `musical` | Theatrical, showstopping performances |
| `documentary` | Authentic, observational storytelling |
| `sports` | Triumphant, underdog journeys |
| `mystery` | Intriguing, clue-driven reveals |
| `historical` | Grand, period-authentic epics |
| `biographical` | Inspiring, personal life journeys |

### Emerging/Modern Genres (7)
| Genre | Description |
|-------|-------------|
| `superhero` | Heroic, powerful, epic triumphs |
| `dystopian` | Bleak, rebellious, industrial |
| `found_footage` | Raw, authentic, immediate terror |
| `kaiju` | Massive scale, monster destruction |
| `cyberpunk` | Neon-soaked, gritty futurism |
| `mumblecore` | Intimate, naturalistic indie |
| `kdrama` | Emotional, romantic melodrama |

## 📋 Prerequisites

- Python 3.9+
- FFmpeg (in PATH)
- Qwen2-VL server (or compatible multimodal API)
- Azure OpenAI API key (GPT-4 + text-embedding-ada-002)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```bash
AZURE_OPENAI_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

Update `trailer_generator/config/settings.yaml`:
```yaml
remote_analysis:
  server_host: "localhost"
  server_base_port: 8000
  server_count: 4  # Number of Qwen servers

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
# Preview the plan and identify local setup issues before starting
python run_multi_genre_pipeline.py hitch --dry-run

# Generate the configured trailers
python run_multi_genre_pipeline.py hitch
```

The example movie files are not included; replace their paths with your own
files. Relative input paths are resolved from the pipeline `config.yaml` file.
Stage scripts, runtime settings, assets, and outputs are resolved from the
repository root, even when you invoke the orchestrator from another directory.

## 📖 Usage

### Multi-Genre Pipeline (Recommended)

```bash
# Generate trailers for all genres defined in config.yaml
python run_multi_genre_pipeline.py hitch

# Specify parallel workers (default: settings.parallel_workers, otherwise 4)
python run_multi_genre_pipeline.py hitch --parallel-workers 4

# Run sequentially with full output streaming (for debugging)
python run_multi_genre_pipeline.py hitch --sequential

# Skip Phase 1 if genre-agnostic stages already complete
python run_multi_genre_pipeline.py hitch --skip-phase1

# Override genres from config.yaml
python run_multi_genre_pipeline.py hitch --genres comedy,thriller,horror

# Force re-run all stages
python run_multi_genre_pipeline.py hitch --force

# Preview a different pipeline configuration without processing or API calls
python run_multi_genre_pipeline.py my_movie --config my_movies.yaml --dry-run
```

`--dry-run` prints the selected stages, exact commands, expected trailer paths,
and local readiness issues without running media tools, making API calls, or
writing outputs. Input/configuration errors exit with code **2**. A valid preview
exits **0**, even if readiness warnings are present; it is not a connectivity
test or a guarantee that generation will succeed.

Generation stops before expensive work if FFmpeg, FFprobe, required Azure
settings, or requested resume artifacts are missing. Shared preprocessing must
succeed before any genre starts. Once genre processing begins, a failed genre
does not cancel the others, including in sequential mode. The summary reports
each result, verifies that successful final files exist and are non-empty, and
exits **1** if any genre fails. Retry only failed genres using, for example:

```bash
python run_multi_genre_pipeline.py hitch --skip-phase1 --genres horror
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
--movie-name NAME        Movie name for story graph lookup
--force                  Force re-run even if completed
--test                   Process only first 5 shots for testing
--parallel-workers N     Positive worker count (configured value, otherwise 4)
--skip-phase1           Skip genre-agnostic stages
--sequential            Run Phase 2 sequentially with full output streaming
--verbose               Enable verbose logging
```

The orchestrator also accepts `--dry-run` and `--config PATH`. Stage-specific
options such as `--input`, `--test`, and `--verbose` are not orchestrator flags.

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
├── trailers/
│   ├── comedy/
│   │   ├── embeddings/
│   │   │   ├── scene_embeddings.pkl
│   │   │   └── beat_embeddings.pkl
│   │   ├── selected_scenes.json
│   │   ├── trailer_timeline.json
│   │   ├── temp/
│   │   ├── trailer_comedy_assembled.mp4
│   │   └── trailer_comedy_final.mp4      # FINAL
│   ├── thriller/
│   │   └── trailer_thriller_final.mp4    # FINAL
│   └── <other_genres>/
├── checkpoint.json          # Per-genre progress tracking
└── trailer_generator.log

outputs/story_graphs/<movie_name>/
├── story_graph.json         # Semantic narrative structure
├── beats_<genre>.json       # Genre-specific beat sheets
└── genre_rewrite_<genre>.json  # Genre transformation
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

### Genre Profiles (`trailer_generator/config/genre_profiles.yaml`)

Each genre defines:
- **Scoring weights**: Attribute importance (38 total attributes)
- **Color grading**: FFmpeg filter specifications
- **Music tags**: Recommended audio styles
- **Music generation**: Instruments and mood descriptors per section
- **Pacing**: Shot timing preferences
- **Text overlay style**: Font, color, animation

This file is the source of truth for the supported genre names across the
orchestrator and stage CLIs. Names are case-insensitive; unknown names are
rejected rather than silently using a thriller profile.

Timeline construction keeps source ranges valid and avoids repeated shots.
When the selected footage cannot fill the requested duration, the timeline
records the actual duration, represented beats, and shortfall instead of
inventing gaps. Video assembly uses the selected source offsets, supplies
silence for clips without audio, and keeps transitions within available source
handles; boundaries without enough footage remain cuts.

Title overlay rendering is not yet implemented. AI title-text generation is
disabled by default to avoid paying for text that is not shown in the export.

### Settings (`trailer_generator/config/settings.yaml`)

```yaml
processing:
  target_trailer_length: 90
  shot_candidate_count: 60

remote_analysis:
  server_host: "localhost"
  server_base_port: 8000
  server_count: 4  # Multi-server support
  batch_size: 16

azure_openai:
  temperature: 0.7
  max_completion_tokens: 50000

embedding:
  model: "text-embedding-ada-002"
  batch_size: 20
  parallel_workers: 20

retrieval:
  top_k: 10
  scoring_weights:
    semantic_similarity: 0.50
    emotional_alignment: 0.25
    visual_match: 0.20
    original_genre_penalty: 0.05

timeline:
  target_duration: 90
```

## 🔌 Qwen2-VL Server

### Setup

```bash
cd qwen_server
./setup.sh        # First-time setup
./start_server.sh # Start server
./stop_server.sh  # Stop server
```

### Multi-Server Support

The pipeline supports multiple Qwen servers for parallel processing:

```yaml
remote_analysis:
  server_host: "localhost"
  server_base_port: 8000
  server_count: 4  # Servers on ports 8000, 8001, 8002, 8003
  load_balancing: "round_robin"
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🎬 OMDB Integration

GenreBender includes OMDB API utilities for fetching movie metadata:

```python
from utilities.omdb_client import OMDBClient

client = OMDBClient()
movie = client.get_movie_by_title("Hitch")
print(f"Genre: {movie.genre}, Rating: {movie.imdb_rating}")
```

Features:
- Smart caching (30-day TTL)
- Structured data models
- Multiple search methods

See `utilities/README.md` for full documentation.

## 🐛 Troubleshooting

### Common Issues

1. **"Qwen2-VL server not responding"**: 
   - Check server URL in settings.yaml
   - Verify server: `curl http://localhost:8000/health`

2. **"FFmpeg not found"**: Ensure FFmpeg is in PATH

3. **Out of memory**: Reduce `batch_size` in settings

4. **Stage already completed**: Use `--force` flag

5. **"Embeddings not found"**: Run stage 13 first

6. **"Beat sheet not found"**: Run stage 12 first with matching genre

### Debug Commands

```bash
# Check pipeline state
cat outputs/<filename>/checkpoint.json | jq

# View logs
tail -n 50 outputs/<filename>/trailer_generator.log

# Check Qwen server
curl http://localhost:8000/health

# Check embeddings for one genre
ls -la outputs/<filename>/trailers/comedy/embeddings/
```

## 📝 Checkpoint System

The checkpoint system tracks progress separately for:
- **Genre-agnostic stages**: Run once, shared across genres
- **Genre-dependent stages**: Tracked per-genre for parallel processing

```json
{
  "version": "2.0",
  "stages": {
    "shot_detection": {"completed": true},
    "keyframe_extraction": {"completed": true}
  },
  "genre_stages": {
    "comedy": {
      "embedding_generation": {"completed": true},
      "scene_retrieval": {"completed": true}
    },
    "thriller": {
      "embedding_generation": {"completed": true}
    }
  }
}
```

Workers update checkpoints under a shared file lock and replace the JSON
atomically, so concurrent genres do not overwrite each other's progress.
Keep the `checkpoint.json.lock` sidecar in place, especially while workers
are running. A malformed checkpoint now stops execution instead of silently
discarding progress: stop active workers, back up the file, then repair it or
move it aside deliberately before restarting.

## 🎵 Audio Assets

Place music files in `audio_assets/music/` with genre keywords:
```
thriller_suspense_01.mp3
action_epic_music.wav
horror_atmospheric.mp3
```

See `audio_assets/README.md` for detailed setup.

Audio finalization uses each genre's own timeline, assembled video, and temporary
directory. When music is available, short tracks loop to the constructed trailer
duration; dialogue remains audible when ducking lowers the music. Missing music
continues to use the assembled video's original audio.

## 🤝 Contributing

Key extension points:
1. **New Genres**: Add profiles to `genre_profiles.yaml` (27 examples included)
2. **Analysis Backends**: Implement `RemoteAnalyzer` interface
3. **LLM Providers**: Extend `AzureOpenAIClient`
4. **Retrieval Algorithms**: Modify `scene_retriever.py`

Run the focused product regressions without contacting AI services:

```bash
python -m pytest -q test_pipeline_orchestrator.py test_checkpoint.py test_genre_profiles.py test_audio_pipeline.py test_trailer_quality.py
```

Synthetic rendering cases run when FFmpeg and FFprobe are on PATH; otherwise
those media cases are skipped. The remaining cases use local fixtures and mocks,
including independent-process checkpoint updates.

## 📚 Additional Documentation

- `audio_assets/README.md` - Music library setup
- `utilities/README.md` - OMDB API utilities
- `qwen_server/README.md` - Qwen2-VL server setup

---

**Final Output**: Reviewable trailer drafts for each genre are saved at `outputs/<video>/trailers/<genre>/trailer_<genre>_final.mp4`.
