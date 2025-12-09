# GenreBender

A sophisticated, AI-powered system for automatically generating cinematic trailers from full-length movies with a twist — the trailer genre doesn't match the original film!

Built with modular architecture supporting multiple genres and configurable processing pipelines.

## 🎬 Overview

GenreBender transforms full-length movies into compelling trailers by:
1. Analyzing visual, temporal, and audio content using multimodal AI
2. Scoring shots based on target genre characteristics
3. Generating narrative structure using GPT-4
4. Assembling video with transitions and color grading
5. Mixing audio with music and effects

The system processes movies through a complete 10-stage pipeline, with intelligent caching and checkpoint support for resumable processing.

## 🏗️ Pipeline Stages

| Stage | Function | Technology |
|-------|----------|------------|
| 1. Shot Detection | Identifies scene boundaries in video | PySceneDetect |
| 2. Keyframe Extraction | Extracts 5 frames per shot for temporal analysis | FFmpeg |
| 3. Audio Extraction | Analyzes audio features (MFCC, spectral, temporal) | librosa |
| 4. Subtitle Management | Parses SRT files and maps dialogue to shots | pysrt |
| 5. Multimodal Analysis | Understands visual + audio content | Qwen2-VL |
| 6. Genre Scoring | Scores shots based on target genre profile | Custom algorithm |
| 7. Shot Selection | Selects top candidates for trailer | Ranking system |
| 8. Narrative Generation | Creates compelling trailer structure | Azure OpenAI GPT-4 |
| 9. Video Assembly | Assembles video with color grading & transitions | FFmpeg |
| 10. Audio Mixing | Mixes music with audio ducking & normalization | FFmpeg |

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

### 3. Run Pipeline

```bash
python main.py --input movie.mp4 --genre thriller
```

## 📖 Usage

### Basic Command

```bash
python main.py --input INPUT_VIDEO --genre GENRE [OPTIONS]
```

### Available Genres

- `thriller` - Suspenseful, building tension
- `action` - Fast-paced, high energy
- `drama` - Emotional, character-driven
- `horror` - Atmospheric, frightening
- `scifi` - Futuristic, wonder-filled
- `comedy` - Upbeat, humorous
- `romance` - Warm, emotional connection

### Key Options

```bash
--input PATH              Input video file (required)
--genre GENRE            Target trailer genre (default: thriller)
--output PATH            Output trailer path (default: output/trailer.mp4)
--test                   Process only first 5 shots for testing
--skip-analysis          Use cached analysis results
--resume-from STAGE      Resume from specific pipeline stage
--force-stage STAGE      Force re-run of specific stage
--reset-checkpoint       Reset checkpoint and start fresh
--skip-clean            Preserve existing output directory
```

### Resume Stages

Use with `--resume-from` or `--force-stage`:
- `shot_detection`
- `keyframe_extraction`
- `audio_extraction`
- `subtitle_management`
- `remote_analysis`
- `genre_scoring`
- `shot_selection`
- `narrative_generation`
- `video_assembly`
- `audio_mixing`

### Examples

```bash
# Basic usage
python main.py --input movie.mp4 --genre action

# Test with short sample
python main.py --input clip.mp4 --genre horror --test

# Resume from failed stage
python main.py --input movie.mp4 --genre thriller --resume-from remote_analysis --skip-clean

# Force re-run genre scoring with new profile
python main.py --input movie.mp4 --genre comedy --force-stage genre_scoring --skip-clean
```

## 📊 Output Structure

```
outputs/<sanitized_filename>/
├── shots/
│   ├── shot_0001.mp4
│   ├── shot_0002.mp4
│   └── shot_metadata.json
├── keyframes/
│   ├── kf_0001_1.jpg (5 frames per shot)
│   ├── kf_0001_2.jpg
│   └── ...
├── cache/
│   └── analysis_cache.json
├── output/
│   ├── timeline.json
│   ├── selected_shots.json
│   ├── trailer_assembled.mp4    # Stage 9 output
│   └── trailer_final.mp4         # Stage 10 output (FINAL)
├── temp/
│   └── partial_analysis.json
├── checkpoint.json
└── trailer_generator.log
```

## 🔧 Configuration

### Genre Profiles (`genre_profiles.yaml`)

Each genre defines:
- **Scoring weights**: Attribute importance (suspense, intensity, etc.)
- **Color grading**: FFmpeg filter specifications
- **Music tags**: Recommended audio styles
- **Pacing**: Shot timing preferences

### Settings (`settings.yaml`)

Key parameters:
```yaml
processing:
  target_trailer_length: 90        # Target duration in seconds
  shot_candidate_count: 60         # Number of shots to select
  max_batch_size: 50              # Processing batch size

shot_detection:
  threshold: 27.0                  # Scene change sensitivity

remote_analysis:
  qwen_server_url: "http://..."   # Multimodal analysis server
  batch_size: 10                   # Analysis batch size

azure_openai:
  temperature: 0.7                 # LLM creativity
  max_completion_tokens: 4000      # Response length
```

## 🔌 Qwen2-VL Server API

Your server should implement:

### Health Check
```
GET /health
```

### Batch Analysis
```
POST /analyze_batch
{
  "shots": [
    {
      "shot_id": 1,
      "images": ["base64_img1", "base64_img2", ...],  // 5 frames
      "audio_features": {
        "mfcc_mean": [...],
        "spectral_centroid_mean": 2500.5,
        "rms_energy_mean": 0.045,
        ...
      },
      "start_time": 10.5,
      "end_time": 12.8,
      "duration": 2.3
    }
  ]
}
```

**Expected Response:**
```json
{
  "caption": "Scene description",
  "attributes": {
    "suspense": 0.78,
    "darkness": 0.65,
    "emotional_tension": 0.59,
    ...
  }
}
```

## 🐛 Troubleshooting

### FFmpeg Not Found
```bash
# Verify installation
ffmpeg -version

# Add to PATH if needed
```

### Librosa Installation
```bash
pip install librosa soundfile

# Ubuntu/Debian: sudo apt-get install libsndfile1
# macOS: brew install libsndfile
```

### Server Connection Issues
- Verify server URL in `settings.yaml`
- Test with: `curl http://your-server:8000/health`
- Check firewall settings

### Out of Memory
- Reduce `batch_size` in settings
- Use `--test` flag for initial runs
- Process shorter clips first

## 📝 Implementation Status

### ✅ Completed
- Complete 10-stage processing pipeline
- Multi-frame + audio multimodal analysis
- Intelligent caching and checkpointing
- Genre-configurable scoring
- LLM-powered narrative generation
- Video assembly with transitions & color grading
- Audio mixing with music & ducking
- Resumable processing with stage control
- AI-powered title generation (optional)
- AI-powered transition & music selection (optional)

### 🚧 Planned
- Title card overlay rendering (drawtext filter)
- Advanced sound effects layer
- Beat detection for music sync
- Web UI dashboard
- Real-time preview mode

## 🤝 Contributing

Key extension points:
1. **New Genres**: Add profiles to `genre_profiles.yaml`
2. **Analysis Backends**: Implement `RemoteAnalyzer` interface
3. **LLM Providers**: Extend `AzureOpenAIClient`
4. **Custom Scoring**: Modify `GenreScorer` algorithms

## 🙏 Acknowledgments

- PySceneDetect for shot detection
- OpenCV for video processing
- librosa for audio analysis
- Azure OpenAI for narrative generation
- Qwen2-VL for multimodal understanding

## 🎵 Audio Assets

Place music files in `audio_assets/music/` for automatic selection. Name files with genre keywords for better matching:
- `thriller_suspense_01.mp3`
- `action_epic_music.wav`
- `horror_atmospheric.mp3`

See `audio_assets/README.md` for detailed music library setup and recommendations.

## 🎬 OMDB API Integration

GenreBender includes utilities for fetching movie metadata from the OMDB API (Open Movie Database). This standalone module can enrich your trailers with official movie information.

### Features
- ✅ Fetch complete movie metadata (plot, cast, ratings, etc.)
- ✅ Smart caching system (30-day TTL)
- ✅ Free API key included (1,000 requests/day)
- ✅ Multiple search methods (title, IMDb ID, general search)
- ✅ Comprehensive error handling

### Quick Example

```python
from utilities import OMDBClient, OMDBCache

# Initialize client and cache
client = OMDBClient()
cache = OMDBCache(output_dir="outputs/movie_name")

# Fetch movie with caching
movie = cache.get_or_fetch("Dumb and Dumber", client.get_movie_by_title)

# Access data
print(f"Title: {movie.title}")
print(f"Genre: {movie.genre}")
print(f"Plot: {movie.plot}")
print(f"IMDb Rating: {movie.imdb_rating}")
print(f"Actors: {movie.actors}")
```

### Documentation
- **Full Guide**: See `utilities/README.md` for complete documentation
- **Example Script**: Run `python utilities/example_omdb.py` for live demonstrations
- **Configuration**: OMDB settings in `trailer_generator/config/settings.yaml`

### Integration Ideas
- Enrich GPT-4 prompts with official plot and genre data
- Generate accurate title cards with movie metadata
- Use official genre classifications for better music selection
- Display ratings and cast information in final output

## 🧠 Story Graph Generator

GenreBender includes a standalone utility for generating comprehensive semantic story graphs from movies. This AI-powered tool creates structured narrative understanding that can be used for advanced trailer generation and analysis.

### What is a Story Graph?

A story graph is a machine-readable JSON representation containing:
- **Characters** with motivations and relationships
- **Plot structure** (setup, inciting incident, rising action, climax, resolution)
- **Scene timeline** with events, emotions, and visual inferences
- **Emotional arc** tracking throughout the film
- **Major themes** and genre indicators

### Quick Start

```bash
python 11_story_graph_generator.py \
  --movie-name "Caddyshack" \
  --synopsis "An exclusive golf course has to deal with a brash new member and a destructive dancing gopher." \
  --srt-file samples/caddyshack.srt
```

### Usage Examples

```bash
# Using inline synopsis
python 11_story_graph_generator.py \
  --movie-name "Movie Title" \
  --synopsis "A detailed plot summary here..." \
  --srt-file movie.srt

# Using synopsis from text file
python 11_story_graph_generator.py \
  --movie-name "Dumb and Dumber" \
  --synopsis synopsis.txt \
  --srt-file movie.srt

# Force overwrite existing graph
python 11_story_graph_generator.py \
  --movie-name "Movie" \
  --synopsis "..." \
  --srt-file movie.srt \
  --force

# Validate inputs without generating
python 11_story_graph_generator.py \
  --movie-name "Movie" \
  --synopsis "..." \
  --srt-file movie.srt \
  --validate-only
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--movie-name` | Movie title | Yes |
| `--synopsis` | Plot summary (text or .txt file path) | Yes |
| `--srt-file` | Path to SRT subtitle file | Yes |
| `--output-dir` | Output directory (default: `outputs/story_graphs`) | No |
| `--force` | Overwrite existing story graph | No |
| `--verbose` | Enable verbose logging | No |
| `--validate-only` | Only validate inputs, don't generate | No |

### Output Structure

```
outputs/story_graphs/<movie_name>/
├── story_graph.json          # Main output (structured JSON)
├── input_synopsis.txt        # Saved synopsis for reference
├── input_subtitles.srt       # Copy of subtitle file
├── metadata.json             # Generation metadata
└── story_graph_generator.log # Detailed logs
```

### Story Graph JSON Schema

```json
{
  "title": "Movie Title",
  "logline": "One-sentence plot summary",
  "characters": [
    {
      "name": "Character Name",
      "description": "Character description",
      "motivations": ["motivation1", "motivation2"],
      "relationships": {
        "Other Character": "relationship description"
      }
    }
  ],
  "major_themes": ["theme1", "theme2"],
  "plot_structure": {
    "setup": "Setup description",
    "inciting_incident": "Inciting incident",
    "rising_action": "Rising action",
    "climax": "Climax",
    "resolution": "Resolution"
  },
  "scene_timeline": [
    {
      "scene_id": 1,
      "start_time": "00:05:30",
      "end_time": "00:07:15",
      "summary": "Scene description",
      "key_events": ["event1", "event2"],
      "characters_present": ["character1"],
      "dominant_emotion": "tense",
      "genre_indicators": ["keyword1"],
      "visual_inferences": ["indoor", "night", "dark"]
    }
  ],
  "emotional_arc": [
    {
      "scene_id": 1,
      "emotion": "calm",
      "intensity": 0.3
    }
  ]
}
```

### Use Cases

1. **Enhanced Trailer Generation**: Use story graph for better shot selection and narrative structure
2. **Genre Analysis**: Identify original genre markers for genre-bending
3. **Character-Focused Edits**: Extract scenes featuring specific characters
4. **Emotional Pacing**: Understand and manipulate emotional flow
5. **Theme Extraction**: Identify and emphasize specific themes

### Technical Details

- **AI Model**: Azure OpenAI GPT-4 with structured JSON output
- **Token Management**: Automatically truncates long subtitles (keeps first 30%, middle 40%, last 30%)
- **Processing Time**: ~30-60 seconds per movie
- **Caching**: Overwrites on re-run (no persistent caching)

### Requirements

- Azure OpenAI API key configured in `settings.yaml`
- Valid SRT subtitle file
- Movie synopsis (at least 50 characters)
- Python dependencies: `pysrt`, `openai`, `pyyaml`

## 📚 Additional Documentation

- `STAGES_8_9_IMPLEMENTATION.md` - Detailed implementation guide for stages 8 & 9
- `audio_assets/README.md` - Music library setup and usage
- `PIPELINE_STAGES.md` - Complete pipeline architecture documentation
- `utilities/README.md` - OMDB API utilities documentation

---

**Final Output**: The system generates complete, broadcast-ready trailers with professional color grading, transitions, and mixed audio at `outputs/<video>/output/trailer_final.mp4`.
