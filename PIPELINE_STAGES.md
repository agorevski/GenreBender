# GenreBender Pipeline Stages

This document describes how to run the GenreBender pipeline in individual stages rather than all at once.

## Overview

Instead of running the entire pipeline with `main.py`, you can now run each stage independently using the numbered scripts:

1. `1_shot_detection.py` - Detect scene boundaries
2. `2_keyframe_extraction.py` - Extract frames from each shot
3. `3_audio_extraction.py` - Extract audio features
4. `4_remote_analysis.py` - Multimodal analysis with Qwen2-VL
5. `5_genre_scoring.py` - Score shots for target genre
6. `6_shot_selection.py` - Select top shots
7. `7_narrative_generation.py` - Generate timeline with GPT-4

## Benefits of Stage-by-Stage Execution

- **Resume capability**: Each stage saves progress and can be resumed
- **Debugging**: Easier to debug individual stages
- **Flexibility**: Re-run specific stages with different parameters
- **Inspection**: Review outputs between stages
- **Resource management**: Run stages at different times

## Quick Start

### Basic Usage

Run each stage in sequence with the same input file and genre:

```bash
# Stage 1: Shot Detection
python 1_shot_detection.py --input samples/movie.mp4 --genre thriller

# Stage 2: Keyframe Extraction
python 2_keyframe_extraction.py --input samples/movie.mp4 --genre thriller

# Stage 3: Audio Extraction
python 3_audio_extraction.py --input samples/movie.mp4 --genre thriller

# Stage 4: Remote Analysis (requires Qwen2-VL server running)
python 4_remote_analysis.py --input samples/movie.mp4 --genre thriller

# Stage 5: Genre Scoring
python 5_genre_scoring.py --input samples/movie.mp4 --genre thriller

# Stage 6: Shot Selection
python 6_shot_selection.py --input samples/movie.mp4 --genre thriller

# Stage 7: Narrative Generation (requires Azure OpenAI)
python 7_narrative_generation.py --input samples/movie.mp4 --genre thriller
```

### Test Mode

Use `--test` flag to process only the first 5 shots (useful for quick validation):

```bash
python 1_shot_detection.py --input samples/clip.mp4 --genre action --test
python 2_keyframe_extraction.py --input samples/clip.mp4 --genre action --test
# ... continue with other stages
```

## Command-Line Arguments

### Common Arguments (all stages)

- `--input PATH` - Path to input video file (required)
- `--genre GENRE` - Target trailer genre (default: thriller)
  - Choices: thriller, action, drama, horror, scifi, comedy, romance
- `--config PATH` - Configuration file path (default: trailer_generator/config/settings.yaml)
- `--test` - Process only first 5 shots for testing
- `--force` - Force re-run this stage even if already completed

### Stage-Specific Arguments

**Stage 4: Remote Analysis**
- `--skip-cache` - Disable analysis caching (forces re-analysis)

## Stage Details

### Stage 1: Shot Detection

Identifies scene boundaries using PySceneDetect.

**Input**: Video file  
**Output**: 
- `outputs/<video_name>/shots/shot_metadata.json` - Shot boundaries and metadata
- `outputs/<video_name>/shots/shot_XXXX.mp4` - Individual shot video files

**Example**:
```bash
python 1_shot_detection.py --input samples/movie.mp4 --genre thriller
```

### Stage 2: Keyframe Extraction

Extracts 5 frames per shot for temporal analysis.

**Prerequisites**: Stage 1 completed  
**Output**: 
- `outputs/<video_name>/keyframes/kf_XXXX_1.jpg` through `kf_XXXX_5.jpg`
- Updated `shot_metadata.json` with keyframe paths

**Example**:
```bash
python 2_keyframe_extraction.py --input samples/movie.mp4 --genre thriller
```

### Stage 3: Audio Extraction

Extracts audio features (MFCC, spectral centroid, RMS energy).

**Prerequisites**: Stages 1-2 completed  
**Output**: Updated `shot_metadata.json` with audio features

**Example**:
```bash
python 3_audio_extraction.py --input samples/movie.mp4 --genre thriller
```

### Stage 4: Remote Analysis

Performs multimodal analysis using Qwen2-VL server.

**Prerequisites**: 
- Stages 1-3 completed
- Qwen2-VL server running (check `settings.yaml` for URL)

**Output**: 
- `outputs/<video_name>/cache/analysis_cache.json` - Cached analysis results
- Updated `shot_metadata.json` with analysis attributes

**Example**:
```bash
# Start Qwen2-VL server first
cd qwen_server && ./start_server.sh

# Run analysis
python 4_remote_analysis.py --input samples/movie.mp4 --genre thriller

# To force re-analysis (ignore cache)
python 4_remote_analysis.py --input samples/movie.mp4 --genre thriller --skip-cache
```

### Stage 5: Genre Scoring

Scores shots based on target genre profile weights.

**Prerequisites**: Stages 1-4 completed  
**Output**: Updated `shot_metadata.json` with genre scores

**Example**:
```bash
python 5_genre_scoring.py --input samples/movie.mp4 --genre thriller

# Change genre (requires --force since previous stages used different genre)
python 5_genre_scoring.py --input samples/movie.mp4 --genre horror --force
```

### Stage 6: Shot Selection

Selects top-scored shots for trailer inclusion.

**Prerequisites**: Stages 1-5 completed  
**Output**: `outputs/<video_name>/output/selected_shots.json`

**Example**:
```bash
python 6_shot_selection.py --input samples/movie.mp4 --genre thriller
```

### Stage 7: Narrative Generation

Generates coherent trailer timeline using GPT-4.

**Prerequisites**: 
- Stages 1-6 completed
- Azure OpenAI API key configured (in `.env` or `settings.yaml`)

**Output**: `outputs/<video_name>/output/timeline.json` (final deliverable)

**Example**:
```bash
python 7_narrative_generation.py --input samples/movie.mp4 --genre thriller
```

## Advanced Usage

### Re-running Specific Stages

Each stage checks if it's already completed. Use `--force` to re-run:

```bash
# Re-run genre scoring with different genre
python 5_genre_scoring.py --input movie.mp4 --genre horror --force

# Re-run narrative generation with different parameters
python 7_narrative_generation.py --input movie.mp4 --genre horror --force
```

### Inspecting Outputs Between Stages

```bash
# View shot metadata
cat outputs/movie/shots/shot_metadata.json | jq '.'

# Count detected shots
jq '.total_shots' outputs/movie/shots/shot_metadata.json

# View selected shots
cat outputs/movie/output/selected_shots.json | jq '.'

# View final timeline
cat outputs/movie/output/timeline.json | jq '.'
```

### Checkpoint System

Each stage updates a checkpoint file that tracks progress:

```bash
# View checkpoint status
cat outputs/movie/checkpoint.json | jq '.'

# Check which stages are completed
jq '.stages' outputs/movie/checkpoint.json
```

### Error Recovery

If a stage fails partway through:

1. Check logs: `outputs/<video_name>/trailer_generator.log`
2. Review partial results in: `outputs/<video_name>/temp/`
3. Fix the issue
4. Re-run the stage (it will resume from where it left off)

## Comparison: Staged vs. Unified Pipeline

### Using `main.py` (unified pipeline):

```bash
python main.py --input movie.mp4 --genre thriller
```

**Pros:**
- Single command runs entire pipeline
- Automatic stage progression

**Cons:**
- Can't inspect intermediate outputs
- Must re-run entire pipeline if one stage fails
- Less flexibility for experimentation

### Using Stage Scripts (this approach):

```bash
python 1_shot_detection.py --input movie.mp4 --genre thriller
python 2_keyframe_extraction.py --input movie.mp4 --genre thriller
# ... etc
```

**Pros:**
- Inspect outputs between stages
- Re-run individual stages
- Better for debugging
- More control over execution

**Cons:**
- Multiple commands required
- Must track stage order manually

## Common Workflows

### First-Time Processing

```bash
VIDEO="samples/movie.mp4"
GENRE="thriller"

python 1_shot_detection.py --input $VIDEO --genre $GENRE
python 2_keyframe_extraction.py --input $VIDEO --genre $GENRE
python 3_audio_extraction.py --input $VIDEO --genre $GENRE
python 4_remote_analysis.py --input $VIDEO --genre $GENRE
python 5_genre_scoring.py --input $VIDEO --genre $GENRE
python 6_shot_selection.py --input $VIDEO --genre $GENRE
python 7_narrative_generation.py --input $VIDEO --genre $GENRE
```

### Experimenting with Different Genres

```bash
VIDEO="samples/movie.mp4"

# Run stages 1-4 once (genre-independent)
python 1_shot_detection.py --input $VIDEO --genre thriller
python 2_keyframe_extraction.py --input $VIDEO --genre thriller
python 3_audio_extraction.py --input $VIDEO --genre thriller
python 4_remote_analysis.py --input $VIDEO --genre thriller

# Try different genres (only re-run stages 5-7)
for GENRE in thriller horror comedy; do
  echo "Generating $GENRE trailer..."
  python 5_genre_scoring.py --input $VIDEO --genre $GENRE --force
  python 6_shot_selection.py --input $VIDEO --genre $GENRE --force
  python 7_narrative_generation.py --input $VIDEO --genre $GENRE --force
done
```

### Testing Configuration Changes

```bash
# Test with first 5 shots
python 1_shot_detection.py --input movie.mp4 --genre thriller --test
python 2_keyframe_extraction.py --input movie.mp4 --genre thriller --test
python 3_audio_extraction.py --input movie.mp4 --genre thriller --test
# ... continue

# If successful, run full pipeline
python 1_shot_detection.py --input movie.mp4 --genre thriller --force
# ... continue without --test flag
```

## Troubleshooting

### "Prerequisite stage not completed"

**Problem**: Trying to run a stage before its prerequisites

**Solution**: Run stages in order (1 through 7)

### "This stage is already completed"

**Problem**: Stage was already run successfully

**Solution**: Use `--force` flag to re-run

### "Qwen2-VL server not responding"

**Problem**: Stage 4 can't connect to analysis server

**Solution**:
```bash
# Check server status
curl http://localhost:8000/health

# Start server if not running
cd qwen_server && ./start_server.sh
```

### "Input file or genre mismatch"

**Problem**: Using different input file or genre than previous stages

**Solution**: Either use same parameters or use `--force` to override

## Output Structure

```
outputs/<sanitized_video_name>/
├── shots/
│   ├── shot_0001.mp4
│   ├── shot_0002.mp4
│   └── shot_metadata.json          # Complete shot data
├── keyframes/
│   ├── kf_0001_1.jpg (5 per shot)
│   └── ...
├── cache/
│   └── analysis_cache.json          # Cached multimodal analysis
├── output/
│   ├── selected_shots.json          # Top-scored shots
│   └── timeline.json                # Final timeline (PRIMARY OUTPUT)
├── temp/
│   └── partial_analysis.json        # Partial results during processing
├── checkpoint.json                   # Pipeline state
└── trailer_generator.log            # Detailed logs
```

## Next Steps

After completing all 7 stages, you'll have a `timeline.json` file that specifies:
- Shot selection and ordering
- Timing for each shot
- Transitions
- Narrative structure

Future stages (planned):
- Stage 8: Video assembly
- Stage 9: Audio mixing

For now, the timeline can be used with external video editing tools or custom assembly scripts.
