#!/bin/bash
# Semantic Genre-Bending Trailer Pipeline
# Orchestrates all stages from shot detection to final trailer assembly
# REQUIRES: synopsis and subtitle files for semantic pipeline
#
# Usage: ./run_semantic_pipeline.sh <video.mp4> <genre> <movie_name> <synopsis.txt> <subtitles.srt>
#
# Arguments:
#   video.mp4    - Input video file
#   genre        - Target genre (thriller, action, drama, horror, scifi, comedy, romance)
#   movie_name   - Movie title (for story graph lookup)
#   synopsis.txt - Path to synopsis text file (REQUIRED)
#   subtitles.srt- Path to SRT subtitle file (REQUIRED)
#
# Example:
#   ./run_semantic_pipeline.sh test_files/hitch.mp4 thriller "Hitch" \
#     test_files/hitch_synopsis.txt test_files/hitch.srt

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
VIDEO="$1"
GENRE="$2"
MOVIE_NAME="$3"
SYNOPSIS="$4"
SUBTITLES="$5"

# Validation
if [ -z "$VIDEO" ] || [ -z "$GENRE" ] || [ -z "$MOVIE_NAME" ] || [ -z "$SYNOPSIS" ] || [ -z "$SUBTITLES" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 <video.mp4> <genre> <movie_name> <synopsis.txt> <subtitles.srt>"
    echo ""
    echo "Available genres: thriller, action, drama, horror, scifi, comedy, romance"
    echo ""
    echo "Example:"
    echo "  $0 movie.mp4 thriller \"Movie Title\" synopsis.txt movie.srt"
    echo ""
    echo "This script runs the SEMANTIC PIPELINE ONLY."
    echo "Synopsis and subtitle files are REQUIRED for story understanding."
    exit 1
fi

# Check if files exist
if [ ! -f "$VIDEO" ]; then
    echo -e "${RED}Error: Video file not found: $VIDEO${NC}"
    exit 1
fi

if [ ! -f "$SYNOPSIS" ]; then
    echo -e "${RED}Error: Synopsis file not found: $SYNOPSIS${NC}"
    exit 1
fi

if [ ! -f "$SUBTITLES" ]; then
    echo -e "${RED}Error: Subtitle file not found: $SUBTITLES${NC}"
    exit 1
fi

# Header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     GenreBender: Semantic Trailer Generator           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Video:      $VIDEO"
echo "  Genre:      $GENRE"
echo "  Movie Name: $MOVIE_NAME"
echo "  Synopsis:   $SYNOPSIS"
echo "  Subtitles:  $SUBTITLES"
echo -e "  ${GREEN}Mode:       SEMANTIC PIPELINE (Stages 1-5 → 11-12 → 13-15 → 9-10)${NC}"
echo ""

# Phase 1: Shot Detection & Multimodal Analysis
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 1: Shot Detection & Multimodal Analysis${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}[1/5] Stage 1: Shot Detection${NC}"
python 1_shot_detection.py --input "$VIDEO" --genre "$GENRE" || exit 1

echo ""
echo -e "${GREEN}[2/5] Stage 2: Keyframe Extraction${NC}"
python 2_keyframe_extraction.py --input "$VIDEO" --genre "$GENRE" || exit 1

echo ""
echo -e "${GREEN}[3/5] Stage 3: Audio Feature Extraction${NC}"
python 3_audio_extraction.py --input "$VIDEO" --genre "$GENRE" || exit 1

echo ""
echo -e "${GREEN}[4/5] Stage 4: Subtitle Management${NC}"
python 4_subtitle_management.py --input "$VIDEO" --genre "$GENRE" || exit 1

echo ""
echo -e "${GREEN}[5/5] Stage 5: Multimodal Analysis (Qwen2-VL)${NC}"
python 5_remote_analysis.py --input "$VIDEO" --genre "$GENRE" || exit 1

# Phase 2: Story Understanding
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 2: Story Understanding Layer${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}[1/2] Stage 11: Story Graph Generation${NC}"
python 11_story_graph_generator.py --movie-name "$MOVIE_NAME" --synopsis "$SYNOPSIS" --srt-file "$SUBTITLES" || exit 1

echo ""
echo -e "${GREEN}[2/2] Stage 12: Beat Sheet Generation${NC}"
python 12_beat_sheet_generator.py --movie-name "$MOVIE_NAME" --target-genre "$GENRE" || exit 1

# Phase 3: Semantic Retrieval
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 3: Semantic Retrieval Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}[1/3] Stage 13: Embedding Generation${NC}"
python 13_embedding_generator.py --input "$VIDEO" --target-genre "$GENRE" --movie-name "$MOVIE_NAME" || exit 1

echo ""
echo -e "${GREEN}[2/3] Stage 14: Scene Retrieval (FAISS)${NC}"
python 14_scene_retrieval.py --input "$VIDEO" --target-genre "$GENRE" --movie-name "$MOVIE_NAME" || exit 1
echo ""
echo -e "${GREEN}[3/3] Stage 15: Timeline Construction${NC}"
python 15_timeline_constructor.py --input "$VIDEO" --target-genre "$GENRE" || exit 1

# Phase 4: Final Assembly
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 4: Video & Audio Assembly${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}[1/2] Stage 9: Video Assembly${NC}"
python 9_video_assembly.py --input "$VIDEO" --genre "$GENRE" || exit 1

echo ""
echo -e "${GREEN}[2/2] Stage 10: Audio Mixing${NC}"
python 10_audio_mixing.py --input "$VIDEO" --genre "$GENRE" || exit 1

# Success
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 ✓ PIPELINE COMPLETE                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get sanitized filename for output path
SANITIZED=$(python -c "import sys; from pathlib import Path; print(''.join(c for c in Path('$VIDEO').stem if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_'))")
OUTPUT_DIR="outputs/$SANITIZED"

echo -e "${GREEN}Final Trailer:${NC} $OUTPUT_DIR/output/trailer_final.mp4"
echo ""
echo -e "${BLUE}Pipeline:${NC} Semantic (Stages 1-5 → 11-12 → 13-15 → 9-10)"
echo -e "${BLUE}Story Graph:${NC} outputs/story_graphs/$MOVIE_NAME/story_graph.json"
echo -e "${BLUE}Beat Sheet:${NC} outputs/story_graphs/$MOVIE_NAME/beats.json"
echo -e "${BLUE}Embeddings:${NC} $OUTPUT_DIR/embeddings/"
echo -e "${BLUE}  - Scene:${NC} $OUTPUT_DIR/embeddings/scene_embeddings.pkl"
echo -e "${BLUE}  - Beat:${NC} $OUTPUT_DIR/embeddings/beat_embeddings.pkl"
echo -e "${BLUE}Retrieval:${NC} $OUTPUT_DIR/output/selected_scenes.json"
echo -e "${BLUE}Timeline:${NC} $OUTPUT_DIR/output/trailer_timeline.json"
echo ""
echo -e "${YELLOW}Logs:${NC} $OUTPUT_DIR/trailer_generator.log"
echo ""
