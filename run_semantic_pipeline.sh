#!/bin/bash
# Semantic Genre-Bending Trailer Pipeline
# Orchestrates all stages from shot detection to final trailer assembly
# REQUIRES: config.yaml with movie configurations
#
# Usage: ./run_semantic_pipeline.sh <config_key>
#
# Arguments:
#   config_key   - Key from config.yaml movies section (e.g., hitch, caddyshack)
#
# Example:
#   ./run_semantic_pipeline.sh hitch
#
# Config file format (config.yaml):
#   movies:
#     hitch:
#       movie_name: "Hitch"
#       video: "./test_files/hitch.mp4"
#       subtitles: "./test_files/hitch.srt"
#       synopsis: "./test_files/hitch_synopsis.txt"
#       genre: "thriller"

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Config file path
CONFIG_FILE="config.yaml"

# Parse arguments
KEY="$1"

# Validation - check for key argument
if [ -z "$KEY" ]; then
    echo -e "${RED}Error: Missing config key argument${NC}"
    echo ""
    echo "Usage: $0 <config_key>"
    echo ""
    echo "Available keys in $CONFIG_FILE:"
    if [ -f "$CONFIG_FILE" ]; then
        yq '.movies | keys | .[]' "$CONFIG_FILE" 2>/dev/null | sed 's/^/  - /'
    else
        echo "  (config file not found)"
    fi
    echo ""
    echo "Example:"
    echo "  $0 hitch"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo -e "${RED}Error: yq is not installed. Please install yq to parse YAML.${NC}"
    echo "Install with: sudo apt install yq  OR  pip install yq"
    exit 1
fi

# Check if key exists in config
KEY_EXISTS=$(yq ".movies | has(\"$KEY\")" "$CONFIG_FILE")
if [ "$KEY_EXISTS" != "true" ]; then
    echo -e "${RED}Error: Key '$KEY' not found in $CONFIG_FILE${NC}"
    echo ""
    echo "Available keys:"
    yq '.movies | keys | .[]' "$CONFIG_FILE" 2>/dev/null | sed 's/^/  - /'
    exit 1
fi

# Extract configuration values using yq
MOVIE_NAME=$(yq ".movies.$KEY.movie_name" "$CONFIG_FILE")
VIDEO=$(yq ".movies.$KEY.video" "$CONFIG_FILE")
SUBTITLES=$(yq ".movies.$KEY.subtitles" "$CONFIG_FILE")
SYNOPSIS=$(yq ".movies.$KEY.synopsis" "$CONFIG_FILE")
GENRE=$(yq ".movies.$KEY.genre" "$CONFIG_FILE")

# Remove quotes if present (yq may add them)
MOVIE_NAME=$(echo "$MOVIE_NAME" | sed 's/^"//;s/"$//')
VIDEO=$(echo "$VIDEO" | sed 's/^"//;s/"$//')
SUBTITLES=$(echo "$SUBTITLES" | sed 's/^"//;s/"$//')
SYNOPSIS=$(echo "$SYNOPSIS" | sed 's/^"//;s/"$//')
GENRE=$(echo "$GENRE" | sed 's/^"//;s/"$//')

# Validate all required fields are present
MISSING_FIELDS=""
if [ -z "$MOVIE_NAME" ] || [ "$MOVIE_NAME" = "null" ]; then
    MISSING_FIELDS="${MISSING_FIELDS}movie_name, "
fi
if [ -z "$VIDEO" ] || [ "$VIDEO" = "null" ]; then
    MISSING_FIELDS="${MISSING_FIELDS}video, "
fi
if [ -z "$SUBTITLES" ] || [ "$SUBTITLES" = "null" ]; then
    MISSING_FIELDS="${MISSING_FIELDS}subtitles, "
fi
if [ -z "$SYNOPSIS" ] || [ "$SYNOPSIS" = "null" ]; then
    MISSING_FIELDS="${MISSING_FIELDS}synopsis, "
fi
if [ -z "$GENRE" ] || [ "$GENRE" = "null" ]; then
    MISSING_FIELDS="${MISSING_FIELDS}genre, "
fi

if [ -n "$MISSING_FIELDS" ]; then
    echo -e "${RED}Error: Missing required fields for key '$KEY': ${MISSING_FIELDS%, }${NC}"
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

# Validate genre
VALID_GENRES="thriller action drama horror scifi comedy romance"
if ! echo "$VALID_GENRES" | grep -qw "$GENRE"; then
    echo -e "${RED}Error: Invalid genre '$GENRE'. Valid genres: $VALID_GENRES${NC}"
    exit 1
fi

# Header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     GenreBender: Semantic Trailer Generator           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration (key: $KEY):${NC}"
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
python 12_beat_sheet_generator.py --movie-name "$MOVIE_NAME" --genre "$GENRE" || exit 1

# Phase 3: Semantic Retrieval
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}PHASE 3: Semantic Retrieval Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}[1/3] Stage 13: Embedding Generation${NC}"
python 13_embedding_generator.py --input "$VIDEO" --genre "$GENRE" --movie-name "$MOVIE_NAME" || exit 1

echo ""
echo -e "${GREEN}[2/3] Stage 14: Scene Retrieval (FAISS)${NC}"
python 14_scene_retrieval.py --input "$VIDEO" --genre "$GENRE" --movie-name "$MOVIE_NAME" || exit 1
echo ""
echo -e "${GREEN}[3/3] Stage 15: Timeline Construction${NC}"
python 15_timeline_constructor.py --input "$VIDEO" --genre "$GENRE" || exit 1

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
echo -e "${BLUE}Config Key:${NC} $KEY"
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
