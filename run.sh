#!/usr/bin/env bash
set -e

# Run all GenreBender pipeline scripts in order
# Usage: ./run.sh filepath.mp4

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filepath.mp4>"
    exit 1
fi

INPUT_ARG="$1"

echo "Starting GenreBender pipeline with input: $INPUT_ARG"
echo "============================================"

echo "[1/9] Running shot detection..."
python 1_shot_detection.py --input $INPUT_ARG

echo "[2/9] Running keyframe extraction..."
python 2_keyframe_extraction.py --input $INPUT_ARG

echo "[3/9] Running audio extraction..."
python 3_audio_extraction.py --input $INPUT_ARG

echo "[4/9] Running remote analysis..."
python 4_remote_analysis.py --input $INPUT_ARG

echo "[5/9] Running genre scoring..."
python 5_genre_scoring.py --input $INPUT_ARG

echo "[6/9] Running shot selection..."
python 6_shot_selection.py --input $INPUT_ARG

echo "[7/9] Running narrative generation..."
python 7_narrative_generation.py --input $INPUT_ARG

echo "[8/9] Running video assembly..."
python 8_video_assembly.py --input $INPUT_ARG

echo "[9/9] Running audio mixing..."
python 9_audio_mixing.py --input $INPUT_ARG

echo "GenreBender pipeline completed successfully!"