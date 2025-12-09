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

echo "[1/10] Running shot detection..."
python 1_shot_detection.py --input $INPUT_ARG

echo "[2/10] Running keyframe extraction..."
python 2_keyframe_extraction.py --input $INPUT_ARG

echo "[3/10] Running audio extraction..."
python 3_audio_extraction.py --input $INPUT_ARG

echo "[4/10] Running subtitle management..."
python 4_subtitle_management.py --input $INPUT_ARG

echo "[5/10] Running remote analysis..."
python 5_remote_analysis.py --input $INPUT_ARG

echo "[6/10] Running genre scoring..."
python 6_genre_scoring.py --input $INPUT_ARG

echo "[7/10] Running shot selection..."
python 7_shot_selection.py --input $INPUT_ARG

echo "[8/10] Running narrative generation..."
python 8_narrative_generation.py --input $INPUT_ARG

echo "[9/10] Running video assembly..."
python 9_video_assembly.py --input $INPUT_ARG

echo "[10/10] Running audio mixing..."
python 10_audio_mixing.py --input $INPUT_ARG

echo "GenreBender pipeline completed successfully!"
