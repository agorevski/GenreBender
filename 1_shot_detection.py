#!/usr/bin/env python3
"""
Stage 1: Shot Detection
Identifies scene boundaries in the input video using PySceneDetect.
"""

import argparse
import sys
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config
)
from trailer_generator.checkpoint import save_shots_to_metadata
from trailer_generator.ingest import ShotDetector

def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Shot Detection - Identify scene boundaries (genre-agnostic)'
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    
    # Initialize (genre-agnostic stage)
    output_base, dirs, checkpoint, logger = initialize_stage(
        'shot_detection', args.input
    )
    
    # Check if already completed
    if checkpoint.is_stage_completed('shot_detection') and not args.force:
        logger.warning("⚠️  Shot detection already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 2_keyframe_extraction.py")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize detector with PySceneDetect (with parallel processing support)
    detector = ShotDetector(
        threshold=config['shot_detection']['threshold'],
        chunk_duration=config['processing']['chunk_duration'],
        overlap=config['processing']['overlap'],
        output_dir=str(dirs['shots']),
        parallel_workers=config['shot_detection'].get('parallel_workers', 0),
        chunk_overlap=config['shot_detection'].get('chunk_overlap', 5.0)
    )
    
    # Get detection parameters from config
    frame_skip = config['shot_detection'].get('frame_skip', 0)
    parallel_detection = config['shot_detection'].get('parallel_detection', True)
    
    # Detect shots (parallel mode enabled by default if workers > 1)
    logger.info(f"Starting shot detection using PySceneDetect (parallel={parallel_detection})...")
    shots = detector.detect_shots(
        args.input, 
        streaming=True, 
        frame_skip=frame_skip,
        parallel=parallel_detection
    )
    logger.info(f"Detected {len(shots)} shots")
    
    # Verify extraction completed
    extracted_count = sum(1 for s in shots if s.get('file'))
    logger.info(f"Extracted {extracted_count} shot video files")
    
    # Save metadata
    shot_metadata_path = dirs['shots'] / 'shot_metadata.json'
    save_shots_to_metadata(shots, shot_metadata_path, args.input)
    logger.info(f"Saved shot metadata to {shot_metadata_path}")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('shot_detection', {'shots_count': len(shots)})
    
    # Print completion
    print_completion_message('shot_detection', checkpoint, output_base)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        import logging
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
