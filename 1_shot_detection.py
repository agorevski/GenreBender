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
    load_config, save_shots_to_metadata
)
from trailer_generator.ingest import ShotDetector

def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Shot Detection - Identify scene boundaries'
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'shot_detection', args.input, args.genre
    )
    
    # Check if already completed
    if checkpoint.is_stage_completed('shot_detection') and not args.force:
        logger.warning("⚠️  Shot detection already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 2_keyframe_extraction.py")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize detector
    detector = ShotDetector(
        threshold=config['shot_detection']['threshold'],
        chunk_duration=config['processing']['chunk_duration'],
        overlap=config['processing']['overlap'],
        output_dir=str(dirs['shots'])
    )
    
    # Detect shots
    logger.info("Starting shot detection...")
    shots = detector.detect_shots(args.input, streaming=True)
    logger.info(f"Detected {len(shots)} shots")
    
    # Limit to first 5 shots in test mode
    if args.test:
        original_count = len(shots)
        shots = shots[:5]
        logger.info(f"TEST MODE: Limited from {original_count} to {len(shots)} shots")
    
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
