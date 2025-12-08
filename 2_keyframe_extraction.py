#!/usr/bin/env python3
"""
Stage 2: Keyframe Extraction
Extracts multiple frames per shot for temporal analysis.
"""

import argparse
import sys
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config, load_shots_from_metadata, save_shots_to_metadata
)
from trailer_generator.ingest import KeyframeExtractor

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: Keyframe Extraction - Extract multiple frames per shot'
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'keyframe_extraction', args.input, args.genre
    )
    
    # Validate prerequisite
    if not checkpoint.is_stage_completed('shot_detection'):
        logger.error("❌ Prerequisite stage 'shot_detection' not completed.")
        print("\n❌ Error: You must run 1_shot_detection.py first!")
        sys.exit(1)
    
    # Check if already completed
    if checkpoint.is_stage_completed('keyframe_extraction') and not args.force:
        logger.warning("⚠️  Keyframe extraction already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 3_audio_extraction.py")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load shots from metadata
    shot_metadata_path = dirs['shots'] / 'shot_metadata.json'
    shots = load_shots_from_metadata(shot_metadata_path)
    if not shots:
        logger.error("Failed to load shots from metadata")
        sys.exit(1)
    
    logger.info(f"Loaded {len(shots)} shots from metadata")
    
    # Initialize extractor
    extractor = KeyframeExtractor(
        output_dir=str(dirs['keyframes']),
        quality=config['keyframe']['quality']
    )
    
    # Extract keyframes (5 frames per shot for temporal analysis)
    logger.info("Starting keyframe extraction...")
    shots = extractor.extract_keyframes(args.input, shots, num_frames=5)
    logger.info(f"Extracted keyframes for {sum(1 for s in shots if s.get('keyframes'))} shots")
    
    # Save updated metadata
    save_shots_to_metadata(shots, shot_metadata_path, args.input)
    logger.info(f"Updated shot metadata with keyframe paths")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('keyframe_extraction', {
        'frames_extracted': sum(len(s.get('keyframes', [])) for s in shots)
    })
    
    # Print completion
    print_completion_message('keyframe_extraction', checkpoint, output_base)

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
