#!/usr/bin/env python3
"""
Stage 3: Audio Extraction
Extracts audio features (MFCC, spectral) for each shot.
"""

import argparse
import sys
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config
)
from trailer_generator.checkpoint import load_shots_from_metadata, save_shots_to_metadata
from trailer_generator.ingest import AudioExtractor

def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: Audio Extraction - Extract audio features per shot (genre-agnostic)'
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    
    # Initialize (genre-agnostic stage)
    output_base, dirs, checkpoint, logger = initialize_stage(
        'audio_extraction', args.input
    )
    
    # Validate prerequisites
    if not checkpoint.is_stage_completed('shot_detection'):
        logger.error("❌ Prerequisite stage 'shot_detection' not completed.")
        print("\n❌ Error: You must run 1_shot_detection.py first!")
        sys.exit(1)
    
    if not checkpoint.is_stage_completed('keyframe_extraction'):
        logger.error("❌ Prerequisite stage 'keyframe_extraction' not completed.")
        print("\n❌ Error: You must run 2_keyframe_extraction.py first!")
        sys.exit(1)
    
    # Check if already completed
    if checkpoint.is_stage_completed('audio_extraction') and not args.force:
        logger.warning("⚠️  Audio extraction already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 4_subtitle_management.py")
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
    
    # Initialize audio extractor
    audio_extractor = AudioExtractor(
        sample_rate=22050,
        n_mfcc=13
    )
    
    # Extract audio features
    logger.info("Starting audio feature extraction...")
    shots = audio_extractor.extract_audio_features(args.input, shots)
    logger.info(f"Extracted audio features for {sum(1 for s in shots if s.get('audio_features'))} shots")
    
    # Save updated metadata
    save_shots_to_metadata(shots, shot_metadata_path, args.input)
    logger.info(f"Updated shot metadata with audio features")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('audio_extraction', {
        'features_extracted': sum(1 for s in shots if s.get('audio_features'))
    })
    
    # Print completion
    print_completion_message('audio_extraction', checkpoint, output_base)

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
