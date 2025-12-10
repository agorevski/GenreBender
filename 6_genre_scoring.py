#!/usr/bin/env python3
"""
Stage 5: Genre Scoring
Scores shots based on target genre profile.
"""

import argparse
import sys
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config, load_genre_profile, load_shots_from_metadata, save_shots_to_metadata
)
from trailer_generator.analysis import GenreScorer

def main():
    parser = argparse.ArgumentParser(
        description='Stage 5: Genre Scoring - Score shots based on genre profile'
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'genre_scoring', args.input, args.genre
    )
    
    # Validate prerequisites
    required_stages = ['shot_detection', 'keyframe_extraction', 'audio_extraction', 'remote_analysis']
    for stage in required_stages:
        if not checkpoint.is_stage_completed(stage):
            logger.error(f"❌ Prerequisite stage '{stage}' not completed.")
            print(f"\n❌ Error: You must complete all previous stages first!")
            print(f"Missing: {stage}")
            sys.exit(1)
    
    # Check if already completed
    if checkpoint.is_stage_completed('genre_scoring') and not args.force:
        logger.warning("⚠️  Genre scoring already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 7_shot_selection.py")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    genre_profile = load_genre_profile(args.genre)
    
    # Load shots from metadata
    shot_metadata_path = dirs['shots'] / 'shot_metadata.json'
    shots = load_shots_from_metadata(shot_metadata_path)
    if not shots:
        logger.error("Failed to load shots from metadata")
        sys.exit(1)
    
    logger.info(f"Loaded {len(shots)} shots from metadata")
    
    # Initialize genre scorer
    scorer = GenreScorer(genre_profile['scoring_weights'])
    
    # Score shots
    logger.info(f"Scoring shots for genre: {args.genre}")
    shots = scorer.score_shots(shots)
    scored_count = sum(1 for s in shots if 'genre_score' in s)
    logger.info(f"Scored {scored_count} shots")
    
    # Save updated metadata
    save_shots_to_metadata(shots, shot_metadata_path, args.input)
    logger.info(f"Updated shot metadata with genre scores")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('genre_scoring', {
        'scored_count': scored_count,
        'genre': args.genre
    })
    
    # Print completion
    print_completion_message('genre_scoring', checkpoint, output_base)

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
