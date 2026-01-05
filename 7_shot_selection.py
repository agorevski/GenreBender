#!/usr/bin/env python3
"""
Stage 6: Shot Selection
Selects top-scored shots for trailer inclusion.
"""

import argparse
import sys
import json
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config, load_shots_from_metadata
)
from trailer_generator.analysis import ShotSelector

def main():
    """Execute the shot selection stage of the trailer generation pipeline.

    This function orchestrates Stage 6 of the pipeline, which selects the
    top-scored shots for inclusion in the final trailer. It loads shot
    metadata from previous stages, applies selection criteria, and saves
    the selected shots for narrative generation.

    Raises:
        SystemExit: If prerequisite stages are not completed, if the stage
            is already completed (without --force), or if shot loading fails.
    """
    parser = argparse.ArgumentParser(
        description='Stage 6: Shot Selection - Select top shots for trailer'
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'shot_selection', args.input, args.genre
    )
    
    # Validate prerequisites
    required_stages = ['shot_detection', 'keyframe_extraction', 'audio_extraction', 
                      'remote_analysis', 'genre_scoring']
    for stage in required_stages:
        if not checkpoint.is_stage_completed(stage):
            logger.error(f"❌ Prerequisite stage '{stage}' not completed.")
            print(f"\n❌ Error: You must complete all previous stages first!")
            print(f"Missing: {stage}")
            sys.exit(1)
    
    # Check if already completed
    if checkpoint.is_stage_completed('shot_selection') and not args.force:
        logger.warning("⚠️  Shot selection already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 8_narrative_generation.py")
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
    
    # Initialize shot selector
    selector = ShotSelector(
        target_count=config['processing']['shot_candidate_count']
    )
    
    # Select top shots
    logger.info(f"Selecting top {config['processing']['shot_candidate_count']} shots...")
    top_shots = selector.select_top_shots(shots)
    logger.info(f"Selected {len(top_shots)} top shots for trailer")
    
    # Save selected shots
    top_shots_path = dirs['output'] / 'selected_shots.json'
    with open(top_shots_path, 'w') as f:
        json.dump(top_shots, f, indent=2)
    logger.info(f"Saved selected shots to {top_shots_path}")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('shot_selection', {
        'selected_count': len(top_shots)
    })
    
    # Print completion
    print_completion_message('shot_selection', checkpoint, output_base)

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
