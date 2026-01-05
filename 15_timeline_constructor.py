#!/usr/bin/env python3
"""
Stage 15: Timeline Construction

Builds deterministic shot-level timeline from beat-matched scenes:
1. Load selected scenes from stage 14
2. Allocate time budget per beat based on position
3. Select shots and assign precise timings
4. Add genre-appropriate transitions
5. Calculate pacing profile

Outputs:
- trailer_timeline.json (complete shot-level timeline)
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline_common import (
    setup_logging,
    load_config,
    get_output_base_dir,
    get_genre_output_dir
)
from trailer_generator.checkpoint import CheckpointManager
from trailer_generator.narrative.timeline_constructor import construct_timeline

logger = logging.getLogger(__name__)

STAGE_NAME = "timeline_construction"

def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - input (str): Input video file path.
            - genre (str): Target trailer genre.
            - target_duration (int): Target trailer duration in seconds.
            - force (bool): Force reconstruction flag.
            - verbose (bool): Verbose logging flag.
    """
    parser = argparse.ArgumentParser(
        description="Stage 15: Construct deterministic trailer timeline"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input video file path'
    )
    parser.add_argument(
        '--genre',
        type=str,
        required=True,
        choices=['comedy', 'horror', 'thriller', 'parody', 'mockumentary', 
                 'crime', 'drama', 'experimental', 'fantasy', 'romance', 'scifi', 'action'],
        help='Target trailer genre'
    )
    parser.add_argument(
        '--target-duration',
        type=int,
        default=90,
        help='Target trailer duration in seconds (default: 90)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reconstruction even if timeline exists'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def validate_inputs(genre_output_dir: Path, args) -> Path:
    """Validate required input files exist.

    Args:
        genre_output_dir: Path to the genre-specific output directory.
        args: Parsed command line arguments containing input and genre.

    Returns:
        Path to the selected_scenes.json file.

    Raises:
        SystemExit: If selected scenes file does not exist.
    """
    # Check selected scenes (from stage 14) - now in genre-specific directory
    selected_scenes_path = genre_output_dir / 'selected_scenes.json'
    
    if not selected_scenes_path.exists():
        logger.error(f"Selected scenes not found: {selected_scenes_path}")
        logger.error(f"Please run stage 14 first: python 14_scene_retrieval.py --input {args.input} --genre {args.genre}")
        sys.exit(1)
    
    logger.info(f"Input validation complete:")
    logger.info(f"  Selected scenes: {selected_scenes_path}")
    
    return selected_scenes_path

def main():
    """Main execution function for timeline construction stage.

    Orchestrates the timeline construction pipeline:
    1. Sets up logging and output directories.
    2. Validates input files from previous stages.
    3. Constructs a shot-level timeline with precise timings.
    4. Saves results and updates checkpoint.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    
    # Setup
    output_dir = get_output_base_dir(args.input)
    log_file = output_dir / 'trailer_generator.log'
    
    # Setup logging with proper level
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_file, level=log_level)
    
    # Get genre-specific output directory
    genre_output_dir = get_genre_output_dir(args.input, args.genre)
    genre_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info(f"Stage 15: Timeline Construction")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Genre: {args.genre}")
    logger.info(f"Target Duration: {args.target_duration}s")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Genre Output: {genre_output_dir}")
    
    # Check if already completed for this genre
    checkpoint = CheckpointManager(output_dir / 'checkpoint.json')
    if not args.force and checkpoint.is_stage_completed(STAGE_NAME, args.genre):
        logger.info(f"Stage '{STAGE_NAME}' already completed for genre '{args.genre}'. Use --force to reconstruct.")
        return 0
    
    # Validate inputs
    selected_scenes_path = validate_inputs(genre_output_dir, args)
    
    # Load configuration
    config = load_config()
    
    # Construct timeline
    logger.info("Constructing timeline...")
    
    # Use genre-specific output path
    output_path = genre_output_dir / 'trailer_timeline.json'
    
    try:
        timeline = construct_timeline(
            selected_scenes_path=selected_scenes_path,
            output_path=output_path,
            target_duration=args.target_duration,
            genre=args.genre
        )
        
        # Log summary
        total_shots = timeline['total_shots']
        actual_duration = timeline['actual_duration']
        pacing = timeline['metadata']['pacing_profile']
        
        logger.info("✓ Timeline construction complete")
        logger.info(f"  Total shots: {total_shots}")
        logger.info(f"  Duration: {actual_duration:.1f}s (target: {args.target_duration}s)")
        logger.info(f"  Avg shot duration: {pacing['avg_shot_duration']:.2f}s")
        logger.info(f"  Shots per minute: {pacing['shots_per_minute']:.1f}")
        logger.info(f"  Output: {output_path}")
        
        # Update checkpoint for this genre
        checkpoint.mark_stage_completed(STAGE_NAME, {
            'timeline': str(output_path),
            'genre': args.genre,
            'target_duration': args.target_duration,
            'actual_duration': actual_duration,
            'total_shots': total_shots,
            'pacing_profile': pacing
        }, genre=args.genre)
        
        return 0
        
    except Exception as e:
        logger.error(f"Timeline construction failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
