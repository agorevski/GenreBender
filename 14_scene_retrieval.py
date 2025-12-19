#!/usr/bin/env python3
"""
Stage 14: Scene Retrieval (Layer 2.3)

Performs semantic beat-to-scene matching using FAISS and multi-factor scoring:
1. Load scene and beat embeddings
2. Build FAISS index for efficient similarity search
3. For each beat, retrieve top-k candidate scenes
4. Score using: semantic similarity + emotional alignment + visual match - genre penalty

Outputs:
- selected_scenes.json (beat_id -> ranked scene candidates)
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline_common import (
    setup_logging,
    load_config,
    get_output_base_dir,
    get_genre_output_dir,
    sanitize_filename,
    get_story_graph_dir
)
from trailer_generator.checkpoint import CheckpointManager
from trailer_generator.retrieval.scene_retriever import retrieve_scenes

logger = logging.getLogger(__name__)

STAGE_NAME = "scene_retrieval"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 14: Semantic scene retrieval for trailer beats"
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
        '--movie-name',
        type=str,
        help='Movie name (for beat sheet lookup, defaults to input filename)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of candidate scenes per beat (default: 10)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-retrieval even if results exist'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def validate_inputs(args, output_dir: Path, genre_output_dir: Path) -> tuple:
    """
    Validate required input files exist.
    
    Returns:
        Tuple of (embeddings_dir, beats_path, shot_metadata_path)
    """
    # Check embeddings (from stage 13) - now in genre-specific directory
    embeddings_dir = genre_output_dir / 'embeddings'
    scene_emb = embeddings_dir / 'scene_embeddings.pkl'
    beat_emb = embeddings_dir / 'beat_embeddings.pkl'
    
    if not scene_emb.exists() or not beat_emb.exists():
        logger.error(f"Embeddings not found in {embeddings_dir}")
        logger.error(f"Please run stage 13 first: python 13_embedding_generator.py --input {args.input} --genre {args.genre}")
        sys.exit(1)
    
    # Check shot metadata
    shot_metadata_path = output_dir / 'shots' / 'shot_metadata.json'
    if not shot_metadata_path.exists():
        logger.error(f"Shot metadata not found: {shot_metadata_path}")
        logger.error("Please run stages 1-5 first")
        sys.exit(1)
    
    # Determine movie name (use input filename if not provided)
    movie_name = args.movie_name
    if not movie_name:
        movie_name = Path(args.input).stem
    
    # Check beat sheet (genre-specific file)
    # Use get_story_graph_dir() to ensure proper sanitization (e.g., "Role Models" -> "Role_Models")
    story_graph_dir = get_story_graph_dir(movie_name)
    beats_path = story_graph_dir / f'beats_{args.genre}.json'
    
    if not beats_path.exists():
        logger.error(f"Beat sheet not found: {beats_path}")
        logger.error(f"Please run stage 12 first: python 12_beat_sheet_generator.py --movie-name '{movie_name}' --genre {args.genre}")
        sys.exit(1)
    
    logger.info(f"Input validation complete:")
    logger.info(f"  Embeddings: {embeddings_dir}")
    logger.info(f"  Beat sheet: {beats_path}")
    logger.info(f"  Shot metadata: {shot_metadata_path}")
    
    return embeddings_dir, beats_path, shot_metadata_path

def main():
    """Main execution function."""
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
    logger.info(f"Stage 14: Scene Retrieval (Layer 2.3)")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Target Genre: {args.genre}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Genre Output: {genre_output_dir}")
    
    # Check if already completed for this genre
    checkpoint = CheckpointManager(output_dir / 'checkpoint.json')
    if not args.force and checkpoint.is_stage_completed(STAGE_NAME, args.genre):
        logger.info(f"Stage '{STAGE_NAME}' already completed for genre '{args.genre}'. Use --force to re-retrieve.")
        return 0
    
    # Validate inputs
    embeddings_dir, beats_path, shot_metadata_path = validate_inputs(args, output_dir, genre_output_dir)
    
    # Load configuration
    config = load_config()
    retrieval_config = config.get('retrieval', {})
    
    # Get scoring weights
    scoring_weights = retrieval_config.get('scoring_weights', {
        'semantic_similarity': 0.50,
        'emotional_alignment': 0.25,
        'visual_match': 0.20,
        'original_genre_penalty': 0.05
    })
    
    # Perform scene retrieval
    logger.info("Retrieving scenes for beats...")
    logger.info(f"Scoring weights: {scoring_weights}")
    
    # Use genre-specific output path
    output_path = genre_output_dir / 'selected_scenes.json'
    
    try:
        results = retrieve_scenes(
            embeddings_dir=embeddings_dir,
            beats_path=beats_path,
            shot_metadata_path=shot_metadata_path,
            output_path=output_path,
            target_genre=args.genre,
            top_k=args.top_k,
            scoring_weights=scoring_weights
        )
        
        # Log summary
        total_beats = len(results)
        total_candidates = sum(len(scenes) for scenes in results.values())
        avg_candidates = total_candidates / total_beats if total_beats > 0 else 0
        
        logger.info("✓ Scene retrieval complete")
        logger.info(f"  Beats processed: {total_beats}")
        logger.info(f"  Total candidates: {total_candidates}")
        logger.info(f"  Avg candidates per beat: {avg_candidates:.1f}")
        logger.info(f"  Output: {output_path}")
        
        # Update checkpoint for this genre
        checkpoint.mark_stage_completed(STAGE_NAME, {
            'selected_scenes': str(output_path),
            'target_genre': args.genre,
            'top_k': args.top_k,
            'total_beats': total_beats,
            'total_candidates': total_candidates
        }, genre=args.genre)
        
        return 0
        
    except Exception as e:
        logger.error(f"Scene retrieval failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
