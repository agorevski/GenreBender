#!/usr/bin/env python3
"""
Stage 13: Embedding Generation

Generates vector embeddings for semantic scene retrieval:
1. Scene embeddings from shot metadata (visual + audio + story context)
2. Beat embeddings from beat sheet prompts

Outputs:
- scene_embeddings.pkl
- beat_embeddings.pkl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline_common import (
    initialize_genre_stage,
    load_config,
    print_completion_message,
    sanitize_filename,
    get_story_graph_dir
)
from trailer_generator.narrative.azure_client import AzureOpenAIClient
from trailer_generator.embeddings.embedding_generator import generate_embeddings

logger = logging.getLogger(__name__)

STAGE_NAME = "embedding_generation"


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - input (str): Input video file path.
            - genre (str): Target trailer genre.
            - movie_name (str | None): Optional movie name for story graph lookup.
            - force (bool): Whether to force regeneration.
            - verbose (bool): Whether to enable verbose logging.
    """
    parser = argparse.ArgumentParser(description="Stage 13: Generate embeddings for semantic scene retrieval")
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--genre', type=str, required=True, 
                       choices=['comedy', 'horror', 'thriller', 'parody', 'mockumentary', 
                                'crime', 'drama', 'experimental', 'fantasy', 'romance', 'scifi', 'action'], 
                       help='Target trailer genre')
    parser.add_argument('--movie-name', type=str, help='Movie name (for story graph lookup, defaults to input filename)')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if embeddings exist')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()


def validate_inputs(args, output_dir: Path) -> tuple:
    """Validate required input files exist.

    Checks that shot metadata and beat sheet files exist. Story graph is optional.

    Args:
        args: Parsed command line arguments containing input, genre, and movie_name.
        output_dir: Base output directory path where shot metadata is located.

    Returns:
        tuple: A tuple containing:
            - shot_metadata_path (Path): Path to the shot metadata JSON file.
            - beats_path (Path): Path to the genre-specific beat sheet JSON file.
            - story_graph_path (Path | None): Path to story graph, or None if not found.

    Raises:
        SystemExit: If shot metadata or beat sheet files are not found.
    """
    # Check shot metadata (from stages 1-5)
    shot_metadata_path = output_dir / 'shots' / 'shot_metadata.json'
    if not shot_metadata_path.exists():
        logger.error(f"Shot metadata not found: {shot_metadata_path}")
        logger.error("Please run stages 1-5 first")
        sys.exit(1)
    
    # Determine movie name (use input filename if not provided)
    movie_name = args.movie_name
    if not movie_name:
        movie_name = Path(args.input).stem
    
    # Check beat sheet (from stage 12) - genre-specific file
    # Use get_story_graph_dir() to ensure proper sanitization (e.g., "Role Models" -> "Role_Models")
    story_graph_dir = get_story_graph_dir(movie_name)
    beats_path = story_graph_dir / f'beats_{args.genre}.json'
    
    if not beats_path.exists():
        logger.error(f"Beat sheet not found: {beats_path}")
        logger.error(f"Please run stage 12 first: python 12_beat_sheet_generator.py --movie-name '{movie_name}' --genre {args.genre}")
        sys.exit(1)
    
    # Story graph is optional but recommended
    story_graph_path = story_graph_dir / 'story_graph.json'
    if not story_graph_path.exists():
        logger.warning(f"Story graph not found: {story_graph_path}")
        logger.warning("Embeddings will be generated without story context")
        story_graph_path = None
    
    logger.info(f"Input validation complete:")
    logger.info(f"  Shot metadata: {shot_metadata_path}")
    logger.info(f"  Beat sheet: {beats_path}")
    if story_graph_path:
        logger.info(f"  Story graph: {story_graph_path}")
    
    return shot_metadata_path, beats_path, story_graph_path


def main():
    """Main execution function for Stage 13 embedding generation.

    Orchestrates the embedding generation pipeline:
        1. Parses command line arguments.
        2. Initializes the stage with checkpoint tracking.
        3. Validates required input files.
        4. Loads configuration and initializes Azure OpenAI client.
        5. Generates scene and beat embeddings.
        6. Marks stage as complete in checkpoint.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    
    # Initialize stage with checkpoint (genre-dependent stage)
    output_base, dirs, checkpoint, logger = initialize_genre_stage(
        STAGE_NAME, 
        args.input, 
        args.genre
    )
    
    # Check if already completed for this genre
    if not args.force and checkpoint.is_stage_completed(STAGE_NAME, args.genre):
        logger.info(f"Stage '{STAGE_NAME}' already completed. Use --force to regenerate.")
        print_completion_message(STAGE_NAME, checkpoint, output_base)
        return 0
    
    # Validate inputs
    shot_metadata_path, beats_path, story_graph_path = validate_inputs(args, output_base)
    
    # Load configuration
    config = load_config()
    
    # Initialize Azure OpenAI client
    azure_config = config.get('azure_openai', {})
    azure_client_instance = AzureOpenAIClient(
        endpoint=azure_config.get('endpoint'),
        api_key=azure_config.get('api_key'),
        deployment_name=azure_config.get('deployment_name'),
        api_version=azure_config.get('api_version')
    )
    
    # Generate embeddings - use genre-specific directory
    logger.info("Generating embeddings...")
    
    # Use genre-specific embeddings directory
    embeddings_dir = dirs['genre_embeddings']
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Embeddings output: {embeddings_dir}")
    
    try:
        scene_emb_path, beat_emb_path = generate_embeddings(
            output_dir=embeddings_dir,
            shot_metadata_path=shot_metadata_path,
            beats_path=beats_path,
            story_graph_path=story_graph_path,
            azure_client=azure_client_instance.client,
            config=config
        )
        
        logger.info("✓ Embedding generation complete")
        logger.info(f"  Scene embeddings: {scene_emb_path}")
        logger.info(f"  Beat embeddings: {beat_emb_path}")
        
        # Mark stage as complete for this genre
        checkpoint.mark_stage_completed(STAGE_NAME, {
            'scene_embeddings': str(scene_emb_path),
            'beat_embeddings': str(beat_emb_path),
            'target_genre': args.genre
        }, genre=args.genre)
        
        # Print completion message
        print_completion_message(STAGE_NAME, checkpoint, output_base)
        
        return 0
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
