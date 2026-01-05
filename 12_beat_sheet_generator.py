#!/usr/bin/env python3
"""
Stage 12: Beat Sheet Generator
===============================

Generates genre-specific trailer beat sheets from story graphs (Stage 11 output).

Usage:
    python 12_beat_sheet_generator.py --movie-name "Movie Title" --genre thriller
    python 12_beat_sheet_generator.py --movie-name "Movie Title" --genre horror --force

Input:
    - story_graph.json from outputs/story_graphs/<movie>/
    - target genre selection

Output:
    - beats_{genre}.json: Complete beat sheet with embedding prompts (per genre)
    - genre_rewrite_{genre}.json: Intermediate genre reinterpretation (per genre)
    - metadata_beats.json: Generation metadata
"""

import sys
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline_common import (
    load_config,
    setup_logging,
    get_story_graph_dir,
    sanitize_filename
)
from trailer_generator.narrative.azure_client import AzureOpenAIClient
from trailer_generator.narrative.beat_sheet_generator import BeatSheetGenerator

def parse_arguments():
    """Parse command line arguments for beat sheet generation.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - movie_name (str): Movie name matching Stage 11 story graph directory.
            - genre (str): Target genre for trailer (e.g., 'thriller', 'horror').
            - output_dir (Path or None): Custom output directory path.
            - force (bool): Whether to force regeneration if beats.json exists.
            - temperature (float or None): LLM temperature override.
            - validate_only (bool): Whether to only validate inputs without generating.
    """
    parser = argparse.ArgumentParser(
        description="Stage 12: Generate genre-specific trailer beat sheet from story graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate thriller beat sheet
  python 12_beat_sheet_generator.py --movie-name "Airplane!" --genre thriller
  
  # Generate horror beat sheet (force regeneration)
  python 12_beat_sheet_generator.py --movie-name "Airplane!" --genre horror --force
  
  # Custom output directory
  python 12_beat_sheet_generator.py --movie-name "Movie" --genre action --output-dir custom/

Available Genres:
  comedy, horror, thriller, parody, mockumentary, crime, drama,
  experimental, fantasy, romance, scifi, action
        """
    )
    
    parser.add_argument(
        '--movie-name',
        type=str,
        required=True,
        help='Movie name (must match Stage 11 story graph directory)'
    )
    
    parser.add_argument(
        '--genre',
        type=str,
        required=True,
        choices=['comedy', 'horror', 'thriller', 'parody', 'mockumentary', 
                 'crime', 'drama', 'experimental', 'fantasy', 'romance', 'scifi', 'action'],
        help='Target genre for trailer'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Custom output directory (default: outputs/story_graphs/<movie>/)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if beats.json exists'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='LLM temperature (default: from settings.yaml)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate inputs without generating'
    )
    
    return parser.parse_args()

def validate_inputs(story_graph_path: Path, movie_name: str) -> None:
    """Validate that required inputs exist and story graph structure is valid.

    Args:
        story_graph_path: Path to the story_graph.json file from Stage 11.
        movie_name: Movie name used for error messaging.

    Raises:
        FileNotFoundError: If the story graph file does not exist.
        ValueError: If the story graph is missing required fields (title, logline,
            characters, plot_structure, scene_timeline).
    """
    if not story_graph_path.exists():
        raise FileNotFoundError(
            f"Story graph not found: {story_graph_path}\n"
            f"Run Stage 11 first: python 11_story_graph_generator.py --movie-name \"{movie_name}\" ..."
        )
    
    # Validate story graph structure
    with open(story_graph_path, 'r') as f:
        story_graph = json.load(f)
    
    required_fields = ['title', 'logline', 'characters', 'plot_structure', 'scene_timeline']
    missing = [f for f in required_fields if f not in story_graph]
    
    if missing:
        raise ValueError(f"Story graph missing required fields: {missing}")
    
    logging.info(f"✓ Story graph validated: {story_graph.get('title')}")

def main():
    """Main execution for Stage 12 beat sheet generation.

    Orchestrates the complete beat sheet generation workflow:
        1. Parses command line arguments and loads configuration.
        2. Validates story graph inputs from Stage 11.
        3. Initializes Azure OpenAI client and beat sheet generator.
        4. Generates genre-specific beat sheet with embedding prompts.
        5. Saves outputs (beats, genre rewrite, metadata) to output directory.

    Returns:
        int: Exit code (0 for success, 1 for failure).

    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If story graph validation fails.
        Exception: For any other generation failures.
    """
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_story_graph_dir(args.movie_name)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "beat_sheet_generator.log"
    setup_logging(log_file, config.get('logging', {}).get('level', 'INFO'))
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Stage 12: Beat Sheet Generator")
    logger.info("=" * 80)
    logger.info(f"Movie: {args.movie_name}")
    logger.info(f"Target Genre: {args.genre}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Force Regeneration: {args.force}")
    
    try:
        # Validate inputs
        story_graph_path = output_dir / "story_graph.json"
        logger.info(f"Story graph path: {story_graph_path}")
        
        validate_inputs(story_graph_path, args.movie_name)
        
        if args.validate_only:
            logger.info("Validation complete. Exiting (--validate-only)")
            print("\n✓ Validation passed")
            return 0
        
        # Check if output already exists (genre-specific)
        beats_path = output_dir / f"beats_{args.genre}.json"
        if beats_path.exists() and not args.force:
            logger.info(f"Beat sheet already exists for genre '{args.genre}': {beats_path}")
            logger.info("Use --force to regenerate")
            print(f"\n✓ Beat sheet already exists for '{args.genre}': {beats_path}")
            print("Use --force to regenerate")
            return 0
        
        # Load story graph
        logger.info(f"Loading story graph from: {story_graph_path}")
        with open(story_graph_path, 'r') as f:
            story_graph = json.load(f)
        
        logger.info(f"Story graph loaded: {story_graph.get('title')}")
        logger.info(f"  Characters: {len(story_graph.get('characters', []))}")
        logger.info(f"  Scenes: {len(story_graph.get('scene_timeline', []))}")
        
        # Initialize Azure OpenAI client
        azure_config = config['azure_openai']
        logger.info("Initializing Azure OpenAI client...")
        
        azure_client = AzureOpenAIClient(
            endpoint=azure_config['endpoint'],
            api_key=azure_config.get('api_key'),
            deployment_name=azure_config['deployment_name'],
            api_version=azure_config['api_version'],
            max_retries=azure_config['max_retries'],
            temperature=args.temperature or azure_config.get('temperature'),
            max_completion_tokens=azure_config.get('max_completion_tokens')
        )
        
        # Initialize beat sheet generator
        beat_sheet_config = config.get('beat_sheet', {})
        logger.info("Initializing beat sheet generator...")
        
        generator = BeatSheetGenerator(
            azure_client=azure_client,
            temperature=args.temperature or beat_sheet_config.get('temperature'),
            min_beats=beat_sheet_config.get('min_beats'),
            max_beats=beat_sheet_config.get('max_beats')
        )
        
        # Generate beat sheet (save genre_rewrite with genre-specific name)
        logger.info("Starting beat sheet generation...")
        start_time = datetime.now()
        
        result = generator.generate_beat_sheet(
            story_graph=story_graph,
            target_genre=args.genre,
            output_dir=output_dir,
            genre_rewrite_filename=f"genre_rewrite_{args.genre}.json"
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Beat sheet generation completed in {duration:.1f} seconds")
        
        # Save beat sheet (genre-specific filename)
        beat_sheet = result['beat_sheet']
        beats_path = output_dir / f"beats_{args.genre}.json"
        generator.save_beat_sheet(beat_sheet, beats_path)
        
        # Save metadata
        metadata = {
            "movie_name": args.movie_name,
            "target_genre": args.genre,
            "story_graph_path": str(story_graph_path),
            "generation_timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "beat_count": len(beat_sheet.get('beats', [])),
            "temperature": args.temperature or beat_sheet_config.get('temperature'),
            "force_regeneration": args.force
        }
        
        metadata_path = output_dir / "metadata_beats.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to: {metadata_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("BEAT SHEET GENERATION COMPLETE")
        print("=" * 80)
        print(f"Movie: {args.movie_name}")
        print(f"Genre: {args.genre}")
        print(f"Beat Count: {len(beat_sheet.get('beats', []))}")
        print(f"Duration: {duration:.1f}s")
        print(f"\nOutputs:")
        print(f"  ✓ beats_{args.genre}.json: {beats_path}")
        print(f"  ✓ genre_rewrite_{args.genre}.json: {output_dir / f'genre_rewrite_{args.genre}.json'}")
        print(f"  ✓ metadata: {metadata_path}")
        print(f"  ✓ logs: {log_file}")
        print("\nNext step:")
        print(f"  Layer 2.3: Scene retrieval using embedding prompts")
        print("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n✗ Error: {e}")
        return 1
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n✗ Validation error: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"Beat sheet generation failed: {e}", exc_info=True)
        print(f"\n✗ Beat sheet generation failed: {e}")
        print(f"Check logs: {log_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
