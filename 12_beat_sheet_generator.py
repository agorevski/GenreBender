#!/usr/bin/env python3
"""
Stage 12: Beat Sheet Generator
===============================

Generates genre-specific trailer beat sheets from story graphs (Stage 11 output).

Usage:
    python 12_beat_sheet_generator.py --movie-name "Movie Title" --target-genre thriller
    python 12_beat_sheet_generator.py --movie-name "Movie Title" --target-genre horror --force

Input:
    - story_graph.json from outputs/story_graphs/<movie>/
    - target genre selection

Output:
    - beats.json: Complete beat sheet with embedding prompts
    - genre_rewrite.json: Intermediate genre reinterpretation
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 12: Generate genre-specific trailer beat sheet from story graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate thriller beat sheet
  python 12_beat_sheet_generator.py --movie-name "Airplane!" --target-genre thriller
  
  # Generate horror beat sheet (force regeneration)
  python 12_beat_sheet_generator.py --movie-name "Airplane!" --target-genre horror --force
  
  # Custom output directory
  python 12_beat_sheet_generator.py --movie-name "Movie" --target-genre action --output-dir custom/

Available Genres:
  thriller, action, drama, horror, scifi, comedy, romance
        """
    )
    
    parser.add_argument(
        '--movie-name',
        type=str,
        required=True,
        help='Movie name (must match Stage 11 story graph directory)'
    )
    
    parser.add_argument(
        '--target-genre',
        type=str,
        required=True,
        choices=['thriller', 'action', 'drama', 'horror', 'scifi', 'comedy', 'romance'],
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
    """
    Validate that required inputs exist.
    
    Args:
        story_graph_path: Path to story_graph.json
        movie_name: Movie name
    
    Raises:
        FileNotFoundError: If inputs don't exist
        ValueError: If story graph is invalid
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
    """Main execution."""
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
    logger.info(f"Target Genre: {args.target_genre}")
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
        
        # Check if output already exists
        beats_path = output_dir / "beats.json"
        if beats_path.exists() and not args.force:
            logger.info(f"Beat sheet already exists: {beats_path}")
            logger.info("Use --force to regenerate")
            print(f"\n✓ Beat sheet already exists: {beats_path}")
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
        
        # Generate beat sheet
        logger.info("Starting beat sheet generation...")
        start_time = datetime.now()
        
        result = generator.generate_beat_sheet(
            story_graph=story_graph,
            target_genre=args.target_genre,
            output_dir=output_dir
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Beat sheet generation completed in {duration:.1f} seconds")
        
        # Save beat sheet
        beat_sheet = result['beat_sheet']
        beats_path = output_dir / "beats.json"
        generator.save_beat_sheet(beat_sheet, beats_path)
        
        # Save metadata
        metadata = {
            "movie_name": args.movie_name,
            "target_genre": args.target_genre,
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
        print(f"Genre: {args.target_genre}")
        print(f"Beat Count: {len(beat_sheet.get('beats', []))}")
        print(f"Duration: {duration:.1f}s")
        print(f"\nOutputs:")
        print(f"  ✓ beats.json: {beats_path}")
        print(f"  ✓ genre_rewrite.json: {output_dir / 'genre_rewrite.json'}")
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
