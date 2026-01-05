#!/usr/bin/env python3
"""
Story Graph Generator
=====================

Standalone utility to generate semantic story graphs from movie synopsis and subtitles.
Uses GPT-4 to extract structured narrative understanding for downstream trailer generation.

Usage:
    python 11_story_graph_generator.py --movie-name "Caddyshack" --synopsis "A comedy about..." --srt-file movie.srt
    python 11_story_graph_generator.py --movie-name "Movie" --synopsis synopsis.txt --srt-file movie.srt --force
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import shutil
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utilities.subtitle_parser import SubtitleParser
from trailer_generator.narrative.azure_client import AzureOpenAIClient
from trailer_generator.analysis.story_graph_generator import StoryGraphGenerator
from pathlib import Path

def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup logging configuration with file and console handlers.

    Args:
        output_dir: Directory where log file will be created.
        verbose: If True, sets log level to DEBUG; otherwise INFO.

    Returns:
        logging.Logger: Configured logger instance for the module.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = output_dir / 'story_graph_generator.log'
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = 'trailer_generator/config/settings.yaml') -> dict:
    """Load configuration from YAML file with environment variable overrides.

    Loads settings from a YAML configuration file and allows Azure OpenAI
    settings to be overridden via environment variables (AZURE_OPENAI_ENDPOINT
    and AZURE_OPENAI_KEY).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary with settings. Returns empty dict if
            loading fails.
    """
    try:
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override Azure OpenAI settings from environment variables if present
        if 'azure_openai' in config:
            # Override endpoint if AZURE_OPENAI_ENDPOINT is set
            env_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            if env_endpoint:
                config['azure_openai']['endpoint'] = env_endpoint
            
            # Override api_key if AZURE_OPENAI_KEY is set
            env_api_key = os.getenv('AZURE_OPENAI_KEY')
            if env_api_key:
                config['azure_openai']['api_key'] = env_api_key
        
        return config
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

def sanitize_filename(name: str) -> str:
    """Sanitize movie name for use as directory name.

    Removes special characters, replaces spaces with underscores, and
    consolidates consecutive underscores.

    Args:
        name: The movie name to sanitize.

    Returns:
        str: Sanitized string safe for use as a directory name.
    """
    # Remove special characters
    sanitized = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized.strip('_')

def load_synopsis(synopsis_input: str) -> str:
    """Load synopsis from string or file.

    Args:
        synopsis_input: Either synopsis text or path to a text file containing
            the synopsis.

    Returns:
        str: The synopsis text.

    Raises:
        SystemExit: If the file exists but cannot be read.
    """
    synopsis_path = Path(synopsis_input)
    
    # Check if it's a file path
    if synopsis_path.exists() and synopsis_path.is_file():
        try:
            with open(synopsis_path, 'r', encoding='utf-8') as f:
                synopsis = f.read().strip()
            print(f"Loaded synopsis from file: {synopsis_path}")
            return synopsis
        except Exception as e:
            print(f"Error reading synopsis file: {e}")
            sys.exit(1)
    else:
        # Treat as inline synopsis text
        return synopsis_input.strip()

def main():
    """Main entry point for the story graph generator CLI.

    Parses command-line arguments, loads synopsis and subtitle files,
    initializes the Azure OpenAI client, generates a semantic story graph
    using hierarchical 3-stage processing, and saves the output files.

    Raises:
        SystemExit: On validation errors, missing configuration, or generation
            failures.
    """
    parser = argparse.ArgumentParser(
        description='Generate semantic story graph from movie synopsis and subtitles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with inline synopsis
  python 11_story_graph_generator.py \\
    --movie-name "Caddyshack" \\
    --synopsis "A comedy about a golf course caddy..." \\
    --srt-file samples/caddyshack.srt
  
  # Using synopsis from text file
  python 11_story_graph_generator.py \\
    --movie-name "Dumb and Dumber" \\
    --synopsis synopsis.txt \\
    --srt-file movie.srt
  
  # Force overwrite existing graph
  python 11_story_graph_generator.py \\
    --movie-name "Movie Title" \\
    --synopsis "..." \\
    --srt-file movie.srt \\
    --force
        """
    )
    
    # Required arguments
    parser.add_argument('--movie-name', type=str, required=True,
                       help='Movie title (e.g., "Caddyshack")')
    parser.add_argument('--synopsis', type=str, required=True,
                       help='Movie synopsis (text string or path to .txt file)')
    parser.add_argument('--srt-file', type=str, required=True,
                       help='Path to SRT subtitle file')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='outputs/story_graphs',
                       help='Output directory (default: outputs/story_graphs)')
    parser.add_argument('--config', type=str, default='trailer_generator/config/settings.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing story graph')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate inputs without generating')
    
    args = parser.parse_args()
    
    # Create output directory
    sanitized_name = sanitize_filename(args.movie_name)
    output_dir = Path(args.output_dir) / sanitized_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.verbose)
    
    print("\n" + "=" * 70)
    print("STORY GRAPH GENERATOR")
    print("=" * 70)
    print(f"Movie: {args.movie_name}")
    print(f"Output: {output_dir}")
    print("=" * 70 + "\n")
    
    # Check if output already exists
    output_file = output_dir / 'story_graph.json'
    if output_file.exists() and not args.force:
        logger.warning(f"Story graph already exists: {output_file}")
        print(f"\n⚠️  Story graph already exists at: {output_file}")
        print("Use --force to overwrite, or delete the file manually.")
        sys.exit(0)
    
    # Load synopsis
    logger.info("Loading synopsis...")
    try:
        synopsis = load_synopsis(args.synopsis)
        logger.info(f"Synopsis loaded: {len(synopsis)} characters")
    except Exception as e:
        logger.error(f"Failed to load synopsis: {e}")
        print(f"\n❌ Error loading synopsis: {e}")
        sys.exit(1)
    
    # Validate synopsis
    if len(synopsis) < 50:
        logger.error("Synopsis too short (minimum 50 characters)")
        print("\n❌ Error: Synopsis must be at least 50 characters")
        sys.exit(1)
    
    # Load subtitles
    logger.info(f"Loading subtitles from: {args.srt_file}")
    subtitle_parser = SubtitleParser(min_dialogue_duration=0.3)
    
    if not subtitle_parser.load_srt(args.srt_file):
        logger.error("Failed to load SRT file")
        print(f"\n❌ Error: Could not load subtitle file: {args.srt_file}")
        print("Make sure the file exists and is in valid SRT format.")
        sys.exit(1)
    
    # Get subtitle statistics
    stats = subtitle_parser.get_statistics()
    logger.info(f"Loaded {stats['total_entries']} subtitle entries")
    logger.info(f"Total words: {stats['total_words']}")
    logger.info(f"Total duration: {stats['total_duration']:.1f} seconds")
    
    print(f"✓ Loaded synopsis: {len(synopsis)} characters")
    print(f"✓ Loaded subtitles: {stats['total_entries']} entries, {stats['total_words']} words")
    
    # Get full transcript
    subtitles_text = subtitle_parser.get_full_transcript(include_timestamps=True)
    logger.info(f"Generated transcript: {len(subtitles_text)} characters")
    
    # Save input files for reference
    logger.info("Saving input files for reference...")
    try:
        # Save synopsis
        with open(output_dir / 'input_synopsis.txt', 'w', encoding='utf-8') as f:
            f.write(synopsis)
        
        # Copy SRT file
        shutil.copy2(args.srt_file, output_dir / 'input_subtitles.srt')
        
        logger.info("Saved input files")
    except Exception as e:
        logger.warning(f"Could not save input files: {e}")
    
    # If validate-only mode, exit here
    if args.validate_only:
        print("\n✓ Validation complete - all inputs valid")
        print(f"\nTo generate story graph, run without --validate-only flag")
        sys.exit(0)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    azure_config = config.get('azure_openai', {})
    if not azure_config:
        logger.error("Azure OpenAI configuration not found")
        print("\n❌ Error: Azure OpenAI configuration missing from settings.yaml")
        sys.exit(1)
    
    # Get story graph configuration
    story_graph_config = config.get('story_graph', {})
    chunk_duration_minutes = story_graph_config.get('chunk_duration_minutes')
    overlap_seconds = story_graph_config.get('overlap_seconds')
    max_parallel_chunks = story_graph_config.get('max_parallel_chunks')
    story_graph_temperature = story_graph_config.get('temperature')
    
    # Get synthesis token limit from Azure config
    synthesis_max_tokens = azure_config.get('max_completion_tokens')
    
    # Initialize Azure client
    logger.info("Initializing Azure OpenAI client...")
    try:
        azure_client = AzureOpenAIClient(
            endpoint=azure_config['endpoint'],
            api_key=azure_config['api_key'],
            deployment_name=azure_config['deployment_name'],
            api_version=azure_config.get('api_version'),
            max_retries=azure_config.get('max_retries'),
            temperature=azure_config.get('temperature'),
            max_completion_tokens=azure_config.get('max_completion_tokens')
        )
        logger.info("Azure client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Azure client: {e}")
        print(f"\n❌ Error initializing Azure OpenAI client: {e}")
        sys.exit(1)
    
    # Initialize story graph generator with cache directory
    logger.info(f"Initializing story graph generator with hierarchical processing...")
    logger.info(f"  Chunk duration: {chunk_duration_minutes} minutes")
    logger.info(f"  Overlap: {overlap_seconds} seconds")
    logger.info(f"  Max parallel chunks: {max_parallel_chunks}")
    logger.info(f"  Story graph temperature: {story_graph_temperature}")
    logger.info(f"  Synthesis max tokens: {synthesis_max_tokens}")
    cache_dir = output_dir / 'chunks'
    story_graph_gen = StoryGraphGenerator(
        azure_client=azure_client,
        chunk_duration_minutes=chunk_duration_minutes,
        overlap_seconds=overlap_seconds,
        max_parallel_chunks=max_parallel_chunks,
        cache_dir=cache_dir,
        synthesis_max_tokens=synthesis_max_tokens,
        temperature=story_graph_temperature
    )
    
    # Generate story graph
    print("\n⏳ Generating story graph using hierarchical 3-stage processing...")
    print("   This may take several minutes depending on movie length.")
    logger.info("Starting story graph generation...")
    
    try:
        story_graph = story_graph_gen.generate_story_graph(
            movie_name=args.movie_name,
            synopsis=synopsis,
            subtitles_text=subtitles_text,
            force_regenerate=args.force
        )
        
        logger.info("Story graph generated successfully")
        
        # Validate output
        if not story_graph_gen.validate_story_graph(story_graph):
            logger.error("Generated story graph failed validation")
            print("\n⚠️  Warning: Story graph validation failed")
            print("The graph may be incomplete or malformed.")
        
    except Exception as e:
        logger.error(f"Story graph generation failed: {e}", exc_info=True)
        print(f"\n❌ Error generating story graph: {e}")
        sys.exit(1)
    
    # Save story graph
    logger.info(f"Saving story graph to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(story_graph, f, indent=2, ensure_ascii=False)
        
        logger.info("Story graph saved successfully")
    except Exception as e:
        logger.error(f"Failed to save story graph: {e}")
        print(f"\n❌ Error saving story graph: {e}")
        sys.exit(1)
    
    # Save metadata
    metadata = {
        'movie_name': args.movie_name,
        'generated_at': datetime.now().isoformat(),
        'synopsis_length': len(synopsis),
        'subtitle_entries': stats['total_entries'],
        'subtitle_words': stats['total_words'],
        'characters_count': len(story_graph.get('characters', [])),
        'scenes_count': len(story_graph.get('scene_timeline', [])),
        'processing_method': 'hierarchical_3_stage',
        'chunks_processed': story_graph.get('_metadata', {}).get('chunks_processed', 0)
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✓ STORY GRAPH GENERATION COMPLETE")
    print("=" * 70)
    print(f"Movie: {args.movie_name}")
    print(f"Processing: Hierarchical 3-stage chunked analysis")
    print(f"Chunks processed: {metadata['chunks_processed']}")
    print(f"Characters: {len(story_graph.get('characters', []))}")
    print(f"Scenes: {len(story_graph.get('scene_timeline', []))}")
    print(f"Themes: {len(story_graph.get('major_themes', []))}")
    print(f"\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {cache_dir}/ (chunk analyses)")
    print(f"  - {output_dir / 'input_synopsis.txt'}")
    print(f"  - {output_dir / 'input_subtitles.srt'}")
    print(f"  - {metadata_file}")
    print(f"\nLog file: {output_dir / 'story_graph_generator.log'}")
    print("\nNote: Chunk analyses are cached. Use --force to regenerate.")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
