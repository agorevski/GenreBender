"""
Shared utilities for pipeline stage scripts.
Common functions used across all pipeline stages.
"""

import argparse
import logging
import yaml
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dotenv import load_dotenv

from trailer_generator.checkpoint import CheckpointManager, load_shots_from_metadata, save_shots_to_metadata

# Load environment variables
load_dotenv()

# All supported genres
ALL_GENRES = [
    'comedy', 'horror', 'thriller', 'parody', 'mockumentary',
    'crime', 'drama', 'experimental', 'fantasy', 'romance', 'scifi', 'action'
]

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for filesystem compatibility.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for directory names
    """
    name = Path(filename).stem
    name = re.sub(r'[^\w\-]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name if name else 'output'

def get_output_base_dir(input_path: str) -> Path:
    """
    Get the base output directory for a given input file.
    
    Args:
        input_path: Path to input video file
        
    Returns:
        Path object for base output directory: outputs/<sanitized_filename>/
    """
    filename = Path(input_path).name
    sanitized = sanitize_filename(filename)
    return Path('outputs') / sanitized

def get_genre_output_dir(input_path: str, genre: str) -> Path:
    """
    Get the genre-specific output directory for a given input file and genre.
    
    Args:
        input_path: Path to input video file
        genre: Target genre
        
    Returns:
        Path object for genre output directory: outputs/<sanitized_filename>/trailers/<genre>/
    """
    base_dir = get_output_base_dir(input_path)
    return base_dir / 'trailers' / genre.lower()

def get_story_graph_dir(movie_name: str) -> Path:
    """
    Get the story graph output directory for a given movie name.
    
    Args:
        movie_name: Movie name (will be sanitized)
        
    Returns:
        Path object for story graph directory: outputs/story_graphs/<sanitized_name>/
    """
    sanitized = sanitize_filename(movie_name)
    return Path('outputs') / 'story_graphs' / sanitized

def setup_logging(log_file: Path, level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = 'trailer_generator/config/settings.yaml') -> Dict:
    """Load configuration from YAML file with environment variable overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override Azure OpenAI settings from environment variables if present
    if 'azure_openai' in config:
        # Override endpoint if AZURE_OPENAI_ENDPOINT is set
        env_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
        if env_endpoint:
            config['azure_openai']['endpoint'] = env_endpoint
        
        # Override api_key if AZURE_OPENAI_KEY is set
        env_api_key = os.getenv('AZURE_OPENAI_KEY', None)
        if env_api_key:
            config['azure_openai']['api_key'] = env_api_key
        # Fallback: Replace ${VAR} style references (legacy support)
        elif config['azure_openai'].get('api_key', '').startswith('${'):
            api_key = config['azure_openai']['api_key']
            env_var = api_key[2:-1]
            config['azure_openai']['api_key'] = os.getenv(env_var)
    
    return config

def load_pipeline_config(config_path: str = 'config.yaml') -> Dict:
    """
    Load pipeline configuration from the main config.yaml file.
    
    Args:
        config_path: Path to main config.yaml
        
    Returns:
        Full pipeline configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_movie_config(movie_key: str, config_path: str = 'config.yaml') -> Dict:
    """
    Get configuration for a specific movie from config.yaml.
    
    Args:
        movie_key: Key identifying the movie in config.yaml
        config_path: Path to config.yaml
        
    Returns:
        Movie configuration dictionary
    """
    config = load_pipeline_config(config_path)
    movies = config.get('movies', {})
    if movie_key not in movies:
        raise ValueError(f"Movie key '{movie_key}' not found in {config_path}")
    return movies[movie_key]

def load_genre_profile(genre: str, 
                      profile_path: str = 'trailer_generator/config/genre_profiles.yaml') -> Dict:
    """Load genre-specific configuration."""
    with open(profile_path, 'r') as f:
        profiles = yaml.safe_load(f)
    
    profile = profiles.get(genre.lower(), profiles.get('thriller'))
    # Add genre name to profile for easy reference
    profile['name'] = genre.lower()
    return profile

def setup_directories(output_base: Path, genre: Optional[str] = None) -> Dict[str, Path]:
    """
    Create and return all output directories.
    
    Args:
        output_base: Base output directory
        genre: Optional genre for genre-specific output subdirectory
        
    Returns:
        Dictionary of directory paths
    """
    dirs = {
        'base': output_base,
        'shots': output_base / 'shots',
        'keyframes': output_base / 'keyframes',
        'cache': output_base / 'cache',
        'output': output_base / 'output',
        'temp': output_base / 'temp',
        'log_file': output_base / 'trailer_generator.log',
        'checkpoint_file': output_base / 'checkpoint.json'
    }
    
    # Add genre-specific directories if genre is specified
    if genre:
        genre_dir = output_base / 'trailers' / genre.lower()
        dirs['genre_output'] = genre_dir
        dirs['genre_embeddings'] = genre_dir / 'embeddings'
    
    for key, path in dirs.items():
        if key not in ['log_file', 'checkpoint_file']:
            path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def setup_genre_directories(output_base: Path, genre: str) -> Dict[str, Path]:
    """
    Create and return genre-specific output directories.
    
    Args:
        output_base: Base output directory
        genre: Target genre
        
    Returns:
        Dictionary of genre-specific directory paths
    """
    genre_dir = output_base / 'trailers' / genre.lower()
    
    dirs = {
        'genre_base': genre_dir,
        'genre_embeddings': genre_dir / 'embeddings',
    }
    
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def validate_input_file(input_path: str) -> bool:
    """
    Validate that input file exists.
    
    Args:
        input_path: Path to input video file
        
    Returns:
        True if file exists
    """
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}")
        return False
    return True

def initialize_stage(stage_name: str, input_path: str, genre: Optional[str] = None) -> Tuple[Path, Dict, CheckpointManager, logging.Logger]:
    """
    Common initialization for all pipeline stages.
    
    Args:
        stage_name: Name of the current stage
        input_path: Path to input video file
        genre: Target genre (optional for genre-agnostic stages)
        
    Returns:
        Tuple of (output_base, directories, checkpoint, logger)
    """
    # Validate input
    if not validate_input_file(input_path):
        sys.exit(1)
    
    # Get output directories
    output_base = get_output_base_dir(input_path)
    dirs = setup_directories(output_base, genre)
    
    # Setup logging
    setup_logging(dirs['log_file'])
    logger = logging.getLogger(__name__)
    
    # Initialize checkpoint
    checkpoint = CheckpointManager(dirs['checkpoint_file'])
    checkpoint.set_metadata(input_path, genre)
    
    logger.info("=" * 60)
    logger.info(f"STAGE: {stage_name}")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    if genre:
        logger.info(f"Genre: {genre}")
    else:
        logger.info("Genre: N/A (genre-agnostic stage)")
    logger.info(f"Output directory: {output_base}")
    
    return output_base, dirs, checkpoint, logger

def initialize_genre_stage(stage_name: str, input_path: str, genre: str) -> Tuple[Path, Dict, CheckpointManager, logging.Logger]:
    """
    Initialize a genre-dependent pipeline stage with genre-specific directories.
    
    Args:
        stage_name: Name of the current stage
        input_path: Path to input video file
        genre: Target genre (required)
        
    Returns:
        Tuple of (output_base, directories, checkpoint, logger)
    """
    # Validate input
    if not validate_input_file(input_path):
        sys.exit(1)
    
    if not genre:
        print("ERROR: Genre is required for this stage")
        sys.exit(1)
    
    # Get output directories
    output_base = get_output_base_dir(input_path)
    dirs = setup_directories(output_base)
    
    # Add genre-specific directories
    genre_dirs = setup_genre_directories(output_base, genre)
    dirs.update(genre_dirs)
    
    # Setup logging
    setup_logging(dirs['log_file'])
    logger = logging.getLogger(__name__)
    
    # Initialize checkpoint
    checkpoint = CheckpointManager(dirs['checkpoint_file'])
    checkpoint.set_metadata(input_path, genre)
    
    logger.info("=" * 60)
    logger.info(f"STAGE: {stage_name} (genre: {genre})")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Genre: {genre}")
    logger.info(f"Output directory: {output_base}")
    logger.info(f"Genre output: {dirs['genre_base']}")
    
    return output_base, dirs, checkpoint, logger

def print_completion_message(stage_name: str, checkpoint: CheckpointManager, output_base: Path, genre: Optional[str] = None):
    """
    Print completion message for a stage.
    
    Args:
        stage_name: Name of completed stage
        checkpoint: Checkpoint manager
        output_base: Base output directory
        genre: Optional genre for genre-specific stages
    """
    stats = checkpoint.get_stats()
    
    print("\n" + "=" * 60)
    if genre:
        print(f"✓ {stage_name.upper()} [{genre.upper()}] COMPLETED")
    else:
        print(f"✓ {stage_name.upper()} COMPLETED")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print(f"Progress: {stats['completed_stages']}/{stats['total_stages']} stages ({stats['progress_percent']:.1f}%)")
    print("\nCompleted stages:")
    for stage in stats['completed_list']:
        print(f"  ✓ {stage}")
    
    # Suggest next stage
    if stats['completed_stages'] < stats['total_stages']:
        current_idx = CheckpointManager.STAGES.index(stage_name) if stage_name in CheckpointManager.STAGES else -1
        if current_idx >= 0 and current_idx + 1 < len(CheckpointManager.STAGES):
            next_stage = CheckpointManager.STAGES[current_idx + 1]
            script_num = current_idx + 2
            print(f"\nNext step: Run {script_num}_{next_stage}.py")
    else:
        print("\n✓ All stages completed!")

def add_common_arguments(parser: argparse.ArgumentParser):
    """
    Add common command-line arguments for genre-agnostic stages.
    Does NOT include --genre argument.
    
    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--config', type=str, 
                       default='trailer_generator/config/settings.yaml',
                       help='Configuration file path')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (limited shots)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run this stage even if completed')

def add_genre_arguments(parser: argparse.ArgumentParser):
    """
    Add command-line arguments for genre-dependent stages.
    Includes --genre argument as required.
    
    Args:
        parser: ArgumentParser instance
    """
    add_common_arguments(parser)
    parser.add_argument('--genre', type=str, required=True,
                       choices=ALL_GENRES,
                       help='Target trailer genre (required)')

def add_common_arguments_with_optional_genre(parser: argparse.ArgumentParser):
    """
    Add common command-line arguments with optional genre.
    For backward compatibility with scripts that may still use --genre.
    
    Args:
        parser: ArgumentParser instance
    """
    add_common_arguments(parser)
    parser.add_argument('--genre', type=str, default=None,
                       choices=ALL_GENRES,
                       help='Trailer genre/style (optional for this stage)')

# Legacy function for backward compatibility
def add_common_arguments_legacy(parser: argparse.ArgumentParser):
    """
    Add common command-line arguments to parser (legacy version with genre).
    DEPRECATED: Use add_common_arguments() or add_genre_arguments() instead.
    
    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--genre', type=str, default='thriller',
                       choices=ALL_GENRES,
                       help='Trailer genre/style')
    parser.add_argument('--config', type=str, 
                       default='trailer_generator/config/settings.yaml',
                       help='Configuration file path')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (limited shots)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run this stage even if completed')

def get_genre_filename(base_name: str, genre: str, extension: str) -> str:
    """
    Generate a genre-specific filename.
    
    Args:
        base_name: Base filename without extension (e.g., 'trailer', 'beats', 'timeline')
        genre: Target genre
        extension: File extension including dot (e.g., '.mp4', '.json')
        
    Returns:
        Genre-specific filename (e.g., 'trailer_comedy_final.mp4')
    """
    return f"{base_name}_{genre.lower()}{extension}"
