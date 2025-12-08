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
from typing import Dict, Tuple
from dotenv import load_dotenv

from trailer_generator.checkpoint import CheckpointManager, load_shots_from_metadata, save_shots_to_metadata

# Load environment variables
load_dotenv()

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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    api_key = config.get('azure_openai', {}).get('api_key', '')
    if api_key.startswith('${') and api_key.endswith('}'):
        env_var = api_key[2:-1]
        config['azure_openai']['api_key'] = os.getenv(env_var)
    
    return config

def load_genre_profile(genre: str, 
                      profile_path: str = 'trailer_generator/config/genre_profiles.yaml') -> Dict:
    """Load genre-specific configuration."""
    with open(profile_path, 'r') as f:
        profiles = yaml.safe_load(f)
    
    profile = profiles.get(genre.lower(), profiles.get('thriller'))
    # Add genre name to profile for easy reference
    profile['name'] = genre.lower()
    return profile

def setup_directories(output_base: Path) -> Dict[str, Path]:
    """
    Create and return all output directories.
    
    Args:
        output_base: Base output directory
        
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
    
    for key, path in dirs.items():
        if key not in ['log_file', 'checkpoint_file']:
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

def initialize_stage(stage_name: str, input_path: str, genre: str) -> Tuple[Path, Dict, CheckpointManager, logging.Logger]:
    """
    Common initialization for all pipeline stages.
    
    Args:
        stage_name: Name of the current stage
        input_path: Path to input video file
        genre: Target genre
        
    Returns:
        Tuple of (output_base, directories, checkpoint, logger)
    """
    # Validate input
    if not validate_input_file(input_path):
        sys.exit(1)
    
    # Get output directories
    output_base = get_output_base_dir(input_path)
    dirs = setup_directories(output_base)
    
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
    logger.info(f"Genre: {genre}")
    logger.info(f"Output directory: {output_base}")
    
    return output_base, dirs, checkpoint, logger

def print_completion_message(stage_name: str, checkpoint: CheckpointManager, output_base: Path):
    """
    Print completion message for a stage.
    
    Args:
        stage_name: Name of completed stage
        checkpoint: Checkpoint manager
        output_base: Base output directory
    """
    stats = checkpoint.get_stats()
    
    print("\n" + "=" * 60)
    print(f"✓ {stage_name.upper()} COMPLETED")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print(f"Progress: {stats['completed_stages']}/{stats['total_stages']} stages ({stats['progress_percent']:.1f}%)")
    print("\nCompleted stages:")
    for stage in stats['completed_list']:
        print(f"  ✓ {stage}")
    
    # Suggest next stage
    if stats['completed_stages'] < stats['total_stages']:
        current_idx = CheckpointManager.STAGES.index(stage_name)
        if current_idx + 1 < len(CheckpointManager.STAGES):
            next_stage = CheckpointManager.STAGES[current_idx + 1]
            script_num = current_idx + 2
            print(f"\nNext step: Run {script_num}_{next_stage}.py")
    else:
        print("\n✓ All stages completed!")

def add_common_arguments(parser: argparse.ArgumentParser):
    """
    Add common command-line arguments to parser.
    
    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--genre', type=str, default='thriller',
                       choices=['thriller', 'action', 'drama', 'horror', 'scifi', 'comedy', 'romance'],
                       help='Trailer genre/style')
    parser.add_argument('--config', type=str, 
                       default='trailer_generator/config/settings.yaml',
                       help='Configuration file path')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (limited shots)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run this stage even if completed')
