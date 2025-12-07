"""
Main orchestrator for Automated Trailer Generator.
Command-line interface for generating trailers from full-length movies.
"""

import argparse
import logging
import yaml
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Import modules
from trailer_generator.ingest import ShotDetector, KeyframeExtractor, BatchProcessor, AudioExtractor
from trailer_generator.analysis import RemoteAnalyzer, AnalysisCache, GenreScorer, ShotSelector
from trailer_generator.narrative import AzureOpenAIClient, TimelineGenerator
import shutil

# Load environment variables
load_dotenv()

def setup_logging(config: Dict):
    """Setup logging configuration."""
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('file', 'trailer_generator.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
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
    api_key = config['azure_openai']['api_key']
    if api_key.startswith('${') and api_key.endswith('}'):
        env_var = api_key[2:-1]
        config['azure_openai']['api_key'] = os.getenv(env_var)
    
    return config

def load_genre_profile(genre: str, 
                      profile_path: str = 'trailer_generator/config/genre_profiles.yaml') -> Dict:
    """Load genre-specific configuration."""
    with open(profile_path, 'r') as f:
        profiles = yaml.safe_load(f)
    
    return profiles.get(genre.lower(), profiles.get('thriller'))

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for filesystem compatibility.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for directory names
    """
    # Remove extension
    name = Path(filename).stem
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^\w\-]', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Fallback if empty
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

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(description='Automated Trailer Generator - Create cinematic trailers from movies')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--genre', type=str, default='thriller', choices=['thriller', 'action', 'drama', 'horror', 'scifi', 'comedy', 'romance'], help='Trailer genre/style')
    parser.add_argument('--output', type=str, default='output/trailer.mp4', help='Output trailer path')
    parser.add_argument('--config', type=str, default='trailer_generator/config/settings.yaml', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run in test mode with sample validation')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip remote analysis (use cached results)')
    parser.add_argument('--no-cache', action='store_true', help='Disable analysis caching')
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"Args:\t{key}: {value}")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    for key, value in config.items():
        print(f"Config:\t{key}: {value}")
        
    genre_profile = load_genre_profile(args.genre)
    
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Automated Trailer Generator")
    logger.info(f"Input: {args.input}")
    logger.info(f"Genre: {args.genre}")
    logger.info(f"Output: {args.output}")
    
    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Get base output directory based on input filename
    output_base = get_output_base_dir(args.input)
    logger.info(f"Output directory: {output_base}")
    
    # Clean output directory if it exists
    if output_base.exists():
        logger.info(f"Cleaning existing output directory: {output_base}")
        shutil.rmtree(output_base)
        logger.info(f"Removed existing directory: {output_base}")
    
    # Define output subdirectories
    shots_dir = output_base / 'shots'
    keyframes_dir = output_base / 'keyframes'
    output_dir = output_base / 'output'
    temp_dir = output_base / 'temp'
    log_file = output_base / 'trailer_generator.log'
    
    # Create output directories
    shots_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with dynamic paths
    config['project']['output_dir'] = str(output_dir)
    config['project']['temp_dir'] = str(temp_dir)
    config['logging']['file'] = str(log_file)
    
    # Reconfigure logging with new path
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=getattr(logging, config.get('logging', {}).get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    # ========== STEP 1: SHOT DETECTION ==========
    logger.info("=" * 60)
    logger.info("STEP 1: Shot Detection")
    logger.info("=" * 60)
    
    detector = ShotDetector(
        threshold=config['shot_detection']['threshold'],
        chunk_duration=config['processing']['chunk_duration'],
        overlap=config['processing']['overlap'],
        output_dir=str(shots_dir)
    )
    
    shots = detector.detect_shots(args.input, streaming=True)
    logger.info(f"Detected {len(shots)} shots")
    
    # Limit to first 5 shots in test mode
    if args.test:
        original_count = len(shots)
        shots = shots[:5]
        logger.info(f"TEST MODE: Limited from {original_count} to {len(shots)} shots")
    
    # ========== STEP 2: KEYFRAME EXTRACTION ==========
    logger.info("=" * 60)
    logger.info("STEP 2: Keyframe Extraction (Multiple Frames)")
    logger.info("=" * 60)
    
    extractor = KeyframeExtractor(
        output_dir=str(keyframes_dir),
        quality=config['keyframe']['quality']
    )
    
    # Extract 5 frames per shot for temporal analysis
    shots = extractor.extract_keyframes(args.input, shots, num_frames=5)
    logger.info(f"Extracted keyframes for {sum(1 for s in shots if s.get('keyframes'))} shots")
    
    # ========== STEP 3: AUDIO FEATURE EXTRACTION ==========
    logger.info("=" * 60)
    logger.info("STEP 3: Audio Feature Extraction (MFCC & Spectral)")
    logger.info("=" * 60)
    
    audio_extractor = AudioExtractor(
        sample_rate=22050,
        n_mfcc=13
    )
    
    shots = audio_extractor.extract_audio_features(args.input, shots)
    logger.info(f"Extracted audio features for {sum(1 for s in shots if s.get('audio_features'))} shots")
    
    # Save shot metadata with audio features
    shot_metadata = {
        'source_video': args.input,
        'total_shots': len(shots),
        'shots': shots
    }
    
    shot_metadata_path = shots_dir / 'shot_metadata.json'
    with open(shot_metadata_path, 'w') as f:
        json.dump(shot_metadata, f, indent=2)
    logger.info(f"Saved shot metadata with audio features to {shot_metadata_path}")
    
    # ========== STEP 4: REMOTE ANALYSIS ==========
    if not args.skip_analysis:
        logger.info("=" * 60)
        logger.info("STEP 4: Multimodal Analysis (Qwen2-VL)")
        logger.info("=" * 60)
        
        # Initialize cache
        cache = AnalysisCache(
            cache_dir=config['remote_analysis']['cache_path'],
            enabled=not args.no_cache
        )
        
        # Check cache first
        cached_shots, uncached_shots = cache.get_batch(shots)
        logger.info(f"Cache: {len(cached_shots)} cached, {len(uncached_shots)} to analyze")
        
        if uncached_shots:
            # Initialize remote analyzer
            analyzer = RemoteAnalyzer(
                server_url=config['remote_analysis']['qwen_server_url'],
                timeout=config['remote_analysis']['timeout'],
                max_retries=config['remote_analysis']['max_retries'],
                batch_size=config['remote_analysis']['batch_size'],
                api_key=config['remote_analysis'].get('api_key')
            )
            
            # Check server health
            if not analyzer.health_check():
                logger.warning("Qwen2-VL server not responding. Using fallback analysis.")
                # Could add fallback logic here
            
            # Batch process uncached shots
            batch_processor = BatchProcessor(
                batch_size=config['processing']['max_batch_size']
            )
            
            analyzed_shots = []
            for batch in batch_processor.batch_shots(uncached_shots):
                batch_results = analyzer.analyze_batch(batch, args.input)
                analyzed_shots.extend(batch_results)
                
                # Save partial results
                partial_results_path = temp_dir / 'partial_analysis.json'
                batch_processor.save_partial_results(
                    analyzed_shots,
                    str(partial_results_path)
                )
            
            # Cache new results
            cache.put_batch(analyzed_shots)
            
            # Combine with cached
            shots = cached_shots + analyzed_shots
        else:
            shots = cached_shots
    else:
        logger.info("Skipping analysis (using cached results)")
    
    # ========== STEP 5: GENRE SCORING ==========
    logger.info("=" * 60)
    logger.info("STEP 5: Genre-Based Scoring")
    logger.info("=" * 60)
    
    scorer = GenreScorer(genre_profile['scoring_weights'])
    shots = scorer.score_shots(shots)
    
    # ========== STEP 6: SHOT SELECTION ==========
    logger.info("=" * 60)
    logger.info("STEP 6: Shot Selection")
    logger.info("=" * 60)
    
    selector = ShotSelector(
        target_count=config['processing']['shot_candidate_count']
    )
    
    top_shots = selector.select_top_shots(shots)
    logger.info(f"Selected {len(top_shots)} top shots for trailer")
    
    # ========== STEP 7: NARRATIVE GENERATION ==========
    logger.info("=" * 60)
    logger.info("STEP 7: Narrative Structure Generation (Azure OpenAI)")
    logger.info("=" * 60)
    
    azure_client = AzureOpenAIClient(
        endpoint=config['azure_openai']['endpoint'],
        api_key=config['azure_openai']['api_key'],
        deployment_name=config['azure_openai']['deployment_name'],
        api_version=config['azure_openai']['api_version'],
        max_retries=config['azure_openai']['max_retries'],
        temperature=config['azure_openai']['temperature'],
        max_completion_tokens=config['azure_openai']['max_completion_tokens']
    )
    
    timeline_gen = TimelineGenerator(
        azure_client=azure_client,
        genre=args.genre
    )
    
    timeline = timeline_gen.generate_timeline(
        top_shots,
        target_duration=config['processing']['target_trailer_length']
    )
    
    # Export timeline
    timeline_path = output_dir / 'timeline.json'
    timeline_gen.export_timeline(timeline, str(timeline_path))
    logger.info(f"Generated timeline with {len(timeline['timeline'])} shots")
    
    # ========== STEP 8: VIDEO ASSEMBLY ==========
    logger.info("=" * 60)
    logger.info("STEP 8: Video Assembly & Color Grading")
    logger.info("=" * 60)
    logger.info("NOTE: Video assembly module to be implemented")
    logger.info("Timeline exported to: output/timeline.json")
    
    # ========== STEP 9: AUDIO MIXING ==========
    logger.info("=" * 60)
    logger.info("STEP 9: Audio Mixing")
    logger.info("=" * 60)
    logger.info("NOTE: Audio mixing module to be implemented")
    
    # ========== COMPLETION ==========
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Timeline saved to: {timeline_path}")
    logger.info(f"Shot metadata: {shot_metadata_path}")
    logger.info(f"Logs: {log_file}")
    
    print("\n" + "=" * 60)
    print("✓ TRAILER GENERATION PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print(f"Timeline: {timeline_path}")
    print(f"Shots analyzed: {len(shots)}")
    print(f"Top shots selected: {len(top_shots)}")
    print(f"Timeline duration: {timeline.get('total_duration', 0):.1f}s")
    print("\nNext steps:")
    print("1. Implement video assembly module")
    print("2. Implement audio mixing module")
    print("3. Run final rendering")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
