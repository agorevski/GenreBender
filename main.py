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
from trailer_generator.ingest import ShotDetector, KeyframeExtractor, BatchProcessor, AudioExtractor, SubtitleExtractor
from trailer_generator.analysis import RemoteAnalyzer, AnalysisCache, GenreScorer, ShotSelector
from trailer_generator.narrative import AzureOpenAIClient, TimelineGenerator
from trailer_generator.checkpoint import CheckpointManager, load_shots_from_metadata, save_shots_to_metadata
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
    parser.add_argument('--resume-from', type=str, choices=['shot_detection', 'keyframe_extraction', 'audio_extraction', 'subtitle_management', 'remote_analysis', 'genre_scoring', 'shot_selection', 'narrative_generation', 'video_assembly', 'audio_mixing'], help='Resume from specific stage')
    parser.add_argument('--skip-clean', action='store_true', help='Skip cleaning output directory (useful for resume)')
    parser.add_argument('--force-stage', type=str, choices=['shot_detection', 'keyframe_extraction', 'audio_extraction', 'subtitle_management', 'remote_analysis', 'genre_scoring', 'shot_selection', 'narrative_generation', 'video_assembly', 'audio_mixing'], help='Force re-run of specific stage')
    parser.add_argument('--reset-checkpoint', action='store_true', help='Reset checkpoint and start fresh')
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
    
    # Define output subdirectories
    shots_dir = output_base / 'shots'
    keyframes_dir = output_base / 'keyframes'
    cache_dir = output_base / 'cache'
    output_dir = output_base / 'output'
    temp_dir = output_base / 'temp'
    log_file = output_base / 'trailer_generator.log'
    checkpoint_file = output_base / 'checkpoint.json'
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager(checkpoint_file)
    
    # Handle checkpoint reset
    if args.reset_checkpoint:
        logger.info("Resetting checkpoint...")
        checkpoint.reset()
        if output_base.exists() and not args.skip_clean:
            logger.info(f"Cleaning output directory: {output_base}")
            shutil.rmtree(output_base)
    
    # Handle directory cleaning
    if not args.skip_clean and not args.resume_from:
        if output_base.exists():
            logger.info(f"Cleaning existing output directory: {output_base}")
            shutil.rmtree(output_base)
            logger.info(f"Removed existing directory: {output_base}")
            checkpoint.reset()
    elif args.resume_from:
        logger.info(f"Resume mode: Preserving existing output directory")
        # Validate resume
        if not checkpoint.validate_resume(args.resume_from, args.input, args.genre):
            logger.error("Resume validation failed. Use --reset-checkpoint to start fresh.")
            sys.exit(1)
    
    # Create output directories
    shots_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set checkpoint metadata
    checkpoint.set_metadata(args.input, args.genre)
    
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
    
    # Determine shot metadata path
    shot_metadata_path = shots_dir / 'shot_metadata.json'
    
    # ========== STEP 1: SHOT DETECTION ==========
    logger.info("=" * 60)
    logger.info("STEP 1: Shot Detection")
    logger.info("=" * 60)
    
    force_shot_detection = args.force_stage == 'shot_detection'
    if checkpoint.should_skip_stage('shot_detection', force=force_shot_detection):
        logger.info("⏭️  Skipping shot detection (already completed)")
        shots = load_shots_from_metadata(shot_metadata_path)
        if not shots:
            logger.error("Failed to load shots from metadata. Cannot skip this stage.")
            sys.exit(1)
    else:
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
        
        # Save and checkpoint
        save_shots_to_metadata(shots, shot_metadata_path, args.input)
        checkpoint.mark_stage_completed('shot_detection', {'shots_count': len(shots)})
    
    # ========== STEP 2: KEYFRAME EXTRACTION ==========
    logger.info("=" * 60)
    logger.info("STEP 2: Keyframe Extraction (Multiple Frames)")
    logger.info("=" * 60)
    
    force_keyframe = args.force_stage == 'keyframe_extraction'
    if checkpoint.should_skip_stage('keyframe_extraction', force=force_keyframe):
        logger.info("⏭️  Skipping keyframe extraction (already completed)")
        shots = load_shots_from_metadata(shot_metadata_path)
    else:
        extractor = KeyframeExtractor(
            output_dir=str(keyframes_dir),
            quality=config['keyframe']['quality']
        )
        
        # Extract 5 frames per shot for temporal analysis
        shots = extractor.extract_keyframes(args.input, shots, num_frames=5)
        logger.info(f"Extracted keyframes for {sum(1 for s in shots if s.get('keyframes'))} shots")
        
        # Save and checkpoint
        save_shots_to_metadata(shots, shot_metadata_path, args.input)
        checkpoint.mark_stage_completed('keyframe_extraction', {
            'frames_extracted': sum(len(s.get('keyframes', [])) for s in shots)
        })
    
    # ========== STEP 3: AUDIO FEATURE EXTRACTION ==========
    logger.info("=" * 60)
    logger.info("STEP 3: Audio Feature Extraction (MFCC & Spectral)")
    logger.info("=" * 60)
    
    force_audio = args.force_stage == 'audio_extraction'
    if checkpoint.should_skip_stage('audio_extraction', force=force_audio):
        logger.info("⏭️  Skipping audio extraction (already completed)")
        shots = load_shots_from_metadata(shot_metadata_path)
    else:
        audio_extractor = AudioExtractor(
            sample_rate=22050,
            n_mfcc=13
        )
        
        shots = audio_extractor.extract_audio_features(args.input, shots)
        logger.info(f"Extracted audio features for {sum(1 for s in shots if s.get('audio_features'))} shots")
        
        # Save and checkpoint
        save_shots_to_metadata(shots, shot_metadata_path, args.input)
        checkpoint.mark_stage_completed('audio_extraction', {
            'features_extracted': sum(1 for s in shots if s.get('audio_features'))
        })
    
    # ========== STEP 4: SUBTITLE MANAGEMENT ==========
    logger.info("=" * 60)
    logger.info("STEP 4: Subtitle Management (SRT Processing)")
    logger.info("=" * 60)
    
    force_subtitle = args.force_stage == 'subtitle_management'
    subtitle_config = config.get('subtitle_management', {})
    
    if checkpoint.should_skip_stage('subtitle_management', force=force_subtitle):
        logger.info("⏭️  Skipping subtitle management (already completed)")
        shots = load_shots_from_metadata(shot_metadata_path)
    else:
        if subtitle_config.get('enabled', True):
            subtitle_extractor = SubtitleExtractor(
                min_dialogue_duration=subtitle_config.get('min_dialogue_duration', 0.3)
            )
            
            # Find SRT file
            srt_path = subtitle_extractor.find_srt_file(args.input, subtitle_config.get('srt_file'))
            
            if srt_path:
                # Load and map subtitles
                subtitles = subtitle_extractor.load_srt(srt_path)
                if subtitles:
                    shots = subtitle_extractor.map_to_shots(subtitles, shots)
                    summary = subtitle_extractor.get_dialogue_summary(shots)
                    logger.info(f"Mapped subtitles: {summary['dialogue_shots']}/{summary['total_shots']} shots have dialogue")
                    logger.info(f"Coverage: {summary['dialogue_coverage']:.1%}, Total words: {summary['total_words']}")
                    
                    # Save and checkpoint
                    save_shots_to_metadata(shots, shot_metadata_path, args.input)
                    checkpoint.mark_stage_completed('subtitle_management', {
                        'subtitles_processed': True,
                        'dialogue_shots': summary['dialogue_shots']
                    })
                else:
                    logger.warning("Failed to load SRT file, continuing without subtitles")
                    # Add empty subtitle data
                    for shot in shots:
                        shot['subtitles'] = {'has_dialogue': False, 'dialogue': None, 'subtitle_entries': [], 'word_count': 0, 'dialogue_density': 0.0, 'emotional_markers': {'questions': 0, 'exclamations': 0, 'all_caps_words': 0}}
                    save_shots_to_metadata(shots, shot_metadata_path, args.input)
                    checkpoint.mark_stage_completed('subtitle_management', {'subtitles_processed': False})
            else:
                if subtitle_config.get('fallback_to_no_subtitles', True):
                    logger.info("No SRT file found, continuing without subtitles")
                    # Add empty subtitle data
                    for shot in shots:
                        shot['subtitles'] = {'has_dialogue': False, 'dialogue': None, 'subtitle_entries': [], 'word_count': 0, 'dialogue_density': 0.0, 'emotional_markers': {'questions': 0, 'exclamations': 0, 'all_caps_words': 0}}
                    save_shots_to_metadata(shots, shot_metadata_path, args.input)
                    checkpoint.mark_stage_completed('subtitle_management', {'subtitles_processed': False})
                else:
                    logger.error("No SRT file found and fallback disabled")
                    sys.exit(1)
        else:
            logger.info("Subtitle processing disabled in configuration")
            # Add empty subtitle data
            for shot in shots:
                shot['subtitles'] = {'has_dialogue': False, 'dialogue': None, 'subtitle_entries': [], 'word_count': 0, 'dialogue_density': 0.0, 'emotional_markers': {'questions': 0, 'exclamations': 0, 'all_caps_words': 0}}
            save_shots_to_metadata(shots, shot_metadata_path, args.input)
            checkpoint.mark_stage_completed('subtitle_management', {'subtitles_processed': False})
    
    # ========== STEP 5: REMOTE ANALYSIS ==========
    force_analysis = args.force_stage == 'remote_analysis'
    logger.info("=" * 60)
    logger.info("STEP 5: Multimodal Analysis (Qwen2-VL)")
    logger.info("=" * 60)
    if not args.skip_analysis and not checkpoint.should_skip_stage('remote_analysis', force=force_analysis):
        
        # Initialize cache
        cache = AnalysisCache(
            cache_dir=str(cache_dir),
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
        
        # Save and checkpoint
        save_shots_to_metadata(shots, shot_metadata_path, args.input)
        checkpoint.mark_stage_completed('remote_analysis', {
            'analyzed_count': sum(1 for s in shots if s.get('analysis'))
        })
    elif checkpoint.should_skip_stage('remote_analysis', force=force_analysis):
        logger.info("⏭️  Skipping remote analysis (already completed)")
        shots = load_shots_from_metadata(shot_metadata_path)
    else:
        logger.info("Skipping analysis (--skip-analysis flag)")
    
    # ========== STEP 6: GENRE SCORING ==========
    logger.info("=" * 60)
    logger.info("STEP 6: Genre-Based Scoring")
    logger.info("=" * 60)
    
    force_scoring = args.force_stage == 'genre_scoring'
    if checkpoint.should_skip_stage('genre_scoring', force=force_scoring):
        logger.info("⏭️  Skipping genre scoring (already completed)")
        shots = load_shots_from_metadata(shot_metadata_path)
    else:
        scorer = GenreScorer(genre_profile['scoring_weights'])
        shots = scorer.score_shots(shots)
        
        # Save and checkpoint
        save_shots_to_metadata(shots, shot_metadata_path, args.input)
        checkpoint.mark_stage_completed('genre_scoring', {
            'scored_count': sum(1 for s in shots if 'genre_score' in s)
        })
    
    # ========== STEP 7: SHOT SELECTION ==========
    logger.info("=" * 60)
    logger.info("STEP 7: Shot Selection")
    logger.info("=" * 60)
    
    force_selection = args.force_stage == 'shot_selection'
    if checkpoint.should_skip_stage('shot_selection', force=force_selection):
        logger.info("⏭️  Skipping shot selection (already completed)")
        # Load top shots from saved file
        top_shots_path = output_dir / 'selected_shots.json'
        if top_shots_path.exists():
            with open(top_shots_path, 'r') as f:
                top_shots = json.load(f)
            logger.info(f"Loaded {len(top_shots)} selected shots from cache")
        else:
            # Fallback: re-run selection
            selector = ShotSelector(
                target_count=config['processing']['shot_candidate_count']
            )
            top_shots = selector.select_top_shots(shots)
    else:
        selector = ShotSelector(
            target_count=config['processing']['shot_candidate_count']
        )
        top_shots = selector.select_top_shots(shots)
        logger.info(f"Selected {len(top_shots)} top shots for trailer")
        
        # Save top shots
        top_shots_path = output_dir / 'selected_shots.json'
        with open(top_shots_path, 'w') as f:
            json.dump(top_shots, f, indent=2)
        
        checkpoint.mark_stage_completed('shot_selection', {
            'selected_count': len(top_shots)
        })
    
    # ========== STEP 8: NARRATIVE GENERATION ==========
    logger.info("=" * 60)
    logger.info("STEP 8: Narrative Structure Generation (Azure OpenAI)")
    logger.info("=" * 60)
    
    timeline_path = output_dir / 'timeline.json'
    force_narrative = args.force_stage == 'narrative_generation'
    
    if checkpoint.should_skip_stage('narrative_generation', force=force_narrative):
        logger.info("⏭️  Skipping narrative generation (already completed)")
        if timeline_path.exists():
            with open(timeline_path, 'r') as f:
                timeline = json.load(f)
            logger.info(f"Loaded timeline with {len(timeline.get('timeline', []))} shots")
        else:
            logger.error("Timeline file not found. Cannot skip this stage.")
            sys.exit(1)
    else:
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
        timeline_gen.export_timeline(timeline, str(timeline_path))
        logger.info(f"Generated timeline with {len(timeline['timeline'])} shots")
        
        checkpoint.mark_stage_completed('narrative_generation', {
            'timeline_shots': len(timeline.get('timeline', [])),
            'duration': timeline.get('total_duration', 0)
        })
    
    # ========== STEP 9: VIDEO ASSEMBLY ==========
    logger.info("=" * 60)
    logger.info("STEP 9: Video Assembly & Color Grading")
    logger.info("=" * 60)
    logger.info("NOTE: Video assembly module to be implemented")
    logger.info("Timeline exported to: output/timeline.json")
    
    # ========== STEP 10: AUDIO MIXING ==========
    logger.info("=" * 60)
    logger.info("STEP 10: Audio Mixing")
    logger.info("=" * 60)
    logger.info("NOTE: Audio mixing module to be implemented")
    
    # ========== COMPLETION ==========
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Timeline saved to: {timeline_path}")
    logger.info(f"Shot metadata: {shot_metadata_path}")
    logger.info(f"Checkpoint: {checkpoint_file}")
    logger.info(f"Logs: {log_file}")
    
    # Display checkpoint stats
    stats = checkpoint.get_stats()
    logger.info(f"Pipeline progress: {stats['completed_stages']}/{stats['total_stages']} stages ({stats['progress_percent']:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ TRAILER GENERATION PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print(f"Timeline: {timeline_path}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Shots analyzed: {len(shots)}")
    print(f"Top shots selected: {len(top_shots)}")
    print(f"Timeline duration: {timeline.get('total_duration', 0):.1f}s")
    print(f"\n✓ Pipeline Progress: {stats['completed_stages']}/{stats['total_stages']} stages completed")
    print("\nCompleted stages:")
    for stage in stats['completed_list']:
        print(f"  ✓ {stage}")
    print("\nNext steps:")
    print("1. Implement video assembly module")
    print("2. Implement audio mixing module")
    print("3. Run final rendering")
    print(f"\nTo resume from a specific stage, use: --resume-from STAGE --skip-clean")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
