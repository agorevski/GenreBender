#!/usr/bin/env python3
"""
Stage 8: Video Assembly & Color Grading
Assembles final trailer video from timeline with genre-specific color grading and transitions.
"""

import argparse
import sys
import json
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config, load_genre_profile, load_shots_from_metadata
)
from trailer_generator.assembly import VideoAssembler
from trailer_generator.narrative import AzureOpenAIClient

def main():
    parser = argparse.ArgumentParser(
        description='Stage 8: Video Assembly - Create final trailer video with color grading'
    )
    add_common_arguments(parser)
    parser.add_argument('--no-color-grade', action='store_true',
                       help='Skip color grading (faster for testing)')
    parser.add_argument('--no-transitions', action='store_true',
                       help='Skip transitions between shots')
    parser.add_argument('--no-ai-titles', action='store_true',
                       help='Disable AI title generation')
    parser.add_argument('--no-ai-transitions', action='store_true',
                       help='Use rule-based transitions instead of AI')
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'video_assembly', args.input, args.genre
    )
    
    # Validate prerequisites (stages 1-7)
    required_stages = ['shot_detection', 'keyframe_extraction', 'audio_extraction',
                      'remote_analysis', 'genre_scoring', 'shot_selection',
                      'narrative_generation']
    for stage in required_stages:
        if not checkpoint.is_stage_completed(stage):
            logger.error(f"❌ Prerequisite stage '{stage}' not completed.")
            print(f"\n❌ Error: You must complete all previous stages first!")
            print(f"Missing: {stage}")
            print(f"Run: {required_stages.index(stage) + 1}_{stage}.py")
            sys.exit(1)
    
    # Check if already completed
    assembled_video_path = dirs['output'] / 'trailer_assembled.mp4'
    if checkpoint.is_stage_completed('video_assembly') and not args.force:
        logger.warning("⚠️  Video assembly already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print(f"Assembled video: {assembled_video_path}")
        if assembled_video_path.exists():
            file_size = assembled_video_path.stat().st_size / (1024*1024)
            print(f"File size: {file_size:.1f} MB")
        print("\nUse --force to re-run, or proceed to: 9_audio_mixing.py")
        sys.exit(0)
    
    # Load configuration and genre profile
    config = load_config(args.config)
    genre_profile = load_genre_profile(args.genre)
    
    # Load timeline
    timeline_path = dirs['output'] / 'timeline.json'
    if not timeline_path.exists():
        logger.error("Timeline not found. Run stage 7 first.")
        print("\n❌ Error: Timeline not found. Run 7_narrative_generation.py first!")
        sys.exit(1)
    
    with open(timeline_path, 'r') as f:
        timeline = json.load(f)
    
    logger.info(f"Loaded timeline with {len(timeline.get('timeline', []))} shots")
    logger.info(f"Timeline duration: {timeline.get('total_duration', 0):.1f}s")
    
    # Load shot metadata (needed for transitions)
    shot_metadata_path = dirs['shots'] / 'shot_metadata.json'
    shot_metadata = load_shots_from_metadata(shot_metadata_path)
    if not shot_metadata:
        logger.warning("Could not load shot metadata, transitions may be affected")
    
    # Override AI settings if flags provided
    if args.no_ai_titles:
        config['video']['ai_title_generation'] = False
        logger.info("AI title generation disabled via --no-ai-titles")
    
    if args.no_ai_transitions:
        config['video']['ai_transition_selection'] = False
        logger.info("AI transition selection disabled via --no-ai-transitions")
    
    # Initialize Azure OpenAI client (if AI features enabled)
    azure_client = None
    if config.get('video', {}).get('ai_title_generation') or \
       config.get('video', {}).get('ai_transition_selection'):
        try:
            azure_client = AzureOpenAIClient(
                endpoint=config['azure_openai']['endpoint'],
                api_key=config['azure_openai']['api_key'],
                deployment_name=config['azure_openai']['deployment_name'],
                api_version=config['azure_openai']['api_version'],
                max_retries=config['azure_openai'].get('max_retries', 3),
                temperature=config['azure_openai'].get('temperature', 0.7),
                max_completion_tokens=config['azure_openai'].get('max_completion_tokens', 4000)
            )
            logger.info("Azure OpenAI client initialized for AI features")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI: {e}")
            logger.warning("Continuing without AI features")
            azure_client = None
    
    # Initialize video assembler
    assembler = VideoAssembler(
        config=config,
        genre_profile=genre_profile,
        output_dir=output_base,
        enable_color_grading=not args.no_color_grade,
        enable_transitions=not args.no_transitions
    )
    
    # Log assembly settings
    logger.info("=" * 60)
    logger.info("Video Assembly Settings:")
    logger.info(f"  Color grading: {'enabled' if not args.no_color_grade else 'disabled'}")
    logger.info(f"  Transitions: {'enabled' if not args.no_transitions else 'disabled'}")
    logger.info(f"  AI titles: {'enabled' if config.get('video', {}).get('ai_title_generation') else 'disabled'}")
    logger.info(f"  AI transitions: {'enabled' if config.get('video', {}).get('ai_transition_selection') else 'disabled'}")
    logger.info(f"  Resolution: {config.get('video', {}).get('resolution')}")
    logger.info(f"  FPS: {config.get('video', {}).get('fps')}")
    logger.info(f"  Codec: {config.get('video', {}).get('codec')}")
    logger.info("=" * 60)
    
    # Assemble video
    logger.info(f"Assembling {args.genre} trailer video...")
    print(f"\n⏳ Assembling video... This may take a few minutes.")
    
    try:
        assembled_video = assembler.assemble_video(
            timeline=timeline,
            shots_dir=dirs['shots'],
            output_path=assembled_video_path,
            shot_metadata=shot_metadata,
            azure_client=azure_client
        )
        
        logger.info(f"Video assembled: {assembled_video}")
        logger.info(f"Duration: {timeline.get('total_duration', 0):.1f}s")
        
        # Get file size
        file_size_mb = Path(assembled_video).stat().st_size / (1024*1024)
        logger.info(f"File size: {file_size_mb:.1f} MB")
        
        # Mark stage completed
        checkpoint.mark_stage_completed('video_assembly', {
            'output_file': str(assembled_video),
            'duration': timeline.get('total_duration', 0),
            'shots_count': len(timeline.get('timeline', [])),
            'file_size_mb': round(file_size_mb, 2)
        })
        
        # Print completion
        stats = checkpoint.get_stats()
        
        print("\n" + "=" * 60)
        print("✓ VIDEO ASSEMBLY COMPLETED")
        print("=" * 60)
        print(f"Assembled video: {assembled_video}")
        print(f"Duration: {timeline.get('total_duration', 0):.1f}s")
        print(f"Shots: {len(timeline.get('timeline', []))}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Genre: {args.genre}")
        print(f"\n✓ Pipeline Progress: {stats['completed_stages']}/{stats['total_stages']} stages ({stats['progress_percent']:.1f}%)")
        print("\nCompleted stages:")
        for stage in stats['completed_list']:
            print(f"  ✓ {stage}")
        
        print(f"\nNext step: Run 9_audio_mixing.py")
        print(f"\nTo preview video: ffplay {assembled_video}")
        
    except Exception as e:
        logger.error(f"Video assembly failed: {e}", exc_info=True)
        print(f"\n❌ Error: Video assembly failed!")
        print(f"Details: {e}")
        print(f"\nCheck logs: {dirs['log_file']}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        import logging
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
