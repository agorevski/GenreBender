#!/usr/bin/env python3
"""
Stage 9: Audio Mixing
Adds music, sound effects, and audio ducking to create the final trailer.
"""

import argparse
import sys
import json
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_genre_arguments,
    load_config, load_genre_profile
)
from trailer_generator.audio import AudioMixer
from trailer_generator.narrative import AzureOpenAIClient

def main():
    """Run the audio mixing pipeline stage.

    This function orchestrates Stage 10 of the trailer generation pipeline,
    which adds music, sound effects, and audio ducking to create the final
    trailer. It validates prerequisites, loads configuration and timeline,
    initializes the audio mixer, and produces the final mixed trailer.

    The function supports AI-powered music selection when configured with
    Azure OpenAI, and allows manual music file specification via command
    line arguments.

    Raises:
        SystemExit: If prerequisites are not met, required files are missing,
            or audio mixing fails.
    """
    parser = argparse.ArgumentParser(
        description='Stage 10: Audio Mixing - Add music and effects to create final trailer'
    )
    add_genre_arguments(parser)
    parser.add_argument('--music-file', type=str,
                       help='Specific music file to use (optional)')
    parser.add_argument('--no-ducking', action='store_true',
                       help='Disable audio ducking')
    parser.add_argument('--no-ai-music', action='store_true',
                       help='Disable AI music selection')
    parser.add_argument('--output-name', type=str, default='trailer_final.mp4',
                       help='Output filename (default: trailer_final.mp4)')
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'audio_mixing', args.input, args.genre
    )
    
    # Validate prerequisites (genre-agnostic stages)
    agnostic_stages = ['shot_detection', 'keyframe_extraction', 'audio_extraction',
                       'subtitle_management', 'remote_analysis']
    for stage in agnostic_stages:
        if not checkpoint.is_stage_completed(stage):
            logger.error(f"❌ Prerequisite stage '{stage}' not completed.")
            print(f"\n❌ Error: You must complete all previous stages first!")
            print(f"Missing: {stage}")
            sys.exit(1)
    
    # Validate genre-dependent prerequisites
    # Reload checkpoint to get latest state (important for parallel execution)
    checkpoint.reload()
    
    genre_stages = ['timeline_construction', 'video_assembly']
    for stage in genre_stages:
        if not checkpoint.is_stage_completed(stage, args.genre):
            logger.error(f"❌ Prerequisite stage '{stage}' for genre '{args.genre}' not completed.")
            print(f"\n❌ Error: You must complete stage '{stage}' for genre '{args.genre}' first!")
            sys.exit(1)
    
    # Genre-specific output path
    genre_output_dir = dirs.get('genre_base', dirs['output'])
    genre_output_dir.mkdir(parents=True, exist_ok=True)
    final_trailer_path = genre_output_dir / f'trailer_{args.genre}_final.mp4'
    
    # Check if already completed for this genre
    if checkpoint.is_stage_completed('audio_mixing', args.genre) and not args.force:
        logger.warning("⚠️  Audio mixing already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print(f"Final trailer: {final_trailer_path}")
        if final_trailer_path.exists():
            file_size = final_trailer_path.stat().st_size / (1024*1024)
            print(f"File size: {file_size:.1f} MB")
        print("\n✓ ALL 9 PIPELINE STAGES COMPLETED!")
        print("\nUse --force to re-run.")
        sys.exit(0)
    
    # Load configuration and genre profile
    config = load_config(args.config)
    genre_profile = load_genre_profile(args.genre)
    
    # Load timeline
    timeline_path = dirs['output'] / 'trailer_timeline.json'
    if not timeline_path.exists():
        logger.error("Timeline not found. Run stage 15 first.")
        print("\n❌ Error: Timeline not found. Run 15_timeline_constructor.py first!")
        sys.exit(1)
    
    with open(timeline_path, 'r') as f:
        timeline = json.load(f)
    
    logger.info(f"Loaded timeline with {len(timeline.get('timeline', []))} shots")
    
    # Get assembled video (genre-specific path)
    assembled_video = genre_output_dir / f'trailer_{args.genre}_assembled.mp4'
    if not assembled_video.exists():
        # Fallback to old path
        assembled_video = dirs['output'] / 'trailer_assembled.mp4'
    
    if not assembled_video.exists():
        logger.error("Assembled video not found. Run stage 9 first.")
        print("\n❌ Error: Assembled video not found. Run 9_video_assembly.py first!")
        sys.exit(1)
    
    logger.info(f"Using assembled video: {assembled_video}")
    
    # Override AI settings if flags provided
    if args.no_ai_music:
        config['audio']['ai_music_selection'] = False
        logger.info("AI music selection disabled via --no-ai-music")
    
    # Initialize Azure OpenAI client (if AI features enabled)
    azure_client = None
    if config.get('audio', {}).get('ai_music_selection', False):
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
            logger.info("Azure OpenAI client initialized for AI music selection")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI: {e}")
            logger.warning("Continuing without AI music selection")
            azure_client = None
    
    # Initialize audio mixer
    mixer = AudioMixer(
        config=config,
        genre_profile=genre_profile,
        output_dir=output_base,
        azure_client=azure_client,
        enable_ducking=not args.no_ducking
    )
    
    # Log mixing settings
    logger.info("=" * 60)
    logger.info("Audio Mixing Settings:")
    logger.info(f"  Audio ducking: {'enabled' if not args.no_ducking else 'disabled'}")
    logger.info(f"  AI music selection: {'enabled' if config.get('audio', {}).get('ai_music_selection') else 'disabled'}")
    logger.info(f"  Music library: {config.get('audio', {}).get('music_library_path')}")
    logger.info(f"  Sample rate: {config.get('audio', {}).get('sample_rate')} Hz")
    logger.info(f"  Bitrate: {config.get('audio', {}).get('bitrate')}")
    logger.info(f"  Normalization target: {config.get('audio', {}).get('normalization_target')} LUFS")
    if args.music_file:
        logger.info(f"  User-specified music: {args.music_file}")
    logger.info("=" * 60)
    
    # Mix audio
    logger.info(f"Mixing audio for {args.genre} trailer...")
    print(f"\n⏳ Mixing audio... This may take a few minutes.")
    
    try:
        final_trailer = mixer.mix_audio(
            timeline=timeline,
            video_path=assembled_video,
            output_path=final_trailer_path,
            music_file=args.music_file
        )
        
        file_size_mb = Path(final_trailer).stat().st_size / (1024*1024)
        logger.info(f"Final trailer created: {final_trailer}")
        logger.info(f"File size: {file_size_mb:.1f} MB")
        
        # Mark stage completed for this genre
        checkpoint.mark_stage_completed('audio_mixing', {
            'output_file': str(final_trailer),
            'file_size_mb': round(file_size_mb, 2),
            'duration': timeline.get('total_duration', 0)
        }, genre=args.genre)
        
        # Print final completion
        stats = checkpoint.get_stats()
        
        print("\n" + "=" * 60)
        print("✓ AUDIO MIXING COMPLETED")
        print("=" * 60)
        print(f"🎬 FINAL TRAILER: {final_trailer}")
        print(f"Duration: {timeline.get('total_duration', 0):.1f}s")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Genre: {args.genre}")
        print(f"\n✓ Pipeline Progress: {stats['completed_stages']}/{stats['total_stages']} stages (100%)")
        
        print("\n" + "=" * 60)
        print(f"🎉 TRAILER GENERATION COMPLETE FOR GENRE: {args.genre.upper()} 🎉")
        print("=" * 60)
        
        print("\nGenerated files:")
        print(f"  📹 Final trailer: {final_trailer}")
        print(f"  🎞️  Assembled video (no audio): {assembled_video}")
        print(f"  📋 Timeline: {timeline_path}")
        print(f"  💾 Shot metadata: {dirs['shots'] / 'shot_metadata.json'}")
        print(f"  📊 Checkpoint: {dirs['checkpoint_file']}")
        print(f"  📝 Logs: {dirs['log_file']}")
        
        print("\nCompleted stages:")
        for stage in stats['completed_list']:
            print(f"  ✓ {stage}")
        
        print(f"\n🎬 To view your trailer:")
        print(f"   ffplay {final_trailer}")
        print(f"\n   or open it in your video player of choice!")
        
    except Exception as e:
        logger.error(f"Audio mixing failed: {e}", exc_info=True)
        print(f"\n❌ Error: Audio mixing failed!")
        print(f"Details: {e}")
        print(f"\nCheck logs: {dirs['log_file']}")
        
        # Check if music library exists
        music_lib = Path(config.get('audio', {}).get('music_library_path', 'audio_assets/music/'))
        if not music_lib.exists():
            print(f"\n💡 Tip: Music library not found at {music_lib}")
            print("   Create the directory and add music files, or use --music-file to specify a track")
        
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
