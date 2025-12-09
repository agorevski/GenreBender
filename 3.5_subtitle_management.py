#!/usr/bin/env python3
"""
Stage 3.5: Subtitle Management
Extracts and maps subtitle data from SRT files to shots.
"""

import argparse
import sys
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config, load_shots_from_metadata, save_shots_to_metadata
)
from trailer_generator.ingest import SubtitleExtractor

def main():
    parser = argparse.ArgumentParser(
        description='Stage 3.5: Subtitle Management - Extract and map subtitles to shots'
    )
    add_common_arguments(parser)
    parser.add_argument('--srt-file', type=str, default=None,
                       help='Explicit path to SRT file (default: auto-detect from video name)')
    parser.add_argument('--skip-subtitles', action='store_true',
                       help='Skip subtitle processing (continue without subtitles)')
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'subtitle_management', args.input, args.genre
    )
    
    # Validate prerequisites
    required_stages = ['shot_detection', 'keyframe_extraction', 'audio_extraction']
    for stage in required_stages:
        if not checkpoint.is_stage_completed(stage):
            logger.error(f"❌ Prerequisite stage '{stage}' not completed.")
            print(f"\n❌ Error: You must complete all previous stages first!")
            print(f"Missing: {stage}")
            sys.exit(1)
    
    # Check if already completed
    if checkpoint.is_stage_completed('subtitle_management') and not args.force:
        logger.warning("⚠️  Subtitle management already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 4_remote_analysis.py")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    subtitle_config = config.get('subtitle_management', {})
    
    # Load shots from metadata
    shot_metadata_path = dirs['shots'] / 'shot_metadata.json'
    shots = load_shots_from_metadata(shot_metadata_path)
    if not shots:
        logger.error("Failed to load shots from metadata")
        sys.exit(1)
    
    logger.info(f"Loaded {len(shots)} shots from metadata")
    
    # Check if subtitles are enabled
    if not subtitle_config.get('enabled', True) or args.skip_subtitles:
        logger.info("Subtitle processing disabled, adding empty subtitle data")
        print("\n⚠️  Subtitle processing disabled")
        
        # Add empty subtitle data to all shots
        for shot in shots:
            shot['subtitles'] = {
                'has_dialogue': False,
                'dialogue': None,
                'subtitle_entries': [],
                'word_count': 0,
                'dialogue_density': 0.0,
                'emotional_markers': {
                    'questions': 0,
                    'exclamations': 0,
                    'all_caps_words': 0
                }
            }
        
        # Save and complete
        save_shots_to_metadata(shots, shot_metadata_path, args.input)
        checkpoint.mark_stage_completed('subtitle_management', {
            'subtitles_processed': False,
            'dialogue_shots': 0
        })
        print_completion_message('subtitle_management', checkpoint, output_base)
        return
    
    # Initialize subtitle extractor
    subtitle_extractor = SubtitleExtractor(
        min_dialogue_duration=subtitle_config.get('min_dialogue_duration', 0.3)
    )
    
    # Find SRT file
    explicit_srt = args.srt_file or subtitle_config.get('srt_file')
    srt_path = subtitle_extractor.find_srt_file(args.input, explicit_srt)
    
    if not srt_path:
        if subtitle_config.get('fallback_to_no_subtitles', True):
            logger.warning("No SRT file found, continuing without subtitles")
            print("\n⚠️  No SRT file found")
            print("Continuing without subtitle data...")
            
            # Add empty subtitle data
            for shot in shots:
                shot['subtitles'] = {
                    'has_dialogue': False,
                    'dialogue': None,
                    'subtitle_entries': [],
                    'word_count': 0,
                    'dialogue_density': 0.0,
                    'emotional_markers': {
                        'questions': 0,
                        'exclamations': 0,
                        'all_caps_words': 0
                    }
                }
            
            # Save and complete
            save_shots_to_metadata(shots, shot_metadata_path, args.input)
            checkpoint.mark_stage_completed('subtitle_management', {
                'subtitles_processed': False,
                'dialogue_shots': 0
            })
            print_completion_message('subtitle_management', checkpoint, output_base)
            return
        else:
            logger.error("No SRT file found and fallback disabled")
            print("\n❌ Error: No SRT file found!")
            print("\nOptions:")
            print("  1. Provide SRT file with --srt-file argument")
            print(f"  2. Place {Path(args.input).with_suffix('.srt').name} in same directory as video")
            print("  3. Use --skip-subtitles to continue without subtitles")
            sys.exit(1)
    
    # Load SRT file
    logger.info(f"Loading SRT file: {srt_path}")
    subtitles = subtitle_extractor.load_srt(srt_path)
    
    if not subtitles:
        logger.error("Failed to load SRT file")
        print("\n❌ Error: Failed to parse SRT file!")
        print("The SRT file may be corrupted or in an unsupported format.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(subtitles)} subtitle entries")
    
    # Map subtitles to shots
    logger.info("Mapping subtitles to shots...")
    enriched_shots = subtitle_extractor.map_to_shots(subtitles, shots)
    
    # Get summary statistics
    summary = subtitle_extractor.get_dialogue_summary(enriched_shots)
    logger.info(f"Subtitle mapping complete:")
    logger.info(f"  - Dialogue shots: {summary['dialogue_shots']}/{summary['total_shots']}")
    logger.info(f"  - Coverage: {summary['dialogue_coverage']:.1%}")
    logger.info(f"  - Total words: {summary['total_words']}")
    logger.info(f"  - Avg density: {summary['avg_dialogue_density']:.2f} words/sec")
    
    # Save updated metadata
    save_shots_to_metadata(enriched_shots, shot_metadata_path, args.input)
    logger.info("Updated shot metadata with subtitle data")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('subtitle_management', {
        'subtitles_processed': True,
        'srt_file': srt_path,
        'dialogue_shots': summary['dialogue_shots'],
        'total_words': summary['total_words'],
        'coverage': summary['dialogue_coverage']
    })
    
    # Print completion with statistics
    print("\n" + "=" * 60)
    print("✓ SUBTITLE MANAGEMENT COMPLETED")
    print("=" * 60)
    print(f"SRT file: {srt_path}")
    print(f"Subtitle entries: {len(subtitles)}")
    print(f"Shots with dialogue: {summary['dialogue_shots']}/{summary['total_shots']} ({summary['dialogue_coverage']:.1%})")
    print(f"Total words: {summary['total_words']}")
    print(f"Avg dialogue density: {summary['avg_dialogue_density']:.2f} words/sec")
    print("\nNext stage: 4_remote_analysis.py")

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
