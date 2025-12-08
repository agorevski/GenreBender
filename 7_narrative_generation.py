#!/usr/bin/env python3
"""
Stage 7: Narrative Generation
Generates coherent trailer narrative structure using Azure OpenAI GPT-4.
"""

import argparse
import sys
import json
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config
)
from trailer_generator.narrative import AzureOpenAIClient, TimelineGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Stage 7: Narrative Generation - Create trailer timeline with GPT-4'
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'narrative_generation', args.input, args.genre
    )
    
    # Validate prerequisites
    required_stages = ['shot_detection', 'keyframe_extraction', 'audio_extraction', 
                      'remote_analysis', 'genre_scoring', 'shot_selection']
    for stage in required_stages:
        if not checkpoint.is_stage_completed(stage):
            logger.error(f"❌ Prerequisite stage '{stage}' not completed.")
            print(f"\n❌ Error: You must complete all previous stages first!")
            print(f"Missing: {stage}")
            sys.exit(1)
    
    # Check if already completed
    timeline_path = dirs['output'] / 'timeline.json'
    if checkpoint.is_stage_completed('narrative_generation') and not args.force:
        logger.warning("⚠️  Narrative generation already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print(f"Timeline saved at: {timeline_path}")
        print("\n✓ All 7 pipeline stages completed!")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load selected shots
    top_shots_path = dirs['output'] / 'selected_shots.json'
    if not top_shots_path.exists():
        logger.error("Selected shots file not found. Run stage 6 first.")
        print("\n❌ Error: Selected shots not found. Run 6_shot_selection.py first!")
        sys.exit(1)
    
    with open(top_shots_path, 'r') as f:
        top_shots = json.load(f)
    
    logger.info(f"Loaded {len(top_shots)} selected shots")
    
    # Initialize Azure OpenAI client
    azure_client = AzureOpenAIClient(
        endpoint=config['azure_openai']['endpoint'],
        api_key=config['azure_openai']['api_key'],
        deployment_name=config['azure_openai']['deployment_name'],
        api_version=config['azure_openai']['api_version'],
        max_retries=config['azure_openai']['max_retries'],
        temperature=config['azure_openai']['temperature'],
        max_completion_tokens=config['azure_openai']['max_completion_tokens']
    )
    
    # Initialize timeline generator
    timeline_gen = TimelineGenerator(
        azure_client=azure_client,
        genre=args.genre
    )
    
    # Generate timeline
    logger.info(f"Generating narrative timeline for {args.genre} trailer...")
    timeline = timeline_gen.generate_timeline(
        top_shots,
        target_duration=config['processing']['target_trailer_length']
    )
    
    # Export timeline
    timeline_gen.export_timeline(timeline, str(timeline_path))
    logger.info(f"Generated timeline with {len(timeline['timeline'])} shots")
    logger.info(f"Total duration: {timeline.get('total_duration', 0):.1f}s")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('narrative_generation', {
        'timeline_shots': len(timeline.get('timeline', [])),
        'duration': timeline.get('total_duration', 0)
    })
    
    # Print completion with final summary
    stats = checkpoint.get_stats()
    
    print("\n" + "=" * 60)
    print("✓ NARRATIVE GENERATION COMPLETED")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print(f"Timeline: {timeline_path}")
    print(f"Timeline shots: {len(timeline.get('timeline', []))}")
    print(f"Duration: {timeline.get('total_duration', 0):.1f}s")
    print(f"\n✓ Pipeline Progress: {stats['completed_stages']}/{stats['total_stages']} stages (100%)")
    print("\n✓ ALL 7 PIPELINE STAGES COMPLETED!")
    print("\nGenerated files:")
    print(f"  - Shot metadata: {dirs['shots'] / 'shot_metadata.json'}")
    print(f"  - Selected shots: {top_shots_path}")
    print(f"  - Timeline: {timeline_path}")
    print(f"  - Checkpoint: {dirs['checkpoint_file']}")
    print(f"  - Logs: {dirs['log_file']}")
    print("\nNext steps (planned for future implementation):")
    print("  - Stage 8: Video assembly")
    print("  - Stage 9: Audio mixing")

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
