#!/usr/bin/env python3
"""
Stage 4: Remote Analysis
Performs multimodal analysis using Qwen2-VL server.
"""

import argparse
import sys
from pathlib import Path

from pipeline_common import (
    initialize_stage, print_completion_message, add_common_arguments,
    load_config, load_shots_from_metadata, save_shots_to_metadata
)
from trailer_generator.analysis import RemoteAnalyzer, MultiServerAnalyzer, AnalysisCache
from trailer_generator.ingest import BatchProcessor

def main():
    parser = argparse.ArgumentParser(
        description='Stage 4: Remote Analysis - Multimodal analysis with Qwen2-VL'
    )
    add_common_arguments(parser)
    parser.add_argument('--skip-cache', action='store_true',
                       help='Disable analysis caching')
    args = parser.parse_args()
    
    # Initialize
    output_base, dirs, checkpoint, logger = initialize_stage(
        'remote_analysis', args.input, args.genre
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
    if checkpoint.is_stage_completed('remote_analysis') and not args.force:
        logger.warning("⚠️  Remote analysis already completed. Use --force to re-run.")
        print("\n⚠️  This stage is already completed.")
        print("Use --force flag to re-run, or proceed to next stage: 5_genre_scoring.py")
        sys.exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load shots from metadata
    shot_metadata_path = dirs['shots'] / 'shot_metadata.json'
    shots = load_shots_from_metadata(shot_metadata_path)
    if not shots:
        logger.error("Failed to load shots from metadata")
        sys.exit(1)
    
    logger.info(f"Loaded {len(shots)} shots from metadata")
    
    # Initialize cache
    cache = AnalysisCache(
        cache_dir=str(dirs['cache']),
        enabled=not args.skip_cache
    )
    
    # Check cache first
    cached_shots, uncached_shots = cache.get_batch(shots)
    logger.info(f"Cache: {len(cached_shots)} cached, {len(uncached_shots)} to analyze")
    
    if uncached_shots:
        # Build server URLs from configuration
        server_host = config['remote_analysis'].get('server_host', 'localhost')
        server_base_port = config['remote_analysis'].get('server_base_port', 8000)
        server_count = config['remote_analysis'].get('server_count', 1)
        
        if server_count > 1:
            # Multi-server mode (true multi-GPU parallelism)
            server_urls = [
                f"http://{server_host}:{server_base_port + i}"
                for i in range(server_count)
            ]
            logger.info(f"Using multi-server mode with {server_count} servers")
            logger.info(f"Server URLs: {server_urls}")
            
            analyzer = MultiServerAnalyzer(
                server_urls=server_urls,
                load_balancing=config['remote_analysis'].get('load_balancing', 'round_robin'),
                timeout=config['remote_analysis']['timeout'],
                max_retries=config['remote_analysis']['max_retries'],
                batch_size=config['remote_analysis']['batch_size'],
                api_key=config['remote_analysis'].get('api_key')
            )
        else:
            # Single-server mode
            server_url = f"http://{server_host}:{server_base_port}"
            logger.info(f"Using single-server mode: {server_url}")
            
            analyzer = RemoteAnalyzer(
                server_url=server_url,
                timeout=config['remote_analysis']['timeout'],
                max_retries=config['remote_analysis']['max_retries'],
                batch_size=config['remote_analysis']['batch_size'],
                api_key=config['remote_analysis'].get('api_key')
            )
        
        # Check server health
        if not analyzer.health_check():
            logger.warning("⚠️  Qwen2-VL server(s) not responding. Check server status.")
            print("\n⚠️  Warning: Qwen2-VL server(s) not responding!")
            if server_count > 1:
                print(f"Make sure all {server_count} servers are running at:")
                for url in server_urls:
                    print(f"  - {url}")
            else:
                print(f"Make sure the server is running at: {server_url}")
            sys.exit(1)
        
        # Batch process uncached shots
        # Use remote_analysis batch_size to match server limits
        batch_processor = BatchProcessor(
            batch_size=config['remote_analysis']['batch_size']
        )
        
        analyzed_shots = []
        for batch in batch_processor.batch_shots(uncached_shots):
            logger.info(f"Analyzing batch of {len(batch)} shots...")
            batch_results = analyzer.analyze_batch(batch, args.input)
            analyzed_shots.extend(batch_results)
            
            # Save partial results
            partial_results_path = dirs['temp'] / 'partial_analysis.json'
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
    
    # Save updated metadata
    save_shots_to_metadata(shots, shot_metadata_path, args.input)
    logger.info(f"Updated shot metadata with analysis results")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('remote_analysis', {
        'analyzed_count': sum(1 for s in shots if s.get('analysis'))
    })
    
    # Print completion
    print_completion_message('remote_analysis', checkpoint, output_base)

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
