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

def create_skipped_analysis(reason: str, min_duration: float = None) -> dict:
    """
    Create a standardized "skipped" analysis result for shots that don't meet criteria.
    
    Args:
        reason: Reason for skipping (e.g., 'duration_too_short')
        min_duration: Minimum duration threshold if applicable
        
    Returns:
        Dictionary with skipped analysis structure
    """
    caption = f"Shot skipped ({reason})"
    if min_duration is not None:
        caption = f"Shot skipped (duration < {min_duration}s)"
    
    return {
        'caption': caption,
        'skipped': True,
        'skip_reason': reason,
        'attributes': {
            'suspense': 0.0,
            'darkness': 0.0,
            'ambiguity': 0.0,
            'emotional_tension': 0.0,
            'intensity': 0.0,
            'motion': 0.0,
            'impact': 0.0,
            'energy': 0.0,
            'emotional_connection': 0.0,
            'intimacy': 0.0,
            'warmth': 0.0,
            'fear': 0.0,
            'unease': 0.0,
            'shock': 0.0,
            'futuristic': 0.0,
            'technology': 0.0,
            'wonder': 0.0,
            'scale': 0.0,
            'humor': 0.0,
            'lightheartedness': 0.0,
            'timing': 0.0,
            'beauty': 0.0
        }
    }

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
        print("Use --force flag to re-run, or proceed to next stage: 6_genre_scoring.py")
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
    
    # Filter shots by minimum duration
    min_shot_duration = config['remote_analysis'].get('min_shot_duration', 2.0)
    logger.info(f"Minimum shot duration for analysis: {min_shot_duration}s")
    
    qualifying_shots = []
    skipped_shots = []
    
    for shot in shots:
        duration = shot.get('duration', 0)
        if duration >= min_shot_duration:
            qualifying_shots.append(shot)
        else:
            # Mark shot as skipped with special analysis
            shot['analysis'] = create_skipped_analysis('duration_too_short', min_shot_duration)
            skipped_shots.append(shot)
    
    logger.info(f"Shot filtering: {len(qualifying_shots)} qualifying, {len(skipped_shots)} skipped (< {min_shot_duration}s)")
    print(f"\n📊 Shot Duration Filter:")
    print(f"  ✓ Qualifying shots (≥ {min_shot_duration}s): {len(qualifying_shots)}")
    print(f"  ⊘ Skipped shots (< {min_shot_duration}s): {len(skipped_shots)}")
    
    # Initialize cache
    cache = AnalysisCache(
        cache_dir=str(dirs['cache']),
        enabled=not args.skip_cache
    )
    
    # Check cache for qualifying shots only
    cached_shots, uncached_shots = cache.get_batch(qualifying_shots)
    logger.info(f"Cache: {len(cached_shots)} cached, {len(uncached_shots)} to analyze")
    
    # Show resume status if some shots are already cached
    if cached_shots:
        print(f"\n🔄 Resuming from previous run:")
        print(f"  ✓ Already analyzed (cached): {len(cached_shots)}")
        print(f"  → Remaining to analyze: {len(uncached_shots)}")
    
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
        total_batches = (len(uncached_shots) + config['remote_analysis']['batch_size'] - 1) // config['remote_analysis']['batch_size']
        batch_num = 0
        
        for batch in batch_processor.batch_shots(uncached_shots):
            batch_num += 1
            start_shot_id = batch[0].get('id', '?') if batch else '?'
            end_shot_id = batch[-1].get('id', '?') if batch else '?'
            logger.info(f"Analyzing batch {batch_num}/{total_batches} ({len(batch)} shots: {start_shot_id}-{end_shot_id})...")
            print(f"  📦 Batch {batch_num}/{total_batches}: shots {start_shot_id}-{end_shot_id} ({len(batch)} shots)")
            
            batch_results = analyzer.analyze_batch(batch, args.input)
            analyzed_shots.extend(batch_results)
            
            # Cache this batch immediately (persists to disk for resume capability)
            cache.put_batch(batch_results)
            logger.info(f"Cached batch {batch_num}/{total_batches} ({len(batch_results)} shots)")
            
            # Save partial results as backup
            partial_results_path = dirs['temp'] / 'partial_analysis.json'
            batch_processor.save_partial_results(
                analyzed_shots,
                str(partial_results_path)
            )
        
        # Combine analyzed shots (cached + newly analyzed)
        analyzed_shots_combined = cached_shots + analyzed_shots
    else:
        analyzed_shots_combined = cached_shots
    
    # Combine all shots: analyzed + skipped (maintain original order)
    all_shots = analyzed_shots_combined + skipped_shots
    
    # Sort by shot ID to maintain original order
    all_shots.sort(key=lambda s: s.get('id', 0))
    
    logger.info(f"Final shot count: {len(all_shots)} total ({len(analyzed_shots_combined)} analyzed, {len(skipped_shots)} skipped)")
    
    # Save updated metadata with all shots
    save_shots_to_metadata(all_shots, shot_metadata_path, args.input)
    logger.info(f"Updated shot metadata with analysis results")
    
    # Mark stage completed
    checkpoint.mark_stage_completed('remote_analysis', {
        'analyzed_count': len(analyzed_shots_combined),
        'skipped_count': len(skipped_shots),
        'total_count': len(all_shots)
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
