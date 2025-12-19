#!/usr/bin/env python3
"""
Multi-Genre Pipeline Orchestrator
==================================

Runs the complete semantic pipeline for generating trailers in multiple genres.

Phase 1 (Sequential, Genre-Agnostic):
  - Stage 1: Shot Detection
  - Stage 2: Keyframe Extraction
  - Stage 3: Audio Extraction
  - Stage 4: Subtitle Management
  - Stage 5: Remote Analysis (Qwen2-VL)
  - Stage 11: Story Graph Generation

Phase 2 (Parallel, Per-Genre):
  For each genre in config.yaml:
  - Stage 12: Beat Sheet Generation
  - Stage 13: Embedding Generation
  - Stage 14: Scene Retrieval
  - Stage 15: Timeline Construction
  - Stage 9: Video Assembly
  - Stage 10: Audio Mixing

Usage:
    python run_multi_genre_pipeline.py <config_key>
    python run_multi_genre_pipeline.py hitch --parallel-workers 4
    python run_multi_genre_pipeline.py hitch --sequential  # Full output streaming
    python run_multi_genre_pipeline.py hitch --skip-phase1
"""

import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import traceback

import yaml

from pipeline_common import (
    load_pipeline_config,
    get_movie_config,
    get_output_base_dir,
    get_story_graph_dir,
    sanitize_filename,
    setup_logging,
    ALL_GENRES
)
from trailer_generator.checkpoint import CheckpointManager


def run_stage_script(
    script_name: str, 
    args: List[str], 
    log_prefix: str = "",
    stream_output: bool = True
) -> Tuple[bool, str]:
    """
    Run a pipeline stage script as a subprocess.
    
    Args:
        script_name: Name of the Python script to run
        args: Command-line arguments to pass
        log_prefix: Prefix for log messages
        stream_output: If True, stream stdout/stderr to console in real-time
        
    Returns:
        Tuple of (success, output_message)
    """
    cmd = [sys.executable, script_name] + args
    
    try:
        if stream_output:
            # Stream output to console in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Force unbuffered Python output
            )
            
            # Stream output line by line
            output_lines = []
            for line in process.stdout:
                prefixed_line = f"{log_prefix}{line}" if log_prefix else line
                print(prefixed_line, end='', flush=True)
                output_lines.append(line)
            
            process.wait()
            
            if process.returncode == 0:
                return True, f"{log_prefix}✓ {script_name} completed successfully"
            else:
                # Get last few lines for error context
                error_context = ''.join(output_lines[-20:]) if output_lines else "No output"
                return False, f"{log_prefix}✗ {script_name} failed (exit code {process.returncode})"
        else:
            # Capture output (for parallel execution to avoid interleaving)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per stage
            )
            
            if result.returncode == 0:
                return True, f"{log_prefix}✓ {script_name} completed successfully"
            else:
                error_msg = result.stderr[-1000:] if result.stderr else "No error output"
                return False, f"{log_prefix}✗ {script_name} failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, f"{log_prefix}✗ {script_name} timed out after 2 hours"
    except Exception as e:
        return False, f"{log_prefix}✗ {script_name} error: {str(e)}"


def run_genre_pipeline(
    genre: str,
    video_path: str,
    movie_name: str,
    config_path: str = 'trailer_generator/config/settings.yaml',
    stream_output: bool = True
) -> Tuple[str, bool, str]:
    """
    Run all genre-dependent stages for a single genre.
    
    Args:
        genre: Target genre
        video_path: Path to input video
        movie_name: Movie name for story graph lookup
        config_path: Path to settings.yaml
        stream_output: If True, stream output to console in real-time
        
    Returns:
        Tuple of (genre, success, message)
    """
    log_prefix = f"[{genre.upper()}] "
    
    stages = [
        ('12_beat_sheet_generator.py', ['--movie-name', movie_name, '--genre', genre]),
        ('13_embedding_generator.py', ['--input', video_path, '--genre', genre, '--movie-name', movie_name]),
        ('14_scene_retrieval.py', ['--input', video_path, '--genre', genre, '--movie-name', movie_name]),
        ('15_timeline_constructor.py', ['--input', video_path, '--genre', genre]),
        ('9_video_assembly.py', ['--input', video_path, '--genre', genre]),
        ('10_audio_mixing.py', ['--input', video_path, '--genre', genre]),
    ]
    
    for script, args in stages:
        print(f"{log_prefix}▶ Running {script}...")
        success, message = run_stage_script(script, args, log_prefix, stream_output=stream_output)
        print(message)
        
        if not success:
            return genre, False, f"Failed at {script}: {message}"
    
    return genre, True, f"{log_prefix}All stages completed successfully"


def run_phase1_stages(
    video_path: str,
    movie_name: str,
    synopsis_path: str,
    srt_path: str,
    force: bool = False
) -> bool:
    """
    Run Phase 1 (genre-agnostic) stages sequentially.
    
    Args:
        video_path: Path to input video
        movie_name: Movie name
        synopsis_path: Path to synopsis file
        srt_path: Path to SRT subtitle file
        force: Force re-run even if completed
        
    Returns:
        True if all stages succeeded
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Genre-Agnostic Processing")
    print("=" * 70)
    
    # Common args for stages 1-5
    common_args = ['--input', video_path]
    if force:
        common_args.append('--force')
    
    # Stages 1-5 (genre-agnostic)
    phase1_stages = [
        ('1_shot_detection.py', common_args),
        ('2_keyframe_extraction.py', common_args),
        ('3_audio_extraction.py', common_args),
        ('4_subtitle_management.py', common_args + ['--srt-file', srt_path]),
        ('5_remote_analysis.py', common_args),
    ]
    
    for script, args in phase1_stages:
        print(f"\n▶ Running {script}...")
        success, message = run_stage_script(script, args)
        print(message)
        
        if not success:
            print(f"\n❌ Phase 1 failed at {script}")
            return False
    
    # Stage 11: Story Graph Generation
    print(f"\n▶ Running 11_story_graph_generator.py...")
    stage11_args = [
        '--movie-name', movie_name,
        '--synopsis', synopsis_path,
        '--srt-file', srt_path
    ]
    if force:
        stage11_args.append('--force')
    
    success, message = run_stage_script('11_story_graph_generator.py', stage11_args)
    print(message)
    
    if not success:
        print(f"\n❌ Phase 1 failed at story graph generation")
        return False
    
    print("\n" + "=" * 70)
    print("✓ PHASE 1 COMPLETE")
    print("=" * 70)
    
    return True


def run_phase2_sequential(
    genres: List[str],
    video_path: str,
    movie_name: str
) -> Dict[str, bool]:
    """
    Run Phase 2 (genre-dependent) stages sequentially with full output streaming.
    
    Args:
        genres: List of genres to process
        video_path: Path to input video
        movie_name: Movie name
        
    Returns:
        Dictionary mapping genre to success status
    """
    print("\n" + "=" * 70)
    print(f"PHASE 2: Sequential Genre Processing ({len(genres)} genres)")
    print("=" * 70)
    print(f"Genres: {', '.join(genres)}")
    
    results = {}
    
    for genre in genres:
        print(f"\n{'=' * 70}")
        print(f"Processing Genre: {genre.upper()}")
        print(f"{'=' * 70}")
        
        result_genre, success, message = run_genre_pipeline(
            genre, video_path, movie_name, stream_output=True
        )
        results[result_genre] = success
        
        if success:
            print(f"\n✓ {result_genre.upper()}: Completed successfully")
        else:
            print(f"\n✗ {result_genre.upper()}: {message}")
            print("\n❌ Stopping processing due to failure")
            return results
    
    return results

def run_phase2_parallel(
    genres: List[str],
    video_path: str,
    movie_name: str,
    parallel_workers: int = 4
) -> Dict[str, bool]:
    """
    Run Phase 2 (genre-dependent) stages in parallel.
    
    Note: In parallel mode, output is captured rather than streamed to avoid
    interleaving. Use --sequential for full output visibility.
    
    Args:
        genres: List of genres to process
        video_path: Path to input video
        movie_name: Movie name
        parallel_workers: Number of parallel workers
        
    Returns:
        Dictionary mapping genre to success status
    """
    print("\n" + "=" * 70)
    print(f"PHASE 2: Parallel Genre Processing ({len(genres)} genres, {parallel_workers} workers)")
    print("=" * 70)
    print(f"Genres: {', '.join(genres)}")
    print("Note: Use --sequential for full output streaming")
    
    results = {}
    
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        # Submit all genre jobs with stream_output=False for parallel execution
        futures = {
            executor.submit(
                run_genre_pipeline, genre, video_path, movie_name,
                'trailer_generator/config/settings.yaml', False  # stream_output=False
            ): genre
            for genre in genres
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            genre = futures[future]
            try:
                result_genre, success, message = future.result()
                results[result_genre] = success
                
                if success:
                    print(f"\n✓ {result_genre.upper()}: Completed successfully")
                else:
                    print(f"\n✗ {result_genre.upper()}: {message}")
                    # Stop all processing on failure
                    print("\n❌ Stopping all processing due to failure")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return results
                    
            except Exception as e:
                results[genre] = False
                print(f"\n✗ {genre.upper()}: Exception - {str(e)}")
                traceback.print_exc()
                # Stop all processing on exception
                print("\n❌ Stopping all processing due to exception")
                executor.shutdown(wait=False, cancel_futures=True)
                return results
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Genre Trailer Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_multi_genre_pipeline.py hitch
  python run_multi_genre_pipeline.py hitch --parallel-workers 4
  python run_multi_genre_pipeline.py hitch --sequential  # Full output for debugging
  python run_multi_genre_pipeline.py hitch --skip-phase1
  python run_multi_genre_pipeline.py hitch --genres comedy,thriller,horror
        """
    )
    
    parser.add_argument('config_key', type=str,
                       help='Movie configuration key from config.yaml')
    parser.add_argument('--parallel-workers', type=int, default=4,
                       help='Number of parallel workers for genre processing (default: 4)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run Phase 2 genres sequentially with full output streaming (useful for debugging)')
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1 (genre-agnostic stages)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run all stages')
    parser.add_argument('--genres', type=str, default=None,
                       help='Comma-separated list of genres to override config.yaml')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to pipeline config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        movie_config = get_movie_config(args.config_key, args.config)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print(f"\nAvailable movie keys:")
        config = load_pipeline_config(args.config)
        for key in config.get('movies', {}).keys():
            print(f"  - {key}")
        sys.exit(1)
    
    # Extract configuration
    video_path = movie_config['video']
    movie_name = movie_config['movie_name']
    synopsis_path = movie_config['synopsis']
    srt_path = movie_config['subtitles']
    
    # Determine genres
    if args.genres:
        genres = [g.strip().lower() for g in args.genres.split(',')]
        # Validate genres
        invalid = [g for g in genres if g not in ALL_GENRES]
        if invalid:
            print(f"❌ Invalid genres: {invalid}")
            print(f"Valid genres: {ALL_GENRES}")
            sys.exit(1)
    else:
        genres = movie_config.get('genres', ['thriller'])
        genres = [g.lower() for g in genres]
    
    # Validate input files
    for path, name in [(video_path, 'Video'), (synopsis_path, 'Synopsis'), (srt_path, 'Subtitles')]:
        if not Path(path).exists():
            print(f"❌ {name} file not found: {path}")
            sys.exit(1)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("GenreBender: Multi-Genre Trailer Pipeline")
    print("=" * 70)
    print(f"Config Key: {args.config_key}")
    print(f"Movie: {movie_name}")
    print(f"Video: {video_path}")
    print(f"Genres: {', '.join(genres)}")
    print(f"Sequential Mode: {args.sequential}")
    print(f"Parallel Workers: {args.parallel_workers if not args.sequential else 'N/A (sequential)'}")
    print(f"Skip Phase 1: {args.skip_phase1}")
    print(f"Force Re-run: {args.force}")
    
    start_time = datetime.now()
    
    # Phase 1: Genre-Agnostic Stages
    if not args.skip_phase1:
        if not run_phase1_stages(video_path, movie_name, synopsis_path, srt_path, args.force):
            print("\n❌ Pipeline failed in Phase 1")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping Phase 1 (--skip-phase1)")
    
    # Phase 2: Multi-Genre Processing
    if args.sequential:
        results = run_phase2_sequential(genres, video_path, movie_name)
    else:
        results = run_phase2_parallel(genres, video_path, movie_name, args.parallel_workers)
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total Duration: {duration/60:.1f} minutes")
    print(f"\nGenre Results:")
    
    success_count = 0
    for genre, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {genre}")
        if success:
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(genres)} genres")
    
    # Output locations
    output_base = get_output_base_dir(video_path)
    print(f"\nOutput Location: {output_base}")
    print(f"Trailers Directory: {output_base / 'trailers'}")
    
    for genre in genres:
        if results.get(genre):
            trailer_path = output_base / 'trailers' / genre / f'trailer_{genre}_final.mp4'
            print(f"  - {genre}: {trailer_path}")
    
    if success_count < len(genres):
        print("\n⚠️  Some genres failed. Check logs for details.")
        sys.exit(1)
    else:
        print("\n🎬 All trailers generated successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
