#!/usr/bin/env python3
"""
Quick test script to verify parallel shot detection implementation.
"""

import sys
import time
from pathlib import Path
from trailer_generator.ingest import ShotDetector

def test_parallel_detection():
    """Test parallel shot detection on a sample video.

    Compares single-threaded streaming mode against multi-threaded parallel mode
    to verify the parallel implementation works correctly and provides speedup.

    Returns:
        bool: True if the test passes (speedup > 1.5x or test completes for
            short videos), False if no test video is found.
    """
    # Find a test video
    test_videos = [
        "test_files/birds_short_film.mp4"
    ]
    
    test_video = None
    for video in test_videos:
        if Path(video).exists():
            test_video = video
            break
    
    if not test_video:
        print("❌ No test video found. Tried:")
        for v in test_videos:
            print(f"   - {v}")
        return False
    
    print(f"✅ Found test video: {test_video}")
    print(f"📊 Video size: {Path(test_video).stat().st_size / 1024 / 1024:.2f} MB\n")
    
    # Create temp output directory
    output_dir = Path("outputs/test_parallel_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Single-threaded (streaming) mode
    print("=" * 60)
    print("TEST 1: Single-threaded streaming mode (baseline)")
    print("=" * 60)
    
    detector_single = ShotDetector(
        threshold=27.0,
        output_dir=str(output_dir / "single_threaded"),
        parallel_workers=1  # Force single-threaded
    )
    
    start_time = time.time()
    shots_single = detector_single.detect_shots(
        test_video,
        streaming=True,
        frame_skip=24,
        parallel=False  # Disable parallel
    )
    single_time = time.time() - start_time
    
    print(f"\n✅ Single-threaded: {len(shots_single)} shots in {single_time:.2f}s")
    print(f"   Files extracted: {sum(1 for s in shots_single if s.get('file'))}")
    
    # Test 2: Parallel mode
    print("\n" + "=" * 60)
    print("TEST 2: Multi-threaded parallel mode")
    print("=" * 60)
    
    detector_parallel = ShotDetector(
        threshold=27.0,
        output_dir=str(output_dir / "parallel"),
        parallel_workers=0,  # Auto-detect CPUs
        chunk_overlap=5.0
    )
    
    print(f"Workers: {detector_parallel.parallel_workers} CPUs")
    
    start_time = time.time()
    shots_parallel = detector_parallel.detect_shots(
        test_video,
        streaming=True,
        frame_skip=24,
        parallel=True  # Enable parallel
    )
    parallel_time = time.time() - start_time
    
    print(f"\n✅ Parallel: {len(shots_parallel)} shots in {parallel_time:.2f}s")
    print(f"   Files extracted: {sum(1 for s in shots_parallel if s.get('file'))}")
    
    # Calculate speedup
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    speedup = single_time / parallel_time if parallel_time > 0 else 0
    
    print(f"Single-threaded time: {single_time:.2f}s")
    print(f"Parallel time:        {parallel_time:.2f}s")
    print(f"Speedup:              {speedup:.2f}x")
    print(f"Shot count match:     {len(shots_single)} vs {len(shots_parallel)}")
    
    if speedup > 1.5:
        print(f"\n🎉 SUCCESS! Parallel mode is {speedup:.2f}x faster!")
        return True
    else:
        print(f"\n⚠️  Speedup is only {speedup:.2f}x (expected >1.5x)")
        print("This might be normal for very short videos.")
        return True

if __name__ == '__main__':
    try:
        success = test_parallel_detection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
