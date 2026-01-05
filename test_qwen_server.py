#!/usr/bin/env python3
"""
Test script for Qwen2-VL server.
Tests the server using the first shot from shots/shot_metadata.json.
"""

import json
import base64
import requests
from pathlib import Path
from typing import Dict, List

# Configuration
SERVER_URL = "http://localhost:8000"
API_KEY = "helloagorevski"
SHOT_METADATA_PATH = "shots/shot_metadata.json"

def load_shot_metadata() -> Dict:
    """Load shot metadata from JSON file.

    Reads the shot metadata JSON file from the configured path and
    parses it into a dictionary.

    Returns:
        Dict: The parsed shot metadata containing shot information.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(SHOT_METADATA_PATH, 'r') as f:
        return json.load(f)

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string.

    Reads the binary content of an image file and encodes it as a
    base64 string suitable for JSON transmission.

    Args:
        image_path: Path to the image file to encode.

    Returns:
        str: Base64 encoded string representation of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        IOError: If the file cannot be read.
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def encode_video_to_base64(video_path: str) -> str:
    """Encode a video file to base64 string.

    Reads the binary content of a video file and encodes it as a
    base64 string suitable for JSON transmission.

    Args:
        video_path: Path to the video file to encode.

    Returns:
        str: Base64 encoded string representation of the video.

    Raises:
        FileNotFoundError: If the video file does not exist.
        IOError: If the file cannot be read.
    """
    with open(video_path, 'rb') as f:
        video_data = f.read()
    return base64.b64encode(video_data).decode('utf-8')

def extract_video_clip(input_video: str, start_time: float, end_time: float, output_path: str) -> bool:
    """Extract a video clip from a larger video file.

    Uses ffmpeg to extract a segment of video between the specified
    start and end times without re-encoding (stream copy mode).

    Args:
        input_video: Path to the input video file.
        start_time: Start time of the clip in seconds.
        end_time: End time of the clip in seconds.
        output_path: Path where the extracted clip will be saved.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    import subprocess
    
    try:
        # Use ffmpeg to extract clip
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error extracting video clip: {e}")
        return False

def test_health_check() -> bool:
    """Test the health check endpoint.

    Sends a GET request to the server's health endpoint to verify
    that the Qwen2-VL server is running and responding correctly.

    Returns:
        bool: True if the server is healthy and responding, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Server is healthy")
            print(f"  Status: {health_data.get('status')}")
            print(f"  Model: {health_data.get('model')}")
            print(f"  Device: {health_data.get('device')}")
            
            if health_data.get('gpu_count'):
                print(f"  GPU Count: {health_data.get('gpu_count')}")
                print(f"  GPU Memory: {health_data.get('gpu_memory_total')}")
            
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {SERVER_URL}")
        print(f"  Make sure the server is running with: cd qwen_server && ./start_server.sh")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_analyze_shot(shot: Dict) -> bool:
    """Test analyzing a single shot using keyframe images.

    Encodes the shot's keyframe images to base64 and sends them to
    the analysis endpoint for visual content analysis.

    Args:
        shot: Shot data dictionary containing 'id', 'keyframes',
            'audio_features', 'start_time', 'end_time', and 'duration'.

    Returns:
        bool: True if analysis succeeded, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Testing Shot Analysis Endpoint")
    print("=" * 60)
    
    # Prepare shot data
    shot_id = shot['id']
    keyframes = shot.get('keyframes', [])
    audio_features = shot.get('audio_features')
    
    print(f"Shot ID: {shot_id}")
    print(f"Duration: {shot['duration']:.2f}s")
    print(f"Keyframes: {len(keyframes)}")
    print(f"Audio features: {'Yes' if audio_features else 'No'}")
    
    # Check if keyframe files exist
    missing_files = []
    for kf in keyframes:
        if not Path(kf).exists():
            missing_files.append(kf)
    
    if missing_files:
        print(f"\n✗ Missing keyframe files:")
        for mf in missing_files:
            print(f"  - {mf}")
        return False
    
    # Encode keyframes to base64
    print(f"\nEncoding {len(keyframes)} keyframes to base64...")
    encoded_images = []
    for kf in keyframes:
        try:
            encoded = encode_image_to_base64(kf)
            encoded_images.append(encoded)
        except Exception as e:
            print(f"✗ Error encoding {kf}: {e}")
            return False
    
    print(f"✓ Encoded {len(encoded_images)} images")
    
    # Prepare request payload
    payload = {
        "shot_id": shot_id,
        "images": encoded_images,
        "audio_features": audio_features,
        "start_time": shot['start_time'],
        "end_time": shot['end_time'],
        "duration": shot['duration']
    }
    
    # Make API request
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"\nSending analysis request to {SERVER_URL}/analyze...")
    
    try:
        response = requests.post(
            f"{SERVER_URL}/analyze",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Analysis completed successfully!")
            print("\n" + "-" * 60)
            print("RESULTS")
            print("-" * 60)
            
            # Display caption
            caption = result.get('caption', 'N/A')
            print(f"\nCaption:")
            print(f"  {caption}")
            
            # Display attributes
            attributes = result.get('attributes', {})
            print(f"\nAttribute Scores:")
            for attr_name, score in attributes.items():
                bar_length = int(score * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"  {attr_name:20s} [{bar}] {score:.3f}")
            
            return True
            
        elif response.status_code == 401:
            print(f"✗ Authentication failed - check API key")
            return False
        elif response.status_code == 400:
            print(f"✗ Bad request: {response.json().get('detail', 'Unknown error')}")
            return False
        else:
            print(f"✗ Analysis failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"✗ Analysis error: {e}")
        return False

def test_analyze_shot_video(shot: Dict, source_video: str) -> bool:
    """Test analyzing a single shot using video mode.

    Extracts a video clip for the shot's time range, encodes it to
    base64, and sends it to the analysis endpoint for processing.

    Args:
        shot: Shot data dictionary containing 'id', 'start_time',
            'end_time', 'duration', and optionally 'audio_features'.
        source_video: Path to the source video file to extract clip from.

    Returns:
        bool: True if analysis succeeded, False otherwise.
    """
    print("\n" + "=" * 60)
    print("Testing Shot Analysis Endpoint (VIDEO MODE)")
    print("=" * 60)
    
    # Prepare shot data
    shot_id = shot['id']
    start_time = shot['start_time']
    end_time = shot['end_time']
    duration = shot['duration']
    audio_features = shot.get('audio_features')
    
    print(f"Shot ID: {shot_id}")
    print(f"Start: {start_time:.2f}s, End: {end_time:.2f}s")
    print(f"Duration: {duration:.2f}s")
    print(f"Audio features: {'Yes' if audio_features else 'No'}")
    print(f"Source video: {source_video}")
    
    # Check if source video exists
    if not Path(source_video).exists():
        print(f"\n✗ Source video not found: {source_video}")
        return False
    
    # Extract video clip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_clip:
        clip_path = tmp_clip.name
    
    print(f"\nExtracting video clip from {start_time:.2f}s to {end_time:.2f}s...")
    
    if not extract_video_clip(source_video, start_time, end_time, clip_path):
        print(f"✗ Failed to extract video clip")
        return False
    
    try:
        # Encode video to base64
        print(f"Encoding video clip to base64...")
        encoded_video = encode_video_to_base64(clip_path)
        print(f"✓ Encoded video clip ({len(encoded_video)} bytes base64)")
        
        # Prepare request payload
        payload = {
            "shot_id": shot_id,
            "video": encoded_video,
            "audio_features": audio_features,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        }
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        print(f"\nSending video analysis request to {SERVER_URL}/analyze...")
        
        response = requests.post(
            f"{SERVER_URL}/analyze",
            json=payload,
            headers=headers,
            timeout=120  # Longer timeout for video processing
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Video analysis completed successfully!")
            print("\n" + "-" * 60)
            print("RESULTS (VIDEO MODE)")
            print("-" * 60)
            
            # Display caption
            caption = result.get('caption', 'N/A')
            print(f"\nCaption:")
            print(f"  {caption}")
            
            # Display attributes
            attributes = result.get('attributes', {})
            print(f"\nAttribute Scores:")
            for attr_name, score in attributes.items():
                bar_length = int(score * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"  {attr_name:20s} [{bar}] {score:.3f}")
            
            return True
            
        elif response.status_code == 401:
            print(f"✗ Authentication failed - check API key")
            return False
        elif response.status_code == 400:
            print(f"✗ Bad request: {response.json().get('detail', 'Unknown error')}")
            return False
        else:
            print(f"✗ Analysis failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"✗ Video analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(clip_path)
        except:
            pass

def main():
    """Main test execution entry point.

    Parses command line arguments, loads shot metadata, runs health
    checks, and executes analysis tests on shots. Supports both
    keyframe and video analysis modes.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Qwen2-VL server')
    parser.add_argument('--video-mode', action='store_true',
                      help='Test video mode instead of keyframe mode')
    parser.add_argument('--source-video', type=str, default='samples/birds_short_film_h264.mp4',
                      help='Source video file for video mode testing')
    parser.add_argument('--first-only', action='store_true',
                      help='Test only the first shot')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("QWEN2-VL SERVER TEST SCRIPT")
    print("=" * 60)
    print(f"Server URL: {SERVER_URL}")
    print(f"API Key: {API_KEY}")
    print(f"Shot Metadata: {SHOT_METADATA_PATH}")
    print(f"Mode: {'VIDEO' if args.video_mode else 'KEYFRAME'}")
    if args.video_mode:
        print(f"Source Video: {args.source_video}")
    
    # Load shot metadata
    try:
        metadata = load_shot_metadata()
        shots = metadata.get('shots', [])
        
        if not shots:
            print("\n✗ No shots found in metadata")
            return 1
        
        print(f"\n✓ Loaded {len(shots)} shots from metadata")
        
    except FileNotFoundError:
        print(f"\n✗ Shot metadata file not found: {SHOT_METADATA_PATH}")
        return 1
    except Exception as e:
        print(f"\n✗ Error loading metadata: {e}")
        return 1
    
    # Test health check
    if not test_health_check():
        return 1
    
    # Test analyzing shots
    successful_tests = 0
    failed_tests = 0
    
    # Limit to first shot if requested
    test_shots = [shots[0]] if args.first_only else shots
    
    for i, shot in enumerate(test_shots, 1):
        print(f"\n{'=' * 60}")
        print(f"Testing Shot {i}/{len(test_shots)}")
        print(f"{'=' * 60}")
        
        # Choose test function based on mode
        if args.video_mode:
            success = test_analyze_shot_video(shot, args.source_video)
        else:
            success = test_analyze_shot(shot)
        
        if success:
            successful_tests += 1
        else:
            failed_tests += 1
            print(f"\n✗ Shot {shot['id']} analysis failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total shots tested: {len(shots)}")
    print(f"Successful: {successful_tests} ✓")
    print(f"Failed: {failed_tests} ✗")
    
    if failed_tests == 0:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe Qwen2-VL server is working correctly!")
        print("\nNext steps:")
        print("  - Integrate with main pipeline")
        print("  - Monitor performance and GPU usage")
        print("  - Test batch analysis endpoint")
        return 0
    else:
        print(f"\n✗ {failed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
