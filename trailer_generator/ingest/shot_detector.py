"""
Shot detection module with streaming support and overlap handling.
Uses PySceneDetect to identify shot boundaries in video files.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from scenedetect import open_video, SceneDetector, detect
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
import logging

logger = logging.getLogger(__name__)


class ShotDetector:
    """
    Detects shots in video files using content-based detection.
    Supports streaming processing for long movies with chunk overlap.
    """
    
    def __init__(self, threshold: float = 27.0, chunk_duration: int = 30, 
                 overlap: int = 5, output_dir: str = "shots"):
        """
        Initialize shot detector.
        
        Args:
            threshold: Content detection threshold (default: 27.0)
            chunk_duration: Duration of each processing chunk in seconds
            overlap: Overlap duration between chunks to catch transitions
            output_dir: Directory to save shot segments
        """
        self.threshold = threshold
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_shots(self, video_path: str, streaming: bool = True) -> List[Dict]:
        """
        Detect shots in video file.
        
        Args:
            video_path: Path to input video file
            streaming: If True, process in chunks with overlap
            
        Returns:
            List of shot dictionaries with metadata
        """
        if streaming:
            return self._detect_shots_streaming(video_path)
        else:
            return self._detect_shots_full(video_path)
    
    def _detect_shots_streaming(self, video_path: str) -> List[Dict]:
        """
        Detect shots with overlapping chunks to avoid missing transitions.
        
        Process strategy:
        - Chunk 1: 0-30s
        - Chunk 2: 25-55s (5s overlap)
        - Chunk 3: 50-80s (5s overlap)
        
        Deduplicates shots detected in overlapping regions.
        """
        logger.info(f"Starting streaming shot detection on {video_path}")
        logger.info(f"Chunk duration: {self.chunk_duration}s, Overlap: {self.overlap}s")
        
        # Get video duration using modern API
        video = open_video(video_path)
        total_duration = video.duration.get_seconds()
        logger.info(f"Total video duration: {total_duration:.2f} seconds")
        
        all_shots = []
        chunk_start = 0
        chunk_id = 0
        
        while chunk_start < total_duration:
            chunk_end = min(chunk_start + self.chunk_duration, total_duration)
            logger.info(f"Processing chunk {chunk_id}: {chunk_start:.2f}s - {chunk_end:.2f}s")
            
            # Detect shots in this chunk
            chunk_shots = self._detect_chunk(video_path, chunk_start, chunk_end)
            
            # Add to all shots (deduplication happens later)
            all_shots.extend(chunk_shots)
            
            # Move to next chunk with overlap
            chunk_start += self.chunk_duration - self.overlap
            chunk_id += 1
        
        # Deduplicate overlapping shots
        deduplicated_shots = self._deduplicate_shots(all_shots)
        
        logger.info(f"Detected {len(deduplicated_shots)} unique shots")
        
        # Save metadata
        self._save_shot_metadata(deduplicated_shots, video_path)
        
        return deduplicated_shots
    
    def _detect_chunk(self, video_path: str, start_time: float, 
                     end_time: float) -> List[Dict]:
        """
        Detect shots within a specific time range.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of shots detected in this chunk
        """
        # Create detector
        detector = ContentDetector(threshold=self.threshold)
        
        # Detect scenes in the specified time range (pass path string, not video object)
        scene_list = detect(video_path, detector, start_time=start_time, end_time=end_time)
        
        # Convert to shot dictionaries
        shots = []
        for i, (start_frame, end_frame) in enumerate(scene_list):
            shot = {
                'start_time': start_frame.get_seconds(),
                'end_time': end_frame.get_seconds(),
                'start_frame': start_frame.get_frames(),
                'start_time': start_frame.get_seconds(),
                'end_time': end_frame.get_seconds(),
                'start_frame': start_frame.get_frames(),
                'end_frame': end_frame.get_frames(),
                'duration': (end_frame - start_frame).get_seconds()
            }
            shots.append(shot)
        
        return shots
    
    def _detect_shots_full(self, video_path: str) -> List[Dict]:
        """
        Detect all shots in video without chunking (for shorter videos).
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of shot dictionaries
        """
        logger.info(f"Starting full video shot detection on {video_path}")
        
        # Open video with modern API
        video = open_video(video_path)
        
        # Create detector
        detector = ContentDetector(threshold=self.threshold)
        
        # Detect all scenes
        scene_list = detect(video, detector)
        
        # Convert to shot dictionaries
        shots = []
        for i, (start_frame, end_frame) in enumerate(scene_list):
            shot = {
                'id': i + 1,
                'start_time': start_frame.get_seconds(),
                'end_time': end_frame.get_seconds(),
                'start_frame': start_frame.get_frames(),
                'end_frame': end_frame.get_frames(),
                'duration': (end_frame - start_frame).get_seconds(),
                'file': None,
                'keyframe': None
            }
            shots.append(shot)
        
        logger.info(f"Detected {len(shots)} shots")
        self._save_shot_metadata(shots, video_path)
        
        return shots
    
    def _deduplicate_shots(self, shots: List[Dict]) -> List[Dict]:
        """
        Remove duplicate shots detected in overlapping chunks.
        
        Strategy: Merge shots that are within 1 second of each other.
        
        Args:
            shots: List of all detected shots (may contain duplicates)
            
        Returns:
            Deduplicated list of shots with assigned IDs
        """
        if not shots:
            return []
        
        # Sort by start time
        sorted_shots = sorted(shots, key=lambda x: x['start_time'])
        
        deduplicated = [sorted_shots[0]]
        
        for shot in sorted_shots[1:]:
            last_shot = deduplicated[-1]
            
            # Check if this shot overlaps with the last one (within 1 second)
            if abs(shot['start_time'] - last_shot['start_time']) < 1.0:
                # Merge: use the shot with longer duration or later end time
                if shot['end_time'] > last_shot['end_time']:
                    deduplicated[-1] = shot
            else:
                deduplicated.append(shot)
        
        # Assign sequential IDs
        for i, shot in enumerate(deduplicated, start=1):
            shot['id'] = i
            shot['file'] = None
            shot['keyframe'] = None
        
        return deduplicated
    
    def _save_shot_metadata(self, shots: List[Dict], video_path: str):
        """
        Save shot metadata to JSON file.
        
        Args:
            shots: List of shot dictionaries
            video_path: Original video path (for reference)
        """
        metadata = {
            'source_video': video_path,
            'total_shots': len(shots),
            'shots': shots
        }
        
        metadata_path = self.output_dir / 'shot_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved shot metadata to {metadata_path}")
    
    def extract_shot_segments(self, video_path: str, shots: List[Dict]) -> List[Dict]:
        """
        Extract individual shot segments as separate video files using FFmpeg.
        
        Args:
            video_path: Path to source video
            shots: List of shot dictionaries
            
        Returns:
            Updated shots list with file paths
        """
        import subprocess
        
        logger.info(f"Extracting {len(shots)} shot segments...")
        
        for i, shot in enumerate(shots, start=1):
            shot_filename = f"shot_{i:04d}.mp4"
            shot_path = self.output_dir / shot_filename
            
            # FFmpeg command to extract segment
            cmd = [
                'ffmpeg',
                '-ss', str(shot['start_time']),
                '-i', video_path,
                '-t', str(shot['duration']),
                '-c', 'copy',  # Copy codec for speed
                '-y',  # Overwrite
                str(shot_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                shot['file'] = str(shot_path)
                
                if (i % 10 == 0):
                    logger.info(f"Extracted {i}/{len(shots)} shots")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract shot {i}: {e}")
                shot['file'] = None
        
        logger.info(f"Successfully extracted {sum(1 for s in shots if s['file'])} shots")
        
        # Update metadata
        self._save_shot_metadata(shots, video_path)
        
        return shots
