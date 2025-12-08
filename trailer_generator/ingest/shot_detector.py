"""
Shot detection module with streaming support and overlap handling.
Uses PySceneDetect to identify shot boundaries in video files.
"""

import os
import json
import threading
import queue
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Callable
from scenedetect import open_video, SceneDetector, detect
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager, save_images
from scenedetect.video_splitter import is_ffmpeg_available
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
        
    def detect_shots(self, video_path: str, streaming: bool = True, 
                    frame_skip: int = 0) -> List[Dict]:
        """
        Detect shots in video file.
        
        Args:
            video_path: Path to input video file
            streaming: If True, use optimized streaming detection
            frame_skip: Number of frames to skip (0=no skip, 1=every other, etc.)
            
        Returns:
            List of shot dictionaries with metadata
        """
        if streaming:
            return self._detect_shots_streaming(video_path, frame_skip=frame_skip)
        else:
            return self._detect_shots_full(video_path)
    
    def _extraction_worker(self, video_path: str, extraction_queue: queue.Queue, 
                          stop_event: threading.Event, extracted_shots: Dict):
        """
        Background thread that extracts shots from queue as they arrive.
        
        Args:
            video_path: Path to source video
            extraction_queue: Queue receiving shots to extract
            stop_event: Event to signal worker should stop
            extracted_shots: Shared dict mapping shot_id -> extracted file path
        """
        logger.info("Extraction worker thread started")
        extraction_count = 0
        
        while not stop_event.is_set() or not extraction_queue.empty():
            try:
                # Get shot from queue with timeout
                shot = extraction_queue.get(timeout=0.5)
                
                # Extract this shot using its final ID
                shot_id = shot['id']
                shot_filename = f"shot_temp_{shot_id:04d}.mp4"
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
                    extracted_shots[shot_id] = str(shot_path)
                    extraction_count += 1
                    
                    if extraction_count % 10 == 0:
                        logger.info(f"Extracted {extraction_count} shots in background")
                        
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to extract shot {shot_id}: {e}")
                    extracted_shots[shot_id] = None
                
                extraction_queue.task_done()
                
            except queue.Empty:
                continue
        
        logger.info(f"Extraction worker thread completed. Total extracted: {extraction_count}")
    
    def _binary_search_cut_frame(self, video, detector, start_frame: int, 
                                 end_frame: int) -> int:
        """
        Use binary search to find the exact frame where a cut occurs.
        
        Args:
            video: Opened video object
            detector: ContentDetector instance
            start_frame: Start of range to search
            end_frame: End of range (cut is known to be <= this frame)
            
        Returns:
            Exact frame number where cut occurs
        """
        left, right = start_frame, end_frame
        
        while right - left > 1:
            mid = (left + right) // 2
            
            # Seek to mid frame and check
            video.seek(mid)
            frame_im = video.read()
            
            if frame_im is None or not frame_im.size:
                # Can't read frame, assume cut is before
                right = mid
                continue
            
            # Check if cut detected at this frame
            cut_list = detector.process_frame(mid, frame_im)
            is_cut = len(cut_list) > 0 if cut_list else False
            
            if is_cut:
                right = mid  # Cut at or before mid
            else:
                left = mid   # Cut after mid
        
        return right
    
    def _detect_shots_streaming(self, video_path: str, async_extraction: bool = True, 
                               frame_skip: int = 0) -> List[Dict]:
        """
        Detect shots using adaptive frame skipping with binary search refinement.
        
        Processes frames with coarse skip, then uses binary search to find exact
        cut points. Queues shots immediately for parallel extraction.
        
        Args:
            video_path: Path to video file
            async_extraction: If True, extract shots asynchronously during detection
            frame_skip: Frames to skip in coarse pass (0=no skip, 10=check every 10th)
        """
        logger.info(f"Starting streaming shot detection on {video_path}")
        logger.info(f"Async extraction: {async_extraction}, Frame skip: {frame_skip}")
        
        # Open video
        video = open_video(video_path)
        total_duration = video.duration.get_seconds()
        frame_rate = video.frame_rate
        total_frames = int(total_duration * frame_rate)
        
        logger.info(f"Total video duration: {total_duration:.2f}s ({total_frames} frames @ {frame_rate} fps)")
        
        # Set up async extraction if enabled
        extraction_queue = None
        stop_event = None
        worker_thread = None
        extracted_shots = {}
        
        if async_extraction:
            extraction_queue = queue.Queue()
            stop_event = threading.Event()
            worker_thread = threading.Thread(
                target=self._extraction_worker,
                args=(video_path, extraction_queue, stop_event, extracted_shots),
                daemon=True
            )
            worker_thread.start()
            logger.info("Started async extraction worker thread")
        
        try:
            # Frame-by-frame detection with adaptive skip
            logger.info(f"Detecting shots with frame skip={frame_skip}, binary search refinement...")
            
            detected_shots = []
            shot_id = 1
            last_cut_frame = 0
            current_frame = 0
            skip_amount = max(1, frame_skip)  # Ensure at least 1
            
            # Create detector
            detector = ContentDetector(threshold=self.threshold)
            
            while current_frame < total_frames:
                # Read current frame
                video.seek(current_frame)
                frame_im = video.read()
                
                if frame_im is None or not frame_im.size:
                    break
                
                # Check for cut at this frame
                cut_list = detector.process_frame(current_frame, frame_im)
                is_cut = len(cut_list) > 0 if cut_list else False
                
                if is_cut:
                    # Cut detected! Refine to find exact frame if we skipped
                    if skip_amount > 1:
                        exact_cut_frame = self._binary_search_cut_frame(
                            video, detector,
                            max(last_cut_frame, current_frame - skip_amount),
                            current_frame
                        )
                    else:
                        exact_cut_frame = current_frame
                    
                    # Create shot from last cut to this cut
                    start_time = last_cut_frame / frame_rate
                    end_time = exact_cut_frame / frame_rate
                    duration = end_time - start_time
                    
                    shot = {
                        'id': shot_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_frame': last_cut_frame,
                        'end_frame': exact_cut_frame,
                        'duration': duration,
                        'file': None,
                        'keyframe': None
                    }
                    
                    detected_shots.append(shot)
                    
                    # Queue immediately for extraction!
                    if extraction_queue is not None:
                        extraction_queue.put(shot.copy())
                        logger.debug(f"Shot {shot_id}: {start_time:.2f}s - {end_time:.2f}s queued for extraction")
                    
                    shot_id += 1
                    last_cut_frame = exact_cut_frame
                
                # Progress logging
                if current_frame % (skip_amount * 100) == 0:
                    progress_pct = (current_frame / total_frames) * 100
                    logger.info(f"Progress: {progress_pct:.1f}% ({current_frame}/{total_frames} frames), {len(detected_shots)} shots found")
                
                # Jump to next frame (coarse skip)
                current_frame += skip_amount
            
            # Handle final shot (from last cut to end)
            if last_cut_frame < total_frames:
                start_time = last_cut_frame / frame_rate
                end_time = total_frames / frame_rate
                duration = end_time - start_time
                
                shot = {
                    'id': shot_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_frame': last_cut_frame,
                    'end_frame': total_frames,
                    'duration': duration,
                    'file': None,
                    'keyframe': None
                }
                
                detected_shots.append(shot)
                
                if extraction_queue is not None:
                    extraction_queue.put(shot.copy())
                    logger.info(f"Final shot {shot_id}: {start_time:.2f}s - {end_time:.2f}s queued")
            
            logger.info(f"Detected {len(detected_shots)} total shots")
            
            # Wait for extraction worker to complete
            if async_extraction and worker_thread:
                logger.info("Waiting for extraction worker to complete...")
                stop_event.set()
                extraction_queue.join()
                worker_thread.join(timeout=60)
                
                if worker_thread.is_alive():
                    logger.warning("Extraction worker did not complete in time")
                else:
                    logger.info("Extraction worker completed successfully")
                
                # Map extracted files back to shots
                for shot in detected_shots:
                    shot_id = shot['id']
                    if shot_id in extracted_shots:
                        temp_file = extracted_shots[shot_id]
                        if temp_file and Path(temp_file).exists():
                            # Rename temp file to final name
                            final_filename = f"shot_{shot_id:04d}.mp4"
                            final_path = self.output_dir / final_filename
                            try:
                                Path(temp_file).rename(final_path)
                                shot['file'] = str(final_path)
                            except Exception as e:
                                logger.error(f"Failed to rename {temp_file}: {e}")
                                shot['file'] = None
            
            # Save metadata
            self._save_shot_metadata(detected_shots, video_path)
            
            return detected_shots
            
        except Exception as e:
            # Clean up worker thread on error
            if async_extraction and stop_event:
                stop_event.set()
            raise e
    
    def _detect_chunk(self, video_path: str, start_time: float, 
                     end_time: float, extraction_queue: Optional[queue.Queue] = None,
                     temp_id_counter: Optional[Dict] = None) -> List[Dict]:
        """
        Detect shots within a specific time range with immediate extraction queuing.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            extraction_queue: Optional queue for immediate shot extraction
            temp_id_counter: Optional dict with 'value' key for tracking temp IDs
            
        Returns:
            List of shots detected in this chunk
        """
        # Create detector
        detector = ContentDetector(threshold=self.threshold)
        
        # Detect scenes in the specified time range (pass path string, not video object)
        scene_list = detect(video_path, detector, start_time=start_time, end_time=end_time)
        
        # Convert to shot dictionaries and queue immediately for extraction
        shots = []
        for i, (start_frame, end_frame) in enumerate(scene_list):
            shot = {
                'start_time': start_frame.get_seconds(),
                'end_time': end_frame.get_seconds(),
                'start_frame': start_frame.get_frames(),
                'end_frame': end_frame.get_frames(),
                'duration': (end_frame - start_frame).get_seconds()
            }
            shots.append(shot)
            
            # Queue immediately for extraction if callback provided
            if extraction_queue is not None and temp_id_counter is not None:
                shot['temp_id'] = temp_id_counter['value']
                temp_id_counter['value'] += 1
                extraction_queue.put(shot)
        
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
    
    def _map_extracted_files(self, final_shots: List[Dict], 
                            temp_shots: List[Dict], 
                            extracted_files: Dict[int, str]):
        """
        Map extracted temporary files to final deduplicated shot IDs.
        Renames files from shot_temp_XXXX.mp4 to shot_YYYY.mp4.
        
        Args:
            final_shots: Deduplicated shots with final IDs
            temp_shots: Original shots with temp_ids
            extracted_files: Dict mapping temp_id -> extracted file path
        """
        logger.info("Mapping extracted files to final shot IDs...")
        
        # Build mapping from (start_time, end_time) to temp_id
        temp_shot_map = {
            (shot['start_time'], shot['end_time']): shot.get('temp_id')
            for shot in temp_shots if 'temp_id' in shot
        }
        
        mapped_count = 0
        renamed_count = 0
        
        for final_shot in final_shots:
            # Find matching temp shot by time
            key = (final_shot['start_time'], final_shot['end_time'])
            temp_id = temp_shot_map.get(key)
            
            if temp_id and temp_id in extracted_files:
                temp_file = extracted_files[temp_id]
                
                if temp_file and Path(temp_file).exists():
                    # Rename to final filename
                    final_filename = f"shot_{final_shot['id']:04d}.mp4"
                    final_path = self.output_dir / final_filename
                    
                    try:
                        Path(temp_file).rename(final_path)
                        final_shot['file'] = str(final_path)
                        renamed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to rename {temp_file} to {final_path}: {e}")
                        final_shot['file'] = None
                else:
                    logger.warning(f"Temp file not found for shot {final_shot['id']}")
                    final_shot['file'] = None
                
                mapped_count += 1
            else:
                logger.warning(f"No extracted file for shot {final_shot['id']}")
                final_shot['file'] = None
        
        logger.info(f"Mapped {mapped_count} shots, renamed {renamed_count} files")
    
    def _cleanup_temp_files(self, extracted_files: Dict[int, str], 
                           final_shots: List[Dict]):
        """
        Clean up temporary files that weren't mapped to final shots.
        This happens when shots are deduplicated/merged.
        
        Args:
            extracted_files: Dict of all extracted temp files
            final_shots: List of final deduplicated shots
        """
        # Get set of temp files that were successfully mapped
        mapped_files = {shot.get('file') for shot in final_shots if shot.get('file')}
        
        # Find and delete unmapped temp files
        deleted_count = 0
        for temp_id, temp_file in extracted_files.items():
            if temp_file and temp_file not in mapped_files:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} unmapped temporary files")
    
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
