"""
Shot detection module using PySceneDetect.
"""

import os
import json
import threading
import queue
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from scenedetect import open_video, SceneDetector, detect
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager, save_images
from scenedetect.video_splitter import is_ffmpeg_available
import logging

logger = logging.getLogger(__name__)


class ShotDetector:
    """
    Detects shots in video files using PySceneDetect.
    """
    
    def __init__(self, threshold: float = 27.0, chunk_duration: int = 30, 
                 overlap: int = 5, output_dir: str = "shots",
                 parallel_workers: int = 0, chunk_overlap: float = 5.0,
                 min_chunk_duration: float = 30.0):
        """Initialize shot detector.

        Args:
            threshold: Content detection threshold. Defaults to 27.0.
            chunk_duration: Duration of each processing chunk in seconds (legacy).
                Defaults to 30.
            overlap: Overlap duration between chunks (legacy). Defaults to 5.
            output_dir: Directory to save shot segments. Defaults to "shots".
            parallel_workers: Number of parallel workers. 0 means auto-detect
                based on CPU count. Defaults to 0.
            chunk_overlap: Overlap in seconds for parallel chunk processing.
                Defaults to 5.0.
            min_chunk_duration: Minimum duration per chunk in seconds.
                Defaults to 30.0.
        """
        self.threshold = threshold
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parallel processing settings
        self.max_parallel_workers = parallel_workers if parallel_workers > 0 else multiprocessing.cpu_count()
        self.chunk_overlap = chunk_overlap
        self.min_chunk_duration = min_chunk_duration
        
        logger.info(f"Initialized ShotDetector with max {self.max_parallel_workers} workers, min chunk duration {min_chunk_duration}s")
        
    def detect_shots(self, video_path: str, streaming: bool = True, 
                    frame_skip: int = 0, parallel: bool = True) -> List[Dict]:
        """Detect shots in video file using PySceneDetect.

        Args:
            video_path: Path to input video file.
            streaming: If True, use optimized streaming detection. Ignored if
                parallel is True. Defaults to True.
            frame_skip: Number of frames to skip during detection.
                Defaults to 0.
            parallel: If True, use parallel chunk-based processing.
                Defaults to True.

        Returns:
            List of shot dictionaries containing metadata such as start_time,
            end_time, duration, start_frame, end_frame, file path, and keyframe.
        """
        if parallel and self.max_parallel_workers > 1:
            return self._detect_shots_parallel(video_path, frame_skip=frame_skip)
        elif streaming:
            return self._detect_shots_streaming(video_path, frame_skip=frame_skip)
        else:
            return self._detect_shots_full(video_path)
    
    def _extraction_worker(self, video_path: str, extraction_queue: queue.Queue, 
                          stop_event: threading.Event, extracted_shots: Dict):
        """Background thread that extracts shots from queue as they arrive.

        This worker runs in a separate thread and continuously processes shots
        from the queue, extracting each as a video segment using FFmpeg.

        Args:
            video_path: Path to source video file.
            extraction_queue: Queue receiving shot dictionaries to extract.
            stop_event: Threading event to signal worker should stop.
            extracted_shots: Shared dict mapping shot_id to extracted file path.
                Updated in-place as shots are extracted.
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
        """Use binary search to find the exact frame where a cut occurs.

        Performs binary search within a frame range to locate the precise
        frame where a scene cut happens.

        Args:
            video: Opened video object from scenedetect.open_video().
            detector: ContentDetector instance for detecting cuts.
            start_frame: Start of range to search (inclusive).
            end_frame: End of range to search. The cut is known to be at or
                before this frame.

        Returns:
            Exact frame number where the cut occurs.
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
        """Detect shots using adaptive frame skipping with binary search refinement.

        Processes frames with a coarse skip interval, then uses binary search
        to find exact cut points. Queues shots immediately for parallel
        extraction in a background thread.

        Args:
            video_path: Path to video file.
            async_extraction: If True, extract shots asynchronously during
                detection using a background thread. Defaults to True.
            frame_skip: Frames to skip in coarse pass. 0 means no skip,
                10 means check every 10th frame. Defaults to 0.

        Returns:
            List of shot dictionaries with metadata and extracted file paths.
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
                
                if frame_im is False or frame_im is None:
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
            
            # If no shots detected at all, treat entire video as one shot
            if len(detected_shots) == 0:
                logger.warning("No scene changes detected in streaming mode - treating entire video as single shot")
                shot = {
                    'id': 1,
                    'start_time': 0.0,
                    'end_time': total_duration,
                    'start_frame': 0,
                    'end_frame': total_frames,
                    'duration': total_duration,
                    'file': None,
                    'keyframe': None
                }
                detected_shots.append(shot)
                
                if extraction_queue is not None:
                    extraction_queue.put(shot.copy())
                    logger.info(f"Single shot: 0.00s - {total_duration:.2f}s queued")
            
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
    
    @staticmethod
    def _detect_chunk_worker(args: Tuple) -> Tuple[int, List[Dict]]:
        """Worker function for parallel chunk detection.

        Must be static for multiprocessing compatibility. Detects shots
        within a specified time range of the video.

        Args:
            args: Tuple containing (chunk_id, video_path, start_time,
                end_time, threshold) where chunk_id is the identifier for
                this chunk, video_path is the path to the video file,
                start_time and end_time define the time range in seconds,
                and threshold is the content detection threshold.

        Returns:
            Tuple of (chunk_id, list of detected shot dictionaries).
        """
        chunk_id, video_path, start_time, end_time, threshold = args
        
        try:
            # Create detector for this worker
            detector = ContentDetector(threshold=threshold)
            
            # Detect scenes in chunk
            scene_list = detect(video_path, detector, start_time=start_time, end_time=end_time)
            
            # Convert to shot dictionaries
            shots = []
            for start_frame, end_frame in scene_list:
                shot = {
                    'start_time': start_frame.get_seconds(),
                    'end_time': end_frame.get_seconds(),
                    'start_frame': start_frame.get_frames(),
                    'end_frame': end_frame.get_frames(),
                    'duration': (end_frame - start_frame).get_seconds(),
                    'chunk_id': chunk_id
                }
                shots.append(shot)
            
            return (chunk_id, shots)
            
        except Exception as e:
            logger.error(f"Error in chunk {chunk_id}: {e}")
            return (chunk_id, [])
    
    @staticmethod
    def _extract_shot_worker(args: Tuple) -> Tuple[int, Optional[str]]:
        """Worker function for parallel shot extraction.

        Must be static for multiprocessing compatibility. Extracts a single
        shot segment from the video using FFmpeg.

        Args:
            args: Tuple containing (shot_temp_id, video_path, start_time,
                duration, output_dir) where shot_temp_id is the temporary
                identifier for this shot, video_path is the source video,
                start_time is the shot start in seconds, duration is the
                shot length in seconds, and output_dir is where to save.

        Returns:
            Tuple of (shot_temp_id, extracted_file_path) on success, or
            (shot_temp_id, None) on failure.
        """
        shot_temp_id, video_path, start_time, duration, output_dir = args
        
        try:
            shot_filename = f"shot_temp_{shot_temp_id:04d}.mp4"
            shot_path = Path(output_dir) / shot_filename
            
            # FFmpeg command to extract segment
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c', 'copy',
                '-y',
                str(shot_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return (shot_temp_id, str(shot_path))
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract shot {shot_temp_id}: {e}")
            return (shot_temp_id, None)
        except Exception as e:
            logger.error(f"Unexpected error extracting shot {shot_temp_id}: {e}")
            return (shot_temp_id, None)
    
    def _detect_shots_parallel(self, video_path: str, frame_skip: int = 0) -> List[Dict]:
        """Detect shots using parallel chunk-based processing across multiple CPUs.

        Uses adaptive parallelization based on video duration. Splits the video
        into chunks, processes them in parallel, deduplicates overlapping shots,
        and extracts all shots concurrently.

        Args:
            video_path: Path to video file.
            frame_skip: Frames to skip. Currently not used in parallel mode.
                Defaults to 0.

        Returns:
            List of deduplicated shot dictionaries with extracted file paths.
        """
        # Get video metadata
        video = open_video(video_path)
        total_duration = video.duration.get_seconds()
        frame_rate = video.frame_rate
        total_frames = int(total_duration * frame_rate)
        video = None  # Close video
        
        logger.info(f"Video duration: {total_duration:.2f}s ({total_frames} frames @ {frame_rate} fps)")
        
        # Adaptive parallelization: calculate effective workers based on video duration
        # Ensure each chunk has at least min_chunk_duration seconds
        max_possible_chunks = int(total_duration / self.min_chunk_duration)
        effective_workers = min(self.max_parallel_workers, max(1, max_possible_chunks))
        
        # Fall back to streaming mode for very short videos
        if effective_workers == 1 or total_duration < self.min_chunk_duration * 2:
            logger.info(f"Video too short ({total_duration:.2f}s) for parallelization - using streaming mode")
            return self._detect_shots_streaming(video_path, async_extraction=True, frame_skip=frame_skip)
        
        logger.info(f"Using {effective_workers} workers (max={self.max_parallel_workers}, video={total_duration:.2f}s, min_chunk={self.min_chunk_duration}s)")
        
        # Calculate chunk boundaries with overlap
        chunk_duration = total_duration / effective_workers
        chunks = []
        
        for i in range(effective_workers):
            start = max(0, i * chunk_duration - self.chunk_overlap)
            end = min(total_duration, (i + 1) * chunk_duration + self.chunk_overlap)
            
            # Skip empty chunks
            if end - start > 0.1:
                chunks.append((i, video_path, start, end, self.threshold))
        
        logger.info(f"Split video into {len(chunks)} chunks (~{chunk_duration:.1f}s each) for parallel processing")
        
        # Process chunks in parallel
        all_shots = []
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_chunk = {executor.submit(ShotDetector._detect_chunk_worker, chunk): chunk[0] 
                              for chunk in chunks}
            
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    chunk_id_result, shots = future.result()
                    all_shots.extend(shots)
                    logger.info(f"Chunk {chunk_id} completed: {len(shots)} shots detected")
                except Exception as e:
                    logger.error(f"Chunk {chunk_id} failed: {e}")
        
        logger.info(f"Parallel detection complete: {len(all_shots)} total shots (before deduplication)")
        
        # Handle case where no scene changes were detected (single continuous shot)
        if len(all_shots) == 0:
            logger.warning("No scene changes detected - treating entire video as single shot")
            all_shots = [{
                'start_time': 0.0,
                'end_time': total_duration,
                'start_frame': 0,
                'end_frame': total_frames,
                'duration': total_duration,
                'chunk_id': -1
            }]
        
        # Deduplicate shots from overlapping chunks
        deduplicated_shots = self._deduplicate_shots(all_shots)
        logger.info(f"After deduplication: {len(deduplicated_shots)} unique shots")
        
        # Assign temp IDs for extraction
        for i, shot in enumerate(deduplicated_shots):
            shot['temp_id'] = i
        
        # Extract shots in parallel
        extraction_tasks = [
            (shot['temp_id'], video_path, shot['start_time'], shot['duration'], str(self.output_dir))
            for shot in deduplicated_shots
        ]
        
        logger.info(f"Extracting {len(extraction_tasks)} shots in parallel...")
        extracted_files = {}
        
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_shot = {executor.submit(ShotDetector._extract_shot_worker, task): task[0] 
                             for task in extraction_tasks}
            
            completed = 0
            for future in as_completed(future_to_shot):
                shot_temp_id = future_to_shot[future]
                try:
                    temp_id, file_path = future.result()
                    extracted_files[temp_id] = file_path
                    completed += 1
                    
                    if completed % 10 == 0:
                        logger.info(f"Extracted {completed}/{len(extraction_tasks)} shots")
                        
                except Exception as e:
                    logger.error(f"Shot {shot_temp_id} extraction failed: {e}")
                    extracted_files[shot_temp_id] = None
        
        logger.info(f"Parallel extraction complete: {len([f for f in extracted_files.values() if f])} successful")
        
        # Map extracted files to final shot IDs and rename
        self._map_extracted_files(deduplicated_shots, deduplicated_shots, extracted_files)
        
        # Clean up unmapped temp files
        self._cleanup_temp_files(extracted_files, deduplicated_shots)
        
        # Save metadata
        self._save_shot_metadata(deduplicated_shots, video_path)
        
        return deduplicated_shots
    
    def _detect_chunk(self, video_path: str, start_time: float, 
                     end_time: float, extraction_queue: Optional[queue.Queue] = None,
                     temp_id_counter: Optional[Dict] = None) -> List[Dict]:
        """Detect shots within a specific time range with immediate extraction queuing.

        Uses PySceneDetect to find scene changes within the specified time
        range and optionally queues detected shots for extraction.

        Args:
            video_path: Path to video file.
            start_time: Start time in seconds for the detection range.
            end_time: End time in seconds for the detection range.
            extraction_queue: Optional queue for immediate shot extraction.
                If provided, detected shots are added to this queue.
            temp_id_counter: Optional dict with 'value' key for tracking
                temporary IDs. Incremented for each shot detected.

        Returns:
            List of shot dictionaries detected in this chunk.
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
        """Detect all shots in video without chunking.

        Processes the entire video in a single pass. Best suited for
        shorter videos where parallel processing overhead is not worthwhile.

        Args:
            video_path: Path to video file.

        Returns:
            List of shot dictionaries containing id, start_time, end_time,
            start_frame, end_frame, duration, file, and keyframe fields.
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
        """Remove duplicate shots detected in overlapping chunks.

        Merges shots that are very close in time (within 0.1 second). When
        duplicates are found, keeps the shot with longer duration.

        Args:
            shots: List of all detected shots, which may contain duplicates
                from overlapping chunk processing.

        Returns:
            Deduplicated list of shots with sequentially assigned IDs.
        """
        if not shots:
            return []
        
        # Sort by start time
        sorted_shots = sorted(shots, key=lambda x: x['start_time'])
        
        deduplicated = [sorted_shots[0]]
        
        for shot in sorted_shots[1:]:
            last_shot = deduplicated[-1]
            
            # Check if this shot is a duplicate (start time within 0.1 second)
            # This catches shots detected in overlapping chunks
            time_diff = abs(shot['start_time'] - last_shot['start_time'])
            
            if time_diff < 0.1:
                # Duplicate - keep the one with longer duration
                if shot['duration'] > last_shot['duration']:
                    deduplicated[-1] = shot
            else:
                # Different shot - add it
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
        """Map extracted temporary files to final deduplicated shot IDs.

        Renames files from shot_temp_XXXX.mp4 to shot_YYYY.mp4 format,
        matching temporary extractions to their final shot assignments.

        Args:
            final_shots: Deduplicated shots with final IDs. Updated in-place
                with 'file' field containing the renamed file path.
            temp_shots: Original shots with temp_ids used during extraction.
            extracted_files: Dict mapping temp_id to extracted file path.
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
        """Clean up temporary files that weren't mapped to final shots.

        Removes orphaned temporary files that remain after deduplication
        when some shots are merged or discarded.

        Args:
            extracted_files: Dict mapping temp_id to extracted temp file paths.
            final_shots: List of final deduplicated shots with mapped files.
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
        """Save shot metadata to JSON file.

        Writes shot information to 'shot_metadata.json' in the output
        directory, including source video reference and all shot details.

        Args:
            shots: List of shot dictionaries to save.
            video_path: Original video path, stored for reference.
        """
        # Ensure shot_path field is populated from file field
        for shot in shots:
            if 'shot_path' not in shot or not shot['shot_path']:
                shot['shot_path'] = shot.get('file', '')
        
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
        """Extract individual shot segments as separate video files using FFmpeg.

        Uses stream copy mode for fast extraction without re-encoding.
        Saves each shot as shot_XXXX.mp4 in the output directory.

        Args:
            video_path: Path to source video file.
            shots: List of shot dictionaries containing start_time and
                duration information.

        Returns:
            Updated shots list with 'file' field populated with extracted
            video segment paths.
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
