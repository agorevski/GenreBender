"""
Keyframe extraction module.
Extracts representative frames from video shots for analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """
    Extracts keyframes from video shots for multimodal analysis.
    Uses middle frame strategy by default.
    """
    
    def __init__(self, output_dir: str = "keyframes", quality: int = 95):
        """
        Initialize keyframe extractor.
        
        Args:
            output_dir: Directory to save keyframes
            quality: JPEG quality (1-100)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
    
    def extract_keyframes(self, video_path: str, shots: List[Dict], 
                         num_frames: int = 5) -> List[Dict]:
        """
        Extract multiple keyframes for all shots in the video.
        
        Args:
            video_path: Path to source video
            shots: List of shot dictionaries with timing information
            num_frames: Number of frames to extract per shot (default: 5)
            
        Returns:
            Updated shots list with keyframe paths
        """
        logger.info(f"Extracting {num_frames} keyframes per shot from {len(shots)} shots...")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return shots
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for i, shot in enumerate(shots, start=1):
            keyframe_paths = self._extract_multiple_frames_internal(
                cap, shot, fps, shot['id'], num_frames
            )
            
            # Store multiple keyframes
            shot['keyframes'] = [str(p) for p in keyframe_paths]
            # Keep first frame as primary keyframe for backward compatibility
            shot['keyframe'] = shot['keyframes'][0] if shot['keyframes'] else None
            
            if i % 20 == 0:
                logger.info(f"Extracted {i}/{len(shots)} shot keyframes")
        
        cap.release()
        
        successful = sum(1 for s in shots if s.get('keyframes'))
        total_frames = sum(len(s.get('keyframes', [])) for s in shots)
        logger.info(f"Successfully extracted {total_frames} keyframes from {successful}/{len(shots)} shots")
        
        return shots
    
    def _extract_multiple_frames_internal(self, cap: cv2.VideoCapture, shot: Dict,
                                         fps: float, shot_id: int, 
                                         num_frames: int) -> List[Path]:
        """
        Extract multiple evenly-spaced frames from a shot.
        
        Args:
            cap: OpenCV VideoCapture object
            shot: Shot dictionary with timing
            fps: Video frame rate
            shot_id: Shot ID for filename
            num_frames: Number of frames to extract
            
        Returns:
            List of paths to extracted frames
        """
        duration = shot['end_time'] - shot['start_time']
        frame_paths = []
        
        # Handle very short shots
        if duration < 0.5 or num_frames == 1:
            # Just extract middle frame for very short shots
            middle_time = (shot['start_time'] + shot['end_time']) / 2.0
            middle_frame = int(middle_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                keyframe_filename = f"kf_{shot_id:04d}_1.jpg"
                keyframe_path = self.output_dir / keyframe_filename
                cv2.imwrite(
                    str(keyframe_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                frame_paths.append(keyframe_path)
            
            return frame_paths
        
        # Extract evenly-spaced frames
        for i in range(num_frames):
            # Calculate evenly-spaced time points
            offset = (i + 1) / (num_frames + 1)
            frame_time = shot['start_time'] + duration * offset
            frame_num = int(frame_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                keyframe_filename = f"kf_{shot_id:04d}_{i+1}.jpg"
                keyframe_path = self.output_dir / keyframe_filename
                
                cv2.imwrite(
                    str(keyframe_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                
                frame_paths.append(keyframe_path)
            else:
                logger.warning(f"Failed to extract frame {i+1}/{num_frames} for shot {shot_id}")
        
        return frame_paths
    
    def _extract_middle_frame(self, cap: cv2.VideoCapture, shot: Dict, 
                             fps: float, shot_id: int) -> Optional[Path]:
        """
        Extract the middle frame of a shot (legacy method).
        
        Args:
            cap: OpenCV VideoCapture object
            shot: Shot dictionary with timing
            fps: Video frame rate
            shot_id: Shot ID for filename
            
        Returns:
            Path to saved keyframe or None if failed
        """
        # Calculate middle frame
        middle_time = (shot['start_time'] + shot['end_time']) / 2.0
        middle_frame = int(middle_time * fps)
        
        # Seek to middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to extract keyframe for shot {shot_id}")
            return None
        
        # Save as JPEG
        keyframe_filename = f"kf_{shot_id:04d}.jpg"
        keyframe_path = self.output_dir / keyframe_filename
        
        cv2.imwrite(
            str(keyframe_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        
        return keyframe_path
    
    def extract_multiple_frames(self, video_path: str, shot: Dict, 
                               num_frames: int = 3) -> List[Path]:
        """
        Extract multiple evenly-spaced frames from a shot.
        Useful for temporal analysis.
        
        Args:
            video_path: Path to source video
            shot: Shot dictionary
            num_frames: Number of frames to extract
            
        Returns:
            List of paths to extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = shot['end_time'] - shot['start_time']
        
        frame_paths = []
        
        for i in range(num_frames):
            # Calculate evenly-spaced time points
            offset = (i + 1) / (num_frames + 1)
            frame_time = shot['start_time'] + duration * offset
            frame_num = int(frame_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                keyframe_filename = f"kf_{shot['id']:04d}_{i+1}.jpg"
                keyframe_path = self.output_dir / keyframe_filename
                
                cv2.imwrite(
                    str(keyframe_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                
                frame_paths.append(keyframe_path)
        
        cap.release()
        
        return frame_paths
    
    def extract_best_frame(self, video_path: str, shot: Dict) -> Optional[Path]:
        """
        Extract the 'best' frame using simple quality heuristics.
        Selects frame with highest sharpness score.
        
        Args:
            video_path: Path to source video
            shot: Shot dictionary
            
        Returns:
            Path to best keyframe or None
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(shot['start_time'] * fps)
        end_frame = int(shot['end_time'] * fps)
        
        # Sample 5 frames from the shot
        sample_frames = np.linspace(start_frame, end_frame, 5, dtype=int)
        
        best_score = -1
        best_frame = None
        
        for frame_num in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Calculate sharpness using Laplacian variance
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                
                if sharpness > best_score:
                    best_score = sharpness
                    best_frame = frame.copy()
        
        cap.release()
        
        if best_frame is not None:
            keyframe_filename = f"kf_{shot['id']:04d}.jpg"
            keyframe_path = self.output_dir / keyframe_filename
            
            cv2.imwrite(
                str(keyframe_path),
                best_frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            
            return keyframe_path
        
        return None
