"""
Subtitle extraction and processing for shot enrichment.
Parses SRT files and maps dialogue to shots.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

try:
    import pysrt
except ImportError:
    pysrt = None

logger = logging.getLogger(__name__)


class SubtitleExtractor:
    """
    Extracts and processes subtitle data from SRT files.
    Maps subtitles to video shots for enhanced narrative generation.
    """
    
    def __init__(self, min_dialogue_duration: float = 0.3):
        """
        Initialize subtitle extractor.
        
        Args:
            min_dialogue_duration: Minimum duration (seconds) for subtitle to be considered
        """
        if pysrt is None:
            logger.warning("pysrt library not installed. Subtitle extraction will be disabled.")
            logger.warning("Install with: pip install pysrt")
        
        self.min_dialogue_duration = min_dialogue_duration
    
    def load_srt(self, srt_path: str) -> Optional[List]:
        """
        Load and parse SRT file.
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            List of subtitle entries, or None if failed
        """
        if pysrt is None:
            logger.error("pysrt not available, cannot load SRT file")
            return None
        
        try:
            subs = pysrt.open(srt_path, encoding='utf-8')
            logger.info(f"Loaded {len(subs)} subtitle entries from {srt_path}")
            return subs
        except FileNotFoundError:
            logger.error(f"SRT file not found: {srt_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse SRT file: {e}")
            return None
    
    def find_srt_file(self, video_path: str, explicit_srt: Optional[str] = None) -> Optional[str]:
        """
        Find SRT file for video.
        
        Search order:
        1. Explicit SRT path if provided
        2. Same directory as video with same basename
        3. Output directory
        
        Args:
            video_path: Path to video file
            explicit_srt: Explicitly provided SRT path
            
        Returns:
            Path to SRT file, or None if not found
        """
        # Check explicit path first
        if explicit_srt:
            if Path(explicit_srt).exists():
                logger.info(f"Using explicit SRT file: {explicit_srt}")
                return explicit_srt
            else:
                logger.warning(f"Explicit SRT file not found: {explicit_srt}")
        
        # Check same directory as video
        video_path_obj = Path(video_path)
        srt_path = video_path_obj.with_suffix('.srt')
        if srt_path.exists():
            logger.info(f"Found SRT file: {srt_path}")
            return str(srt_path)
        
        logger.warning(f"No SRT file found for {video_path}")
        return None
    
    def map_to_shots(self, subtitles: List, shots: List[Dict]) -> List[Dict]:
        """
        Map subtitle entries to shots based on temporal overlap.
        
        Args:
            subtitles: List of pysrt subtitle entries
            shots: List of shot dictionaries with start_time and end_time
            
        Returns:
            Enhanced shots with subtitle data
        """
        if not subtitles:
            logger.warning("No subtitles to map")
            return shots
        
        logger.info(f"Mapping {len(subtitles)} subtitles to {len(shots)} shots")
        
        # Create index for fast lookup
        subtitle_index = []
        for sub in subtitles:
            start_sec = self._timecode_to_seconds(sub.start)
            end_sec = self._timecode_to_seconds(sub.end)
            duration = end_sec - start_sec
            
            # Skip very short subtitles
            if duration < self.min_dialogue_duration:
                continue
            
            subtitle_index.append({
                'index': sub.index,
                'start': start_sec,
                'end': end_sec,
                'text': sub.text,
                'cleaned_text': self._clean_text(sub.text),
                'duration': duration
            })
        
        # Map to shots
        enriched_shots = []
        for shot in shots:
            shot_start = shot['start_time']
            shot_end = shot['end_time']
            
            # Find overlapping subtitles
            overlapping_subs = []
            for sub in subtitle_index:
                # Check for temporal overlap
                if self._has_overlap(shot_start, shot_end, sub['start'], sub['end']):
                    overlapping_subs.append(sub)
            
            # Add subtitle data to shot
            if overlapping_subs:
                subtitle_data = self._create_subtitle_data(overlapping_subs, shot)
                shot['subtitles'] = subtitle_data
            else:
                shot['subtitles'] = {
                    'has_dialogue': False,
                    'dialogue': None,
                    'subtitle_entries': [],
                    'word_count': 0,
                    'dialogue_density': 0.0,
                    'emotional_markers': {
                        'questions': 0,
                        'exclamations': 0,
                        'all_caps_words': 0
                    }
                }
            
            enriched_shots.append(shot)
        
        # Log statistics
        dialogue_count = sum(1 for s in enriched_shots if s['subtitles']['has_dialogue'])
        logger.info(f"Mapped subtitles: {dialogue_count}/{len(enriched_shots)} shots have dialogue")
        
        return enriched_shots
    
    def _timecode_to_seconds(self, timecode) -> float:
        """
        Convert pysrt timecode to seconds.
        
        Args:
            timecode: pysrt SubRipTime object
            
        Returns:
            Time in seconds
        """
        return (timecode.hours * 3600 + 
                timecode.minutes * 60 + 
                timecode.seconds + 
                timecode.milliseconds / 1000.0)
    
    def _has_overlap(self, start1: float, end1: float, 
                    start2: float, end2: float) -> bool:
        """
        Check if two time ranges overlap.
        
        Args:
            start1, end1: First time range
            start2, end2: Second time range
            
        Returns:
            True if ranges overlap
        """
        return start1 < end2 and start2 < end1
    
    def _clean_text(self, text: str) -> str:
        """
        Clean subtitle text by removing formatting tags and normalizing.
        
        Args:
            text: Raw subtitle text
            
        Returns:
            Cleaned text
        """
        # Remove HTML-style tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove formatting characters
        text = text.replace('\n', ' ')
        text = text.replace('\r', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _create_subtitle_data(self, subtitle_entries: List[Dict], 
                             shot: Dict) -> Dict:
        """
        Create subtitle data structure for shot.
        
        Args:
            subtitle_entries: List of subtitle entry dictionaries
            shot: Shot dictionary
            
        Returns:
            Subtitle data dictionary
        """
        # Combine all dialogue
        combined_dialogue = ' '.join(sub['cleaned_text'] for sub in subtitle_entries)
        
        # Calculate metrics
        word_count = len(combined_dialogue.split())
        dialogue_density = word_count / shot['duration'] if shot['duration'] > 0 else 0.0
        
        # Detect emotional markers
        emotional_markers = self._detect_emotional_markers(combined_dialogue)
        
        return {
            'has_dialogue': True,
            'dialogue': combined_dialogue,
            'subtitle_entries': subtitle_entries,
            'word_count': word_count,
            'dialogue_density': dialogue_density,
            'emotional_markers': emotional_markers
        }
    
    def _detect_emotional_markers(self, text: str) -> Dict:
        """
        Detect emotional markers in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of emotional marker counts
        """
        return {
            'questions': text.count('?'),
            'exclamations': text.count('!'),
            'all_caps_words': len([w for w in text.split() if w.isupper() and len(w) > 1])
        }
    
    def get_dialogue_summary(self, shots: List[Dict]) -> Dict:
        """
        Get summary statistics about dialogue coverage.
        
        Args:
            shots: List of enriched shots
            
        Returns:
            Summary statistics dictionary
        """
        total_shots = len(shots)
        dialogue_shots = sum(1 for s in shots if s.get('subtitles', {}).get('has_dialogue', False))
        
        total_words = sum(s.get('subtitles', {}).get('word_count', 0) for s in shots)
        avg_density = sum(s.get('subtitles', {}).get('dialogue_density', 0) for s in shots) / total_shots if total_shots > 0 else 0
        
        return {
            'total_shots': total_shots,
            'dialogue_shots': dialogue_shots,
            'dialogue_coverage': dialogue_shots / total_shots if total_shots > 0 else 0,
            'total_words': total_words,
            'avg_dialogue_density': avg_density
        }
