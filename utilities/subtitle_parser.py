"""
Subtitle Parser Utility
=======================

Pure subtitle parsing functionality for extracting dialogue from SRT files.
This is a reusable utility that can be used independently of the main pipeline.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import re

try:
    import pysrt
except ImportError:
    pysrt = None

logger = logging.getLogger(__name__)

class SubtitleParser:
    """
    Parser for SRT subtitle files.
    Provides pure parsing functionality without shot mapping.
    """
    
    def __init__(self, min_dialogue_duration: float = 0.3):
        """
        Initialize subtitle parser.
        
        Args:
            min_dialogue_duration: Minimum duration (seconds) for subtitle to be considered
        """
        if pysrt is None:
            logger.warning("pysrt library not installed. Subtitle parsing will be disabled.")
            logger.warning("Install with: pip install pysrt")
        
        self.min_dialogue_duration = min_dialogue_duration
        self._subtitles = None
        self._parsed_entries = None
    
    def load_srt(self, srt_path: str) -> bool:
        """
        Load and parse SRT file.
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            True if successful, False otherwise
        """
        if pysrt is None:
            logger.error("pysrt not available, cannot load SRT file")
            return False
        
        try:
            self._subtitles = pysrt.open(srt_path, encoding='utf-8')
            logger.info(f"Loaded {len(self._subtitles)} subtitle entries from {srt_path}")
            self._parse_entries()
            return True
        except FileNotFoundError:
            logger.error(f"SRT file not found: {srt_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to parse SRT file: {e}")
            return False
    
    def _parse_entries(self):
        """Parse subtitle entries into structured format."""
        if not self._subtitles:
            self._parsed_entries = []
            return
        
        self._parsed_entries = []
        for sub in self._subtitles:
            start_sec = self._timecode_to_seconds(sub.start)
            end_sec = self._timecode_to_seconds(sub.end)
            duration = end_sec - start_sec
            
            # Skip very short subtitles
            if duration < self.min_dialogue_duration:
                continue
            
            self._parsed_entries.append({
                'index': sub.index,
                'start_time': start_sec,
                'end_time': end_sec,
                'start_timecode': self._format_timecode(sub.start),
                'end_timecode': self._format_timecode(sub.end),
                'duration': duration,
                'text': sub.text,
                'cleaned_text': self._clean_text(sub.text)
            })
        
        logger.info(f"Parsed {len(self._parsed_entries)} subtitle entries")
    
    def get_all_entries(self) -> List[Dict]:
        """
        Get all parsed subtitle entries.
        
        Returns:
            List of subtitle entry dictionaries
        """
        if self._parsed_entries is None:
            logger.warning("No subtitles loaded")
            return []
        
        return self._parsed_entries
    
    def get_full_transcript(self, include_timestamps: bool = False) -> str:
        """
        Get full transcript as single string.
        
        Args:
            include_timestamps: Whether to include timestamps in output
            
        Returns:
            Full dialogue as string
        """
        if not self._parsed_entries:
            logger.warning("No subtitles loaded")
            return ""
        
        if include_timestamps:
            lines = []
            for entry in self._parsed_entries:
                timestamp = f"[{entry['start_timecode']}]"
                lines.append(f"{timestamp} {entry['cleaned_text']}")
            return "\n".join(lines)
        else:
            return " ".join(entry['cleaned_text'] for entry in self._parsed_entries)
    
    def get_formatted_subtitles(self) -> List[str]:
        """
        Get formatted subtitle entries with timestamps.
        
        Returns:
            List of formatted subtitle strings
        """
        if not self._parsed_entries:
            logger.warning("No subtitles loaded")
            return []
        
        formatted = []
        for entry in self._parsed_entries:
            formatted.append(
                f"[{entry['start_timecode']} --> {entry['end_timecode']}] {entry['cleaned_text']}"
            )
        
        return formatted
    
    def get_statistics(self) -> Dict:
        """
        Get subtitle statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self._parsed_entries:
            return {
                'total_entries': 0,
                'total_duration': 0.0,
                'total_words': 0,
                'avg_words_per_entry': 0.0,
                'avg_entry_duration': 0.0
            }
        
        total_words = sum(len(e['cleaned_text'].split()) for e in self._parsed_entries)
        total_duration = sum(e['duration'] for e in self._parsed_entries)
        
        return {
            'total_entries': len(self._parsed_entries),
            'total_duration': total_duration,
            'total_words': total_words,
            'avg_words_per_entry': total_words / len(self._parsed_entries),
            'avg_entry_duration': total_duration / len(self._parsed_entries)
        }
    
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
    
    def _format_timecode(self, timecode) -> str:
        """
        Format timecode as HH:MM:SS.
        
        Args:
            timecode: pysrt SubRipTime object
            
        Returns:
            Formatted timecode string
        """
        return f"{timecode.hours:02d}:{timecode.minutes:02d}:{timecode.seconds:02d}"
    
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


def load_subtitles_from_srt(srt_path: str, min_duration: float = 0.3) -> Optional[SubtitleParser]:
    """
    Convenience function to load subtitles from SRT file.
    
    Args:
        srt_path: Path to SRT file
        min_duration: Minimum subtitle duration
        
    Returns:
        SubtitleParser instance or None if failed
    """
    parser = SubtitleParser(min_dialogue_duration=min_duration)
    if parser.load_srt(srt_path):
        return parser
    return None
