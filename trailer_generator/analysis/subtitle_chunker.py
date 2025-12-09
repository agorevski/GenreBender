"""
Subtitle Chunker
================

Splits subtitles into time-based chunks with overlap for hierarchical processing.
"""

import logging
from typing import List, Dict
from datetime import timedelta

logger = logging.getLogger(__name__)


class SubtitleChunker:
    """
    Chunks subtitle entries into time-based segments with overlap.
    """
    
    def __init__(self, chunk_duration_minutes: int = 15, overlap_seconds: int = 30):
        """
        Initialize subtitle chunker.
        
        Args:
            chunk_duration_minutes: Duration of each chunk in minutes
            overlap_seconds: Overlap between consecutive chunks in seconds
        """
        self.chunk_duration = chunk_duration_minutes * 60  # Convert to seconds
        self.overlap = overlap_seconds
        logger.info(f"Initialized SubtitleChunker: {chunk_duration_minutes}min chunks, {overlap_seconds}s overlap")
    
    def chunk_subtitle_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Chunk subtitle entries by time windows with overlap.
        
        Args:
            entries: List of subtitle dicts with 'start_time', 'end_time', 'text'
                    Times should be in seconds (float)
        
        Returns:
            List of chunk dicts:
            {
                'chunk_id': int,
                'start_time': float,
                'end_time': float,
                'entries': List[Dict],
                'transcript': str,
                'entry_count': int
            }
        """
        if not entries:
            logger.warning("No subtitle entries to chunk")
            return []
        
        # Get total duration
        last_entry = max(entries, key=lambda x: x['end_time'])
        total_duration = last_entry['end_time']
        
        logger.info(f"Chunking {len(entries)} entries spanning {total_duration:.1f} seconds")
        
        chunks = []
        chunk_id = 1
        current_start = 0
        
        while current_start < total_duration:
            # Calculate chunk boundaries
            chunk_end = min(current_start + self.chunk_duration, total_duration)
            
            # Find entries within this chunk
            chunk_entries = [
                entry for entry in entries
                if entry['start_time'] < chunk_end and entry['end_time'] > current_start
            ]
            
            if chunk_entries:
                # Generate transcript
                transcript = self._format_transcript(chunk_entries)
                
                chunk = {
                    'chunk_id': chunk_id,
                    'start_time': current_start,
                    'end_time': chunk_end,
                    'entries': chunk_entries,
                    'transcript': transcript,
                    'entry_count': len(chunk_entries)
                }
                
                chunks.append(chunk)
                logger.debug(f"Chunk {chunk_id}: {self._format_time(current_start)} - {self._format_time(chunk_end)} ({len(chunk_entries)} entries)")
                chunk_id += 1
            
            # Move to next chunk (with overlap)
            current_start += self.chunk_duration - self.overlap
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _format_transcript(self, entries: List[Dict]) -> str:
        """
        Format subtitle entries into a readable transcript.
        
        Args:
            entries: List of subtitle entries
        
        Returns:
            Formatted transcript string
        """
        lines = []
        for entry in entries:
            timestamp = self._format_time(entry['start_time'])
            lines.append(f"[{timestamp}] {entry['text']}")
        
        return '\n'.join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """
        Format seconds as HH:MM:SS.
        
        Args:
            seconds: Time in seconds
        
        Returns:
            Formatted time string
        """
        td = timedelta(seconds=int(seconds))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_chunk_summary(self, chunks: List[Dict]) -> Dict:
        """
        Get summary statistics for chunks.
        
        Args:
            chunks: List of chunk dicts
        
        Returns:
            Summary dict with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_entries': 0,
                'total_duration': 0,
                'avg_entries_per_chunk': 0
            }
        
        total_entries = sum(c['entry_count'] for c in chunks)
        total_duration = chunks[-1]['end_time']
        avg_entries = total_entries / len(chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_entries': total_entries,
            'total_duration': total_duration,
            'avg_entries_per_chunk': round(avg_entries, 1),
            'chunk_duration_minutes': self.chunk_duration / 60,
            'overlap_seconds': self.overlap
        }
