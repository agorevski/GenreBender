#!/usr/bin/env python3
"""
Test script for subtitle_parser utility
Tests extraction of text from SRT file
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utilities.subtitle_parser import SubtitleParser

def main():
    """Test the SubtitleParser utility by loading and analyzing an SRT file.

    Loads a sample SRT file, displays parsing statistics including total entries,
    word counts, and durations, then prints the full transcript.

    Returns:
        None

    Raises:
        SystemExit: If the SRT file fails to load.
    """
    # Test file path
    srt_file = "test_files/Dumb.And.Dumber.1994.1080p.BluRay.x264-CiNEFiLE.ENG.srt"
    
    print("=" * 70)
    print("SUBTITLE PARSER TEST")
    print("=" * 70)
    print(f"Testing file: {srt_file}\n")
    
    # Initialize parser
    parser = SubtitleParser(min_dialogue_duration=0.3)
    
    # Load SRT file
    if not parser.load_srt(srt_file):
        print("❌ Failed to load SRT file")
        sys.exit(1)
    
    print("✓ Successfully loaded SRT file\n")
    
    # Get statistics
    stats = parser.get_statistics()
    print("Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Total duration: {stats['total_duration']:.1f} seconds")
    print(f"  Avg words/entry: {stats['avg_words_per_entry']:.2f}")
    print(f"  Avg entry duration: {stats['avg_entry_duration']:.2f} seconds\n")
    
    # Get first 5 entries
    transcript = parser.get_full_transcript(include_timestamps=False)
    print(transcript)

if __name__ == '__main__':
    main()
