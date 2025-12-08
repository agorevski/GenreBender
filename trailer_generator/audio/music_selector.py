"""
AI-powered music selection for trailers.
Analyzes timeline and selects appropriate music from library.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class MusicSelector:
    """
    Selects and structures music tracks for trailers using AI analysis.
    """
    
    def __init__(self, azure_client, genre_profile: Dict, music_library_path: str,
                 enable_ai: bool = True):
        """
        Initialize music selector.
        
        Args:
            azure_client: Azure OpenAI client (can be None if enable_ai=False)
            genre_profile: Genre-specific configuration
            music_library_path: Path to music library directory
            enable_ai: Whether to use AI for music selection
        """
        self.azure_client = azure_client
        self.genre_profile = genre_profile
        self.music_library = Path(music_library_path)
        self.enable_ai = enable_ai
        self.genre_tags = genre_profile.get('music_tags', [])
    
    def select_music(self, timeline: Dict, music_file: Optional[str] = None) -> Dict:
        """
        Select music track for trailer.
        
        Args:
            timeline: Timeline dictionary
            music_file: Optional specific music file to use
            
        Returns:
            Dictionary with music file path and structure
        """
        if music_file:
            # User specified music file
            logger.info(f"Using user-specified music: {music_file}")
            return {
                'file': music_file,
                'structure': self._generate_simple_structure(timeline),
                'source': 'user_specified'
            }
        
        # Search library for matching tracks
        available_tracks = self._scan_music_library()
        
        if not available_tracks:
            logger.warning("No music tracks found in library")
            return {
                'file': None,
                'structure': self._generate_simple_structure(timeline),
                'source': 'none'
            }
        
        # Use AI to analyze and select if enabled
        if self.enable_ai and self.azure_client:
            try:
                return self._select_with_ai(timeline, available_tracks)
            except Exception as e:
                logger.warning(f"AI music selection failed: {e}, using fallback")
                return self._select_with_rules(timeline, available_tracks)
        else:
            return self._select_with_rules(timeline, available_tracks)
    
    def _scan_music_library(self) -> List[Dict]:
        """
        Scan music library directory for available tracks.
        
        Returns:
            List of track dictionaries with metadata
        """
        if not self.music_library.exists():
            logger.warning(f"Music library not found: {self.music_library}")
            return []
        
        tracks = []
        audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.ogg']
        
        for file_path in self.music_library.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                # Extract metadata from filename
                filename = file_path.stem.lower()
                
                # Check for genre tags in filename
                matching_tags = [tag for tag in self.genre_tags if tag.lower() in filename]
                
                tracks.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'tags': matching_tags,
                    'tag_score': len(matching_tags)
                })
        
        logger.info(f"Found {len(tracks)} music tracks in library")
        return tracks
    
    def _select_with_ai(self, timeline: Dict, available_tracks: List[Dict]) -> Dict:
        """
        Use AI to analyze timeline and select optimal music.
        
        Args:
            timeline: Timeline dictionary
            available_tracks: Available music tracks
            
        Returns:
            Music selection dictionary
        """
        # Build prompt
        prompt = f"""Analyze this trailer timeline and recommend music structure:

Timeline Details:
- Total duration: {timeline.get('total_duration', 90)}s
- Number of shots: {len(timeline.get('timeline', []))}
- Genre: {self.genre_profile.get('pacing', 'measured')} pacing
- Music cues: {timeline.get('music_cues', [])}
- Required tags: {', '.join(self.genre_tags)}

Available tracks ({len(available_tracks)} total):
{self._format_track_list(available_tracks[:10])}  # Show top 10

Provide:
1. Recommended track selection (by name or tags)
2. Music structure with timestamps:
   - intro: description and duration
   - build: description and duration
   - climax: description and duration
   - outro: description and duration
3. Key tempo/beat markers for sync
4. Energy curve description

Return ONLY valid JSON:
{{
    "recommended_track": "track name or best matching tags",
    "structure": {{
        "intro": {{"start": 0, "end": 20, "description": "..."}},
        "build": {{"start": 20, "end": 60, "description": "..."}},
        "climax": {{"start": 60, "end": 80, "description": "..."}},
        "outro": {{"start": 80, "end": 90, "description": "..."}}
    }},
    "tempo_markers": [10, 25, 45, 70],
    "energy_curve": "description"
}}"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional trailer music supervisor. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Generate analysis
        response = self.azure_client.generate_structured_output(messages=messages)
        
        # Parse response
        try:
            analysis = json.loads(response)
            
            # Find best matching track
            selected_track = self._match_track(
                analysis.get('recommended_track', ''),
                available_tracks
            )
            
            if selected_track:
                logger.info(f"AI selected track: {selected_track['name']}")
                return {
                    'file': selected_track['path'],
                    'structure': analysis.get('structure', {}),
                    'tempo_markers': analysis.get('tempo_markers', []),
                    'energy_curve': analysis.get('energy_curve', ''),
                    'source': 'ai_selected',
                    'ai_analysis': analysis
                }
            else:
                logger.warning("AI recommendation did not match available tracks")
                return self._select_with_rules(timeline, available_tracks)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise
    
    def _select_with_rules(self, timeline: Dict, available_tracks: List[Dict]) -> Dict:
        """
        Select music using rule-based logic.
        
        Args:
            timeline: Timeline dictionary
            available_tracks: Available music tracks
            
        Returns:
            Music selection dictionary
        """
        # Sort tracks by tag match score
        sorted_tracks = sorted(available_tracks, key=lambda t: t['tag_score'], reverse=True)
        
        if not sorted_tracks:
            return {
                'file': None,
                'structure': self._generate_simple_structure(timeline),
                'source': 'none'
            }
        
        # Select highest scoring track
        selected = sorted_tracks[0]
        
        logger.info(f"Rule-based selection: {selected['name']} (score: {selected['tag_score']})")
        
        return {
            'file': selected['path'],
            'structure': self._generate_simple_structure(timeline),
            'tempo_markers': [],
            'source': 'rule_based'
        }
    
    def _match_track(self, recommendation: str, available_tracks: List[Dict]) -> Optional[Dict]:
        """
        Match AI recommendation to available track.
        
        Args:
            recommendation: Track name or tags from AI
            available_tracks: Available tracks
            
        Returns:
            Matching track or None
        """
        recommendation_lower = recommendation.lower()
        
        # Try exact name match first
        for track in available_tracks:
            if recommendation_lower in track['name'].lower():
                return track
        
        # Try tag matching
        rec_words = recommendation_lower.split()
        best_match = None
        best_score = 0
        
        for track in available_tracks:
            score = sum(1 for word in rec_words if word in track['name'].lower())
            score += track['tag_score'] * 2  # Weight genre tags higher
            
            if score > best_score:
                best_score = score
                best_match = track
        
        return best_match if best_score > 0 else None
    
    def _generate_simple_structure(self, timeline: Dict) -> Dict:
        """
        Generate simple 4-section music structure.
        
        Args:
            timeline: Timeline dictionary
            
        Returns:
            Structure dictionary
        """
        duration = timeline.get('total_duration', 90)
        
        return {
            'intro': {
                'start': 0,
                'end': duration * 0.2,
                'description': 'Opening music'
            },
            'build': {
                'start': duration * 0.2,
                'end': duration * 0.6,
                'description': 'Building tension'
            },
            'climax': {
                'start': duration * 0.6,
                'end': duration * 0.9,
                'description': 'Climactic music'
            },
            'outro': {
                'start': duration * 0.9,
                'end': duration,
                'description': 'Closing'
            }
        }
    
    def _format_track_list(self, tracks: List[Dict]) -> str:
        """
        Format track list for AI prompt.
        
        Args:
            tracks: List of track dictionaries
            
        Returns:
            Formatted string
        """
        lines = []
        for i, track in enumerate(tracks, 1):
            tags_str = ', '.join(track['tags']) if track['tags'] else 'no tags'
            lines.append(f"{i}. {track['name']} (tags: {tags_str})")
        
        return '\n'.join(lines)
