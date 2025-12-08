"""
AI-powered title card generation for trailers.
Uses Azure OpenAI to generate compelling title text based on timeline and genre.
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TitleGenerator:
    """
    Generates title cards using AI or fallback templates.
    """
    
    # Fallback templates by genre
    GENRE_TEMPLATES = {
        'thriller': [
            {'text': 'SOME SECRETS', 'position': 'intro'},
            {'text': 'SHOULD STAY BURIED', 'position': 'middle'}
        ],
        'action': [
            {'text': 'ONE MAN', 'position': 'intro'},
            {'text': 'AGAINST ALL ODDS', 'position': 'climax'}
        ],
        'horror': [
            {'text': 'FEAR', 'position': 'intro'},
            {'text': 'HAS A NEW NAME', 'position': 'middle'}
        ],
        'drama': [
            {'text': 'SOME CHOICES', 'position': 'intro'},
            {'text': 'DEFINE US', 'position': 'middle'}
        ],
        'scifi': [
            {'text': 'THE FUTURE', 'position': 'intro'},
            {'text': 'BEGINS NOW', 'position': 'climax'}
        ],
        'comedy': [
            {'text': 'THIS SUMMER', 'position': 'intro'},
            {'text': 'EXPECT THE UNEXPECTED', 'position': 'middle'}
        ],
        'romance': [
            {'text': 'WHEN TWO HEARTS', 'position': 'intro'},
            {'text': 'BECOME ONE', 'position': 'middle'}
        ]
    }
    
    def __init__(self, azure_client, genre: str, enable_ai: bool = True):
        """
        Initialize title generator.
        
        Args:
            azure_client: Azure OpenAI client instance (can be None if enable_ai=False)
            genre: Target genre for trailer
            enable_ai: Whether to use AI for title generation
        """
        self.azure_client = azure_client
        self.genre = genre.lower()
        self.enable_ai = enable_ai
    
    def generate_titles(self, timeline: Dict) -> List[Dict]:
        """
        Generate title cards for trailer.
        
        Args:
            timeline: Timeline dictionary with shot sequence
            
        Returns:
            List of title card dictionaries with text, timestamp, duration
        """
        if self.enable_ai and self.azure_client:
            try:
                logger.info("Generating titles with AI...")
                return self._generate_ai_titles(timeline)
            except Exception as e:
                logger.warning(f"AI title generation failed: {e}, using fallback templates")
                return self._generate_fallback_titles(timeline)
        else:
            logger.info("Using fallback title templates...")
            return self._generate_fallback_titles(timeline)
    
    def _generate_ai_titles(self, timeline: Dict) -> List[Dict]:
        """
        Generate titles using Azure OpenAI.
        
        Args:
            timeline: Timeline dictionary
            
        Returns:
            List of title dictionaries
        """
        # Build prompt
        prompt = f"""Generate 2-3 compelling title cards for a {self.genre} trailer.

Timeline summary:
- Duration: {timeline.get('total_duration', 90)}s
- Number of shots: {len(timeline.get('timeline', []))}
- Pacing notes: {timeline.get('pacing_notes', 'Standard pacing')}
- Music cues: {len(timeline.get('music_cues', []))} cues

Requirements:
1. Keep titles SHORT (3-7 words max)
2. Match {self.genre} genre conventions and tone
3. Place titles at key moments (intro, middle, climax)
4. Create suspense/intrigue without revealing plot
5. Use impactful, memorable language

Return ONLY valid JSON in this exact format (no markdown, no explanations):
{{
    "titles": [
        {{"text": "YOUR TITLE HERE", "timestamp": 0, "duration": 3, "position": "intro"}},
        {{"text": "SECOND TITLE", "timestamp": 45, "duration": 3, "position": "middle"}}
    ]
}}"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional trailer editor. Return only valid JSON with title cards."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Generate titles
        response = self.azure_client.generate_structured_output(messages=messages)
        
        # Parse response
        try:
            titles_data = json.loads(response)
            titles = titles_data.get('titles', [])
            
            # Validate and adjust timestamps
            total_duration = timeline.get('total_duration', 90)
            for title in titles:
                # Ensure timestamp is within bounds
                if title['timestamp'] > total_duration - title['duration']:
                    title['timestamp'] = max(0, total_duration - title['duration'] - 2)
                
                # Convert text to uppercase for impact
                title['text'] = title['text'].upper()
            
            logger.info(f"Generated {len(titles)} AI titles")
            logger.info(f"Titles: {titles}")
            return titles
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise
    
    def _generate_fallback_titles(self, timeline: Dict) -> List[Dict]:
        """
        Generate titles using genre templates.
        
        Args:
            timeline: Timeline dictionary
            
        Returns:
            List of title dictionaries
        """
        templates = self.GENRE_TEMPLATES.get(self.genre, self.GENRE_TEMPLATES['thriller'])
        total_duration = timeline.get('total_duration', 90)
        
        titles = []
        for template in templates:
            # Calculate timestamp based on position
            if template['position'] == 'intro':
                timestamp = 0
            elif template['position'] == 'middle':
                timestamp = total_duration * 0.5
            elif template['position'] == 'climax':
                timestamp = total_duration * 0.75
            else:
                timestamp = 0
            
            titles.append({
                'text': template['text'],
                'timestamp': timestamp,
                'duration': 3,
                'position': template['position']
            })
        
        logger.info(f"Generated {len(titles)} fallback titles")
        return titles
