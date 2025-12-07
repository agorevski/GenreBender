"""
Genre-specific prompt templates for trailer structure generation.
"""

from typing import Dict, List

class StructurePrompts:
    """
    Repository of genre-specific prompts for trailer generation.
    """
    
    THRILLER_PROMPT = """You are a professional trailer editor specializing in thriller films. 
Given a selection of shots with descriptions and scores, create a compelling 90-120 second trailer structure.

THRILLER TRAILER ARC:
1. **Hook** (5-10s): Establish normalcy with subtle unease
2. **Inciting Incident** (10-15s): Something is wrong
3. **Rising Tension** (30-40s): Escalating mysteries and threats
4. **Climactic Montage** (20-30s): Fast cuts, high suspense
5. **Sting/Tagline** (5-10s): Final shocking moment + title card

SHOTS AVAILABLE:
{shot_descriptions}

Create a timeline that:
- Builds suspense gradually
- Uses shorter shot durations as tension increases
- Places highest-scoring shots in climactic section
- Includes strategic black frames for dramatic pauses
- Suggests text overlays at key moments

Return ONLY valid JSON in this exact format:
{{
  "timeline": [
    {{"shot_id": 12, "duration": 2.5, "text_overlay": "Something is wrong..."}},
    {{"shot_id": 48, "duration": 1.8}},
    {{"shot_id": 3, "duration": 0.8, "text_overlay": "WHO CAN YOU TRUST?"}}
  ],
  "music_cues": [
    {{"timestamp": 0, "type": "ambient_tension", "description": "Low, rumbling tension"}},
    {{"timestamp": 45, "type": "rising_action", "description": "Building strings"}},
    {{"timestamp": 75, "type": "climax", "description": "Full orchestral hit"}}
  ],
  "pacing_notes": "Build slowly, accelerate after 60s mark"
}}"""

    ACTION_PROMPT = """You are a professional trailer editor specializing in action films.
Given a selection of shots with descriptions and scores, create an explosive 90-120 second trailer structure.

ACTION TRAILER ARC:
1. **Hook** (5-8s): Immediate action beat
2. **Setup** (15-20s): Establish stakes and protagonist
3. **Rising Action** (35-45s): Escalating set pieces
4. **Climactic Montage** (25-35s): Rapid-fire action sequences
5. **Tagline** (5-10s): Final impact + title reveal

SHOTS AVAILABLE:
{shot_descriptions}

Create a timeline that:
- Maintains high energy throughout
- Uses quick cuts (0.5-2s per shot in climax)
- Emphasizes motion and intensity
- Places highest-scoring action moments in climax
- Suggests impactful text overlays

Return ONLY valid JSON in this exact format:
{{
  "timeline": [
    {{"shot_id": 5, "duration": 1.5, "text_overlay": "ONE MAN"}},
    {{"shot_id": 23, "duration": 0.8}},
    {{"shot_id": 67, "duration": 1.2, "text_overlay": "ONE MISSION"}}
  ],
  "music_cues": [
    {{"timestamp": 0, "type": "epic_percussion", "description": "Heavy drums"}},
    {{"timestamp": 30, "type": "rising_energy", "description": "Building intensity"}},
    {{"timestamp": 70, "type": "climax", "description": "Full epic orchestration"}}
  ],
  "pacing_notes": "Fast-paced throughout, accelerate after 60s"
}}"""

    DRAMA_PROMPT = """You are a professional trailer editor specializing in dramatic films.
Given a selection of shots with descriptions and scores, create an emotional 90-120 second trailer structure.

DRAMA TRAILER ARC:
1. **Introduction** (10-15s): Establish world and characters
2. **Conflict** (20-30s): Present the central challenge
3. **Emotional Journey** (30-40s): Show character struggles
4. **Climactic Moments** (20-25s): Peak emotional beats
5. **Resolution Tease** (10-15s): Hopeful or ambiguous ending

SHOTS AVAILABLE:
{shot_descriptions}

Create a timeline that:
- Emphasizes emotional connection
- Uses longer, contemplative shots
- Builds to emotional peak
- Includes breathing room between key moments
- Suggests evocative text overlays

Return ONLY valid JSON in this exact format:
{{
  "timeline": [
    {{"shot_id": 8, "duration": 3.5, "text_overlay": "Some journeys..."}},
    {{"shot_id": 34, "duration": 2.8}},
    {{"shot_id": 52, "duration": 2.2, "text_overlay": "...change us forever"}}
  ],
  "music_cues": [
    {{"timestamp": 0, "type": "intimate_piano", "description": "Gentle, melancholic"}},
    {{"timestamp": 40, "type": "building_strings", "description": "Emotional swell"}},
    {{"timestamp": 80, "type": "resolution", "description": "Hopeful melody"}}
  ],
  "pacing_notes": "Measured pace, allow emotional moments to breathe"
}}"""

    HORROR_PROMPT = """You are a professional trailer editor specializing in horror films.
Given a selection of shots with descriptions and scores, create a terrifying 90-120 second trailer structure.

HORROR TRAILER ARC:
1. **False Security** (10-15s): Seemingly normal scenario
2. **First Signs** (15-20s): Subtle wrongness emerges
3. **Escalation** (30-40s): Terror intensifies
4. **Chaos Montage** (20-30s): Rapid frightening images
5. **Final Scare** (5-10s): Ultimate shock + title

SHOTS AVAILABLE:
{shot_descriptions}

Create a timeline that:
- Builds dread gradually
- Uses silence and sudden cuts
- Places jump scares strategically
- Includes black frames for impact
- Suggests minimal, ominous text

Return ONLY valid JSON in this exact format:
{{
  "timeline": [
    {{"shot_id": 15, "duration": 3.0}},
    {{"shot_id": 42, "duration": 1.5, "text_overlay": "It's watching..."}},
    {{"shot_id": 71, "duration": 0.5}}
  ],
  "music_cues": [
    {{"timestamp": 0, "type": "ambient_dread", "description": "Unsettling silence"}},
    {{"timestamp": 50, "type": "building_terror", "description": "Dissonant strings"}},
    {{"timestamp": 90, "type": "shock", "description": "Sudden loud hit"}}
  ],
  "pacing_notes": "Slow build with sudden accelerations, strategic silence"
}}"""

    @classmethod
    def get_prompt(cls, genre: str) -> str:
        """
        Get prompt template for specific genre.
        
        Args:
            genre: Genre name (thriller, action, drama, horror, etc.)
            
        Returns:
            Prompt template string
        """
        genre_lower = genre.lower()
        
        prompts = {
            'thriller': cls.THRILLER_PROMPT,
            'action': cls.ACTION_PROMPT,
            'drama': cls.DRAMA_PROMPT,
            'horror': cls.HORROR_PROMPT
        }
        
        return prompts.get(genre_lower, cls.THRILLER_PROMPT)
    
    @classmethod
    def format_shot_descriptions(cls, shots: List[Dict], max_shots: int = 30) -> str:
        """
        Format shot descriptions for prompt.
        
        Args:
            shots: List of shot dictionaries with analysis
            max_shots: Maximum number of shots to include in prompt
            
        Returns:
            Formatted shot descriptions string
        """
        # Limit number of shots to avoid token limits
        selected_shots = shots[:max_shots]
        
        descriptions = []
        for shot in selected_shots:
            shot_id = shot.get('id')
            duration = shot.get('duration', 0)
            score = shot.get('score', 0)
            analysis = shot.get('analysis', {})
            caption = analysis.get('caption', 'No description')
            
            desc = f"Shot {shot_id}: {caption} (score: {score:.2f}, duration: {duration:.1f}s)"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    @classmethod
    def create_chunked_prompt(cls, genre: str, shots: List[Dict], 
                             section: str, context: str = "") -> str:
        """
        Create prompt for a specific section of the trailer (for chunked generation).
        
        Args:
            genre: Genre name
            shots: Shots for this section
            section: Section name (intro, middle, climax)
            context: Context from previous sections
            
        Returns:
            Formatted prompt
        """
        base_prompt = cls.get_prompt(genre)
        shot_desc = cls.format_shot_descriptions(shots)
        
        section_instructions = {
            'intro': "Focus on establishing tone and hook (15-25 seconds)",
            'middle': "Build tension and develop narrative (30-45 seconds)",
            'climax': "Create climactic montage and resolution (30-40 seconds)"
        }
        
        instruction = section_instructions.get(section, "")
        
        chunked_prompt = f"""{base_prompt}

SECTION: {section.upper()}
{instruction}

{f"CONTEXT FROM PREVIOUS SECTION: {context}" if context else ""}

SHOTS FOR THIS SECTION:
{shot_desc}

Generate timeline for THIS SECTION ONLY, maintaining consistency with any previous context."""
        
        return chunked_prompt
