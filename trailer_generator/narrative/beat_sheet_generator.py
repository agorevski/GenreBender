"""
Beat Sheet Generator
======================

Transforms a story graph into a genre-specific trailer beat sheet through two-stage LLM processing:
1. Genre Reinterpretation: Reframe the movie in the target genre
2. Beat Sheet Generation: Create 8-12 trailer beats for scene retrieval

Output: beats.json with embedding prompts for Layer 2.3 retrieval
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class BeatSheetGenerator:
    """
    Generates trailer beat sheets from story graphs using two-stage LLM approach.
    
    Stage 1: Genre Reinterpretation
        - Reframe the movie concept in the target genre
        - Output: genre_rewrite.json with logline, conflict, tone, emotional arc
    
    Stage 2: Beat Sheet Generation
        - Convert reinterpretation into 8-12 trailer beats
        - Output: beats.json with embedding prompts for retrieval
    """
    
    # LLM Call 1: Genre Reinterpretation Prompt Template
    REINTERPRETATION_SYSTEM_PROMPT = """You are a film re-interpretation engine specializing in genre transformation.
Your task is to rewrite movie concepts into different genres while preserving the core plot and characters.
You change only: tone, emotional framing, pacing, genre tropes, and implied meaning.
You must return valid JSON only, with no additional commentary."""
    
    REINTERPRETATION_USER_TEMPLATE = """Input:
- STORY_GRAPH:
{story_graph_json}

- TARGET GENRE: {target_genre}

- GENRE CONTEXT:
{genre_context}

Task:
Reinterpret this film in the style of the TARGET GENRE. Do not change main events or characters. Change only tone, emotional framing, pacing, genre tropes, and implied meaning.

Output JSON (no prose):

{{
  "new_genre": "{target_genre}",
  "logline": "1-2 sentence genre-specific pitch that reframes the story",
  "primary_conflict": "core tension reframed for new genre with specific stakes",
  "antagonistic_forces": ["list of 3-5 threats, obstacles, or opposing forces in genre terms"],
  "genre_motifs": ["list of 4-6 recurring symbols, visual patterns, or thematic elements for this genre"],
  "tone_profile": {{
    "pace": "slow/medium/fast or descriptive phrase",
    "visual_tone": "3-5 keywords describing color, lighting, atmosphere",
    "sound_profile": "3-5 keywords describing audio mood, music style, sound design"
  }},
  "emotional_arc_transformed": [
    {{"phase": "opening", "emotion": "specific emotion", "description": "brief how this phase feels"}},
    {{"phase": "build", "emotion": "specific emotion", "description": "brief how this phase feels"}},
    {{"phase": "climax", "emotion": "specific emotion", "description": "brief how this phase feels"}},
    {{"phase": "resolution", "emotion": "specific emotion", "description": "brief how this phase feels"}}
  ]
}}

Be specific and concrete. Use vivid genre language."""
    
    # LLM Call 2: Beat Sheet Generation Prompt Template
    BEAT_SHEET_SYSTEM_PROMPT = """You are a professional trailer editor with expertise in creating compelling beat sheets.
Your beat sheets are used for automated scene retrieval using semantic embeddings.
Each beat must have a rich, detailed embedding_prompt that combines mood, visuals, tension, and genre keywords.
You must return valid JSON only, with no additional commentary."""
    
    BEAT_SHEET_USER_TEMPLATE = """You are creating a 90-second {target_genre} trailer beat sheet.

Input (GENRE_REWRITE):
{genre_rewrite_json}

Task:
Produce a trailer beat sheet with 8-12 beats that tell a compelling story in {target_genre} style.
Each beat must be machine-usable for retrieval and clip selection.

Output JSON only:

{{
  "target_genre": "{target_genre}",
  "target_duration": 90,
  "beat_count": 10,
  "beats": [
    {{
      "id": "beat_01",
      "name": "Cold Open",
      "description": "2-3 sentence description of what happens in this beat",
      "target_emotion": "single emotion word",
      "visual_requirements": ["requirement 1", "requirement 2", "requirement 3", "requirement 4", "requirement 5"],
      "audio_cue": "description of sound/music that should accompany this",
      "voiceover": "optional voiceover line or null",
      "embedding_prompt": "CRITICAL: Dense 3-4 sentence paragraph combining mood keywords, visual atmosphere, tension level, genre indicators, character states, and scene qualities. This text is used for semantic search to find matching scenes. Be specific and vivid. Example: 'Dark atmospheric establishing shot with ominous tension. Shadowy figures in dimly lit urban environment. Sense of foreboding and unease. Mystery thriller mood with high suspense. Character isolation and vulnerability.'"
    }},
    {{
      "id": "beat_02",
      "name": "Introduce Protagonist",
      "description": "show main character in their world",
      "target_emotion": "another emotion",
      "visual_requirements": ["req 1", "req 2", "req 3", "req 4", "req 5"],
      "audio_cue": "sound description",
      "voiceover": "optional line or null",
      "embedding_prompt": "Dense paragraph with genre keywords, visual mood, character state, atmosphere, tension..."
    }}
  ]
}}

Requirements:
- Exactly 8-12 beats
- Each visual_requirements must have 3-5 items
- embedding_prompt is MANDATORY and must be detailed (50+ words)
- Follow standard trailer structure: hook → setup → conflict → escalation → climax tease
- Use {target_genre} conventions and tropes
- Make embedding_prompt rich with searchable keywords"""
    
    def __init__(
        self,
        azure_client,
        genre_profiles_path: Optional[Path] = None,
        temperature: float = 0.7,
        min_beats: int = 8,
        max_beats: int = 12
    ):
        """
        Initialize beat sheet generator.
        
        Args:
            azure_client: AzureOpenAIClient instance
            genre_profiles_path: Path to genre_profiles.yaml (None = use default)
            temperature: Sampling temperature for generation
            min_beats: Minimum number of beats (default: 8)
            max_beats: Maximum number of beats (default: 12)
        """
        self.azure_client = azure_client
        self.temperature = temperature
        self.min_beats = min_beats
        self.max_beats = max_beats
        
        # Load genre profiles
        if genre_profiles_path is None:
            genre_profiles_path = Path(__file__).parent.parent / "config" / "genre_profiles.yaml"
        
        with open(genre_profiles_path, 'r') as f:
            self.genre_profiles = yaml.safe_load(f)
        
        logger.info(f"Initialized BeatSheetGenerator (temp={temperature}, "
                   f"beats={min_beats}-{max_beats}, "
                   f"genres={list(self.genre_profiles.keys())})")
    
    def generate_beat_sheet(
        self,
        story_graph: Dict,
        target_genre: str,
        output_dir: Optional[Path] = None,
        genre_rewrite_filename: Optional[str] = None
    ) -> Dict:
        """
        Generate complete beat sheet from story graph.
        
        Args:
            story_graph: Story graph dictionary from Stage 11
            target_genre: Target genre for trailer
            output_dir: Optional directory to save intermediate outputs
            genre_rewrite_filename: Optional filename for genre_rewrite output 
                                   (default: genre_rewrite.json, use genre_rewrite_{genre}.json for multi-genre)
        
        Returns:
            Dictionary with 'beats' and 'embeddings' (placeholder for now)
        
        Raises:
            ValueError: If target genre not supported
            Exception: If LLM generation fails
        """
        # Validate genre
        if target_genre not in self.genre_profiles:
            raise ValueError(
                f"Unsupported genre: {target_genre}. "
                f"Available: {list(self.genre_profiles.keys())}"
            )
        
        logger.info(f"Generating beat sheet for '{story_graph.get('title', 'Unknown')}' "
                   f"as {target_genre} trailer")
        
        # Stage 1: Genre Reinterpretation
        logger.info("Stage 1: Genre Reinterpretation...")
        genre_rewrite = self._stage1_genre_reinterpretation(story_graph, target_genre)
        
        # Save intermediate output (with genre-specific filename if provided)
        if output_dir:
            rewrite_filename = genre_rewrite_filename or "genre_rewrite.json"
            rewrite_path = output_dir / rewrite_filename
            with open(rewrite_path, 'w') as f:
                json.dump(genre_rewrite, f, indent=2)
            logger.info(f"Saved genre rewrite to: {rewrite_path}")
        
        # Stage 2: Beat Sheet Generation
        logger.info("Stage 2: Beat Sheet Generation...")
        beat_sheet = self._stage2_beat_sheet_generation(genre_rewrite, target_genre)
        
        # Validate beat sheet
        self._validate_beat_sheet(beat_sheet)
        
        # Generate embeddings (placeholder for Layer 2.3)
        embeddings = self._generate_embeddings(beat_sheet)
        
        # Combine results
        result = {
            "genre_rewrite": genre_rewrite,
            "beat_sheet": beat_sheet,
            "embeddings": embeddings
        }
        
        logger.info(f"Beat sheet generation complete: {len(beat_sheet['beats'])} beats")
        
        return result
    
    def _stage1_genre_reinterpretation(
        self,
        story_graph: Dict,
        target_genre: str
    ) -> Dict:
        """
        Stage 1: Reinterpret the story in the target genre.
        
        Args:
            story_graph: Story graph dictionary
            target_genre: Target genre
        
        Returns:
            Genre rewrite dictionary
        """
        # Build genre context from profile
        genre_profile = self.genre_profiles[target_genre]
        genre_context = self._build_genre_context(genre_profile, target_genre)
        
        # Prepare story graph (compact version to fit in prompt)
        story_graph_compact = self._compact_story_graph(story_graph)
        
        # Build messages
        user_prompt = self.REINTERPRETATION_USER_TEMPLATE.format(
            story_graph_json=json.dumps(story_graph_compact, indent=2),
            target_genre=target_genre,
            genre_context=genre_context
        )
        
        messages = [
            {"role": "system", "content": self.REINTERPRETATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Log prompt size
        prompt_size = len(self.REINTERPRETATION_SYSTEM_PROMPT) + len(user_prompt)
        logger.info(f"Stage 1 prompt size: {prompt_size} chars")
        
        # Call LLM
        try:
            response = self.azure_client.generate_structured_output(
                messages=messages,
                temperature=self.temperature
            )
            
            if not response or not response.strip():
                raise Exception("Empty response from LLM for genre reinterpretation")
            
            genre_rewrite = json.loads(response)
            logger.info(f"Stage 1 complete: {genre_rewrite.get('logline', '')[:100]}...")
            
            return genre_rewrite
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in genre reinterpretation: {e}")
            logger.error(f"Response: {response[:500] if response else 'EMPTY'}")
            raise Exception(f"Failed to parse genre reinterpretation response: {e}")
    
    def _stage2_beat_sheet_generation(
        self,
        genre_rewrite: Dict,
        target_genre: str
    ) -> Dict:
        """
        Stage 2: Generate beat sheet from genre reinterpretation.
        
        Args:
            genre_rewrite: Genre rewrite from Stage 1
            target_genre: Target genre
        
        Returns:
            Beat sheet dictionary
        """
        # Build messages
        user_prompt = self.BEAT_SHEET_USER_TEMPLATE.format(
            target_genre=target_genre,
            genre_rewrite_json=json.dumps(genre_rewrite, indent=2)
        )
        
        messages = [
            {"role": "system", "content": self.BEAT_SHEET_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Log prompt size
        prompt_size = len(self.BEAT_SHEET_SYSTEM_PROMPT) + len(user_prompt)
        logger.info(f"Stage 2 prompt size: {prompt_size} chars")
        
        # Call LLM
        try:
            response = self.azure_client.generate_structured_output(
                messages=messages,
                temperature=self.temperature
            )
            
            if not response or not response.strip():
                raise Exception("Empty response from LLM for beat sheet generation")
            
            beat_sheet = json.loads(response)
            logger.info(f"Stage 2 complete: {len(beat_sheet.get('beats', []))} beats generated")
            
            return beat_sheet
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in beat sheet generation: {e}")
            logger.error(f"Response: {response[:500] if response else 'EMPTY'}")
            raise Exception(f"Failed to parse beat sheet response: {e}")
    
    def _build_genre_context(self, genre_profile: Dict, genre_name: str) -> str:
        """
        Build genre context string from profile.
        
        Args:
            genre_profile: Genre profile dictionary
            genre_name: Genre name
        
        Returns:
            Formatted genre context string
        """
        context_parts = [f"Genre: {genre_name}"]
        
        # Scoring weights
        if 'scoring_weights' in genre_profile:
            weights = genre_profile['scoring_weights']
            context_parts.append(f"Key Attributes: {', '.join(weights.keys())}")
        
        # Pacing
        if 'pacing' in genre_profile:
            context_parts.append(f"Pacing Style: {genre_profile['pacing']}")
        
        # Music tags
        if 'music_tags' in genre_profile:
            context_parts.append(f"Audio Mood: {', '.join(genre_profile['music_tags'])}")
        
        # Color grade description
        if 'color_grade' in genre_profile and 'description' in genre_profile['color_grade']:
            context_parts.append(f"Visual Style: {genre_profile['color_grade']['description']}")
        
        return "\n".join(context_parts)
    
    def _compact_story_graph(self, story_graph: Dict) -> Dict:
        """
        Create compact version of story graph for LLM prompt.
        
        Args:
            story_graph: Full story graph
        
        Returns:
            Compact story graph with essential fields
        """
        # Keep essential fields only
        compact = {
            "title": story_graph.get("title", ""),
            "logline": story_graph.get("logline", ""),
            "characters": [],
            "major_themes": story_graph.get("major_themes", []),
            "plot_structure": story_graph.get("plot_structure", {}),
            "emotional_arc": story_graph.get("emotional_arc", {})
        }
        
        # Compact characters (top 5 most important)
        characters = story_graph.get("characters", [])[:5]
        for char in characters:
            compact["characters"].append({
                "name": char.get("name", ""),
                "description": char.get("description", ""),
                "motivations": char.get("motivations", [])[:3]
            })
        
        # Compact scene timeline (sample every few scenes)
        scenes = story_graph.get("scene_timeline", [])
        if len(scenes) > 10:
            # Sample: beginning, middle, end
            sampled_indices = [0, len(scenes)//4, len(scenes)//2, 3*len(scenes)//4, len(scenes)-1]
            sampled_scenes = [scenes[i] for i in sampled_indices if i < len(scenes)]
        else:
            sampled_scenes = scenes
        
        compact["scene_summary"] = [
            {
                "scene_id": s.get("scene_id"),
                "summary": s.get("summary", "")[:200],  # Truncate long summaries
                "dominant_emotion": s.get("dominant_emotion", "")
            }
            for s in sampled_scenes
        ]
        
        return compact
    
    def _validate_beat_sheet(self, beat_sheet: Dict) -> None:
        """
        Validate beat sheet structure and content.
        
        Args:
            beat_sheet: Beat sheet dictionary
        
        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        if "beats" not in beat_sheet:
            raise ValueError("Beat sheet missing 'beats' field")
        
        beats = beat_sheet["beats"]
        
        # Check beat count
        if len(beats) < self.min_beats or len(beats) > self.max_beats:
            raise ValueError(
                f"Beat count {len(beats)} outside valid range "
                f"({self.min_beats}-{self.max_beats})"
            )
        
        # Validate each beat
        required_beat_fields = [
            "id", "name", "description", "target_emotion",
            "visual_requirements", "audio_cue", "embedding_prompt"
        ]
        
        for i, beat in enumerate(beats, 1):
            # Check required fields
            missing = [f for f in required_beat_fields if f not in beat]
            if missing:
                raise ValueError(f"Beat {i} missing fields: {missing}")
            
            # Check visual_requirements count
            visuals = beat.get("visual_requirements", [])
            if len(visuals) < 3 or len(visuals) > 5:
                raise ValueError(
                    f"Beat {i} has {len(visuals)} visual requirements, "
                    f"expected 3-5"
                )
            
            # Check embedding_prompt length
            embedding_prompt = beat.get("embedding_prompt", "")
            if len(embedding_prompt.split()) < 30:
                logger.warning(
                    f"Beat {i} embedding_prompt is short ({len(embedding_prompt.split())} words), "
                    f"should be 50+ words for effective retrieval"
                )
        
        logger.info(f"Beat sheet validation passed: {len(beats)} beats")
    
    def _generate_embeddings(self, beat_sheet: Dict) -> List:
        """
        Generate embeddings from beat embedding_prompts.
        
        TODO: Implement actual embedding generation for Layer 2.3
        Currently returns empty placeholder arrays.
        
        Args:
            beat_sheet: Beat sheet dictionary
        
        Returns:
            List of embedding vectors (placeholder: empty arrays)
        """
        beats = beat_sheet.get("beats", [])
        
        # Placeholder: return empty arrays
        # Layer 2.3 will implement actual embedding generation
        embeddings = [[] for _ in beats]
        
        logger.info(f"Generated {len(embeddings)} embedding placeholders "
                   f"(Layer 2.3 TODO: implement actual embeddings)")
        
        return embeddings
    
    def save_beat_sheet(
        self,
        beat_sheet: Dict,
        output_path: Path,
        include_metadata: bool = True
    ) -> None:
        """
        Save beat sheet to JSON file.
        
        Args:
            beat_sheet: Beat sheet dictionary
            output_path: Path to save beats.json
            include_metadata: Whether to include generation metadata
        """
        output_data = beat_sheet.copy()
        
        if include_metadata:
            output_data["_metadata"] = {
                "generator": "beat_sheet_generator_v1",
                "temperature": self.temperature,
                "beat_count_range": f"{self.min_beats}-{self.max_beats}",
                "actual_beat_count": len(beat_sheet.get("beats", []))
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved beat sheet to: {output_path}")
