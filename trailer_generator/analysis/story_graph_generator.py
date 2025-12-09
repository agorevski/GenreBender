"""
Story Graph Generator
=====================

Generates comprehensive semantic story graphs from movie synopsis and subtitles.
Uses hierarchical 3-stage approach with chunking for handling long movies.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from .subtitle_chunker import SubtitleChunker

logger = logging.getLogger(__name__)

class StoryGraphGenerator:
    """
    Generates story graphs using LLM analysis of synopsis and subtitles.
    Uses 3-stage hierarchical approach:
    1. Chunk-level analysis (15min chunks)
    2. Narrative synthesis (combine chunks)
    3. Emotional arc generation
    """
    
    def __init__(self, azure_client, chunk_duration_minutes: int = 15, 
                 overlap_seconds: int = 30, max_parallel_chunks: int = 5,
                 cache_dir: Optional[Path] = None,
                 synthesis_max_tokens: Optional[int] = None,
                 temperature: float = 0.3):
        """
        Initialize story graph generator.
        
        Args:
            azure_client: AzureOpenAIClient instance
            chunk_duration_minutes: Duration of subtitle chunks
            overlap_seconds: Overlap between chunks
            max_parallel_chunks: Maximum number of chunks to process in parallel
            cache_dir: Directory for caching chunk analyses
            synthesis_max_tokens: Max tokens for synthesis stage (None = use client default)
            temperature: Sampling temperature for story graph generation (default: 0.3)
        """
        self.azure_client = azure_client
        self.chunker = SubtitleChunker(chunk_duration_minutes, overlap_seconds)
        self.max_parallel_chunks = max_parallel_chunks
        self.cache_dir = cache_dir
        self.synthesis_max_tokens = synthesis_max_tokens
        self.temperature = temperature
        logger.info(f"Initialized StoryGraphGenerator with hierarchical processing "
                   f"(max {max_parallel_chunks} parallel chunks, temperature={temperature})")
    
    def generate_story_graph(
        self,
        movie_name: str,
        synopsis: str,
        subtitles_text: str,
        force_regenerate: bool = False
    ) -> Dict:
        """
        Generate story graph from movie data using 3-stage approach.
        
        Args:
            movie_name: Movie title
            synopsis: Movie synopsis/plot summary
            subtitles_text: Full subtitle transcript with timestamps
            force_regenerate: If True, ignore cache and regenerate all
            
        Returns:
            Story graph dictionary
            
        Raises:
            Exception: If generation fails
        """
        logger.info(f"Generating story graph for: {movie_name}")
        
        # Clear cache if force regenerate
        if force_regenerate and self.cache_dir:
            self._clear_cache()
        
        # Parse subtitle entries
        entries = self._parse_subtitle_entries(subtitles_text)
        logger.info(f"Parsed {len(entries)} subtitle entries")
        
        # Stage 1: Chunk and analyze (with parallel processing)
        logger.info("Stage 1: Analyzing chunks in parallel...")
        chunk_analyses = self._stage1_analyze_chunks(entries)
        
        # Stage 2: Synthesize narrative
        logger.info("Stage 2: Synthesizing narrative from chunks...")
        story_graph = self._stage2_synthesize_narrative(
            movie_name, synopsis, chunk_analyses
        )
        
        # Stage 3: Generate emotional arc
        logger.info("Stage 3: Generating emotional arc...")
        story_graph['emotional_arc'] = self._stage3_generate_emotional_arc(
            story_graph['scene_timeline']
        )
        
        # Add metadata
        story_graph['_metadata'] = {
            'movie_name': movie_name,
            'generated_by': 'story_graph_generator_v2_hierarchical_parallel',
            'synopsis_length': len(synopsis),
            'subtitles_entries': len(entries),
            'chunks_processed': len(chunk_analyses),
            'hierarchical_stages': 3,
            'parallel_processing': True,
            'max_parallel_chunks': self.max_parallel_chunks
        }
        
        logger.info(f"Story graph complete: {len(story_graph.get('characters', []))} characters, "
                   f"{len(story_graph.get('scene_timeline', []))} scenes")
        
        return story_graph
    
    def _parse_subtitle_entries(self, subtitles_text: str) -> List[Dict]:
        """
        Parse subtitle text into structured entries.
        
        Args:
            subtitles_text: Raw subtitle text with timestamps
        
        Returns:
            List of dicts with 'start_time', 'end_time', 'text'
        """
        entries = []
        
        # Split by timestamp pattern [HH:MM:SS]
        pattern = r'\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+?)(?=\[|\Z)'
        matches = re.finditer(pattern, subtitles_text, re.DOTALL)
        
        prev_time = 0
        for match in matches:
            hours, minutes, seconds, text = match.groups()
            start_time = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            
            # Clean text
            text = text.strip()
            if not text:
                continue
            
            # Estimate end time (use next entry's start or add 3 seconds)
            end_time = start_time + 3
            
            entry = {
                'start_time': float(start_time),
                'end_time': float(end_time),
                'text': text
            }
            entries.append(entry)
            prev_time = start_time
        
        # Update end times based on actual next start times
        for i in range(len(entries) - 1):
            entries[i]['end_time'] = entries[i + 1]['start_time']
        
        return entries
    
    def _process_chunk_with_cache(self, chunk: Dict, chunk_index: int, total_chunks: int) -> Optional[Dict]:
        """
        Process a single chunk with cache checking (thread-safe).
        
        Args:
            chunk: Chunk dict with transcript
            chunk_index: Index of chunk (1-based for logging)
            total_chunks: Total number of chunks
        
        Returns:
            Analysis dict or None if cached
        """
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"chunk_{chunk['chunk_id']:03d}.json"
            if cache_file.exists():
                logger.info(f"  [{chunk_index}/{total_chunks}] Chunk {chunk['chunk_id']}: Using cached analysis")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Analyze chunk
        logger.info(f"  [{chunk_index}/{total_chunks}] Chunk {chunk['chunk_id']}: Analyzing "
                   f"({chunk['entry_count']} entries, "
                   f"{self.chunker._format_time(chunk['start_time'])} - "
                   f"{self.chunker._format_time(chunk['end_time'])})")
        
        try:
            analysis = self._analyze_single_chunk(chunk)
            
            # Cache result
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self.cache_dir / f"chunk_{chunk['chunk_id']:03d}.json"
                with open(cache_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
            
            return analysis
        except Exception as e:
            logger.warning(f"  [{chunk_index}/{total_chunks}] Chunk {chunk['chunk_id']}: "
                         f"Skipping due to error: {e}")
            return None
    
    def _stage1_analyze_chunks(self, entries: List[Dict]) -> List[Dict]:
        """
        Stage 1: Analyze subtitle chunks in parallel.
        
        Args:
            entries: Parsed subtitle entries
        
        Returns:
            List of chunk analysis results
        """
        # Create chunks
        chunks = self.chunker.chunk_subtitle_entries(entries)
        summary = self.chunker.get_chunk_summary(chunks)
        
        logger.info(f"Processing {summary['total_chunks']} chunks in parallel "
                   f"({summary['chunk_duration_minutes']}min each, "
                   f"{summary['overlap_seconds']}s overlap, "
                   f"max {self.max_parallel_chunks} concurrent)")
        
        results = []
        skipped_chunks = []
        
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._process_chunk_with_cache, 
                    chunk, 
                    i, 
                    len(chunks)
                ): chunk
                for i, chunk in enumerate(chunks, 1)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        skipped_chunks.append(chunk['chunk_id'])
                except Exception as e:
                    logger.error(f"Chunk {chunk['chunk_id']} failed with exception: {e}")
                    skipped_chunks.append(chunk['chunk_id'])
        
        # Sort results by chunk_id to maintain order
        results.sort(key=lambda x: x['chunk_id'])
        
        if skipped_chunks:
            logger.warning(f"Skipped {len(skipped_chunks)} chunks due to errors: {sorted(skipped_chunks)}")
        
        logger.info(f"Stage 1 complete: {len(results)} chunks analyzed ({len(skipped_chunks)} skipped)")
        return results
    
    def _analyze_single_chunk(self, chunk: Dict) -> Dict:
        """
        Analyze a single subtitle chunk with GPT-4.
        
        Args:
            chunk: Chunk dict with transcript
        
        Returns:
            Analysis dict with characters, events, emotions, etc.
        """
        system_prompt = """You are a film analysis expert. Analyze this movie segment and extract structured information."""
        
        user_prompt = f"""Analyze this {int((chunk['end_time'] - chunk['start_time']) / 60)}-minute movie segment:

TIME RANGE: {self.chunker._format_time(chunk['start_time'])} to {self.chunker._format_time(chunk['end_time'])}

SUBTITLES:
{chunk['transcript']}

Extract the following in JSON format:

{{
  "chunk_id": {chunk['chunk_id']},
  "time_range": "{self.chunker._format_time(chunk['start_time'])} - {self.chunker._format_time(chunk['end_time'])}",
  "characters_present": ["character names mentioned or speaking"],
  "key_events": ["atomic event descriptions"],
  "dominant_emotions": ["primary emotions conveyed"],
  "visual_inferences": ["setting clues", "time of day", "indoor/outdoor", "atmosphere"],
  "genre_indicators": ["keywords suggesting genre"],
  "notable_dialogue": ["memorable or important lines"],
  "summary": "brief 2-3 sentence summary of this segment"
}}

Be concise but thorough."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.azure_client.generate_structured_output(
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.synthesis_max_tokens
            )
            
            if not response or not response.strip():
                logger.error(f"Empty response for chunk {chunk['chunk_id']}")
                raise Exception(f"Empty response from API for chunk {chunk['chunk_id']}")
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for chunk {chunk['chunk_id']}: {e}")
            logger.error(f"Response content: {response[:500] if response else 'EMPTY'}")
            raise Exception(f"Failed to parse JSON response for chunk {chunk['chunk_id']}: {e}")
    
    def _stage2_synthesize_narrative(
        self,
        movie_name: str,
        synopsis: str,
        chunk_analyses: List[Dict]
    ) -> Dict:
        """
        Stage 2: Synthesize complete story graph from chunk analyses.
        
        Args:
            movie_name: Movie title
            synopsis: Movie synopsis
            chunk_analyses: Results from stage 1
        
        Returns:
            Complete story graph
        """
        # Build synthesis prompt (exclude notable_dialogue to avoid content filtering)
        chunk_summaries = "\n\n".join([
            f"SEGMENT {a['chunk_id']} ({a['time_range']}):\n"
            f"Summary: {a.get('summary', 'No summary available')}\n"
            f"Characters: {', '.join(a.get('characters_present', [])[:10]) if a.get('characters_present') else 'None identified'}\n"
            f"Key Events: {'; '.join(a.get('key_events', [])[:5]) if a.get('key_events') else 'None identified'}"
            for a in chunk_analyses
        ])
        
        logger.info(f"Synthesizing narrative from {len(chunk_analyses)} chunk analyses")
        logger.info(f"Chunk summaries size: {len(chunk_summaries)} chars")
        logger.info(f"Synopsis size: {len(synopsis)} chars")
        
        system_prompt = """You are a film-understanding engine. Synthesize a complete Story Graph from the movie synopsis and segment analyses."""
        
        user_prompt = f"""Movie: {movie_name}

SYNOPSIS:
{synopsis}

SEGMENT ANALYSES:
{chunk_summaries}

Create a comprehensive Story Graph in JSON format:

{{
  "title": "{movie_name}",
  "logline": "1-2 sentence summary of main conflict",
  "characters": [
    {{
      "name": "string",
      "description": "short description",
      "motivations": ["list"],
      "relationships": {{"other_char": "relationship"}}
    }}
  ],
  "major_themes": ["abstract themes"],
  "plot_structure": {{
    "setup": "1-3 sentences",
    "inciting_incident": "1-2 sentences",
    "rising_action": "1-3 sentences",
    "climax": "1-2 sentences",
    "resolution": "1-2 sentences"
  }},
  "scene_timeline": [
    {{
      "scene_id": 1,
      "start_time": "from segment time_range",
      "end_time": "from segment time_range",
      "summary": "what happens",
      "key_events": ["atomic events"],
      "characters_present": ["names"],
      "dominant_emotion": "one word",
      "genre_indicators": ["keywords"],
      "visual_inferences": ["setting clues"]
    }}
  ]
}}

Synthesize all segments into coherent scenes. Merge related segments into scenes."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Log prompt details for debugging
        total_prompt_size = len(system_prompt) + len(user_prompt)
        logger.info(f"Synthesis prompt total size: {total_prompt_size} chars")
        logger.debug(f"System prompt: {system_prompt[:200]}...")
        logger.debug(f"User prompt preview: {user_prompt[:500]}...")
        
        # Save debug information
        if self.cache_dir:
            debug_file = self.cache_dir.parent / "synthesis_prompt_debug.txt"
            try:
                with open(debug_file, 'w') as f:
                    f.write("=== SYSTEM PROMPT ===\n")
                    f.write(system_prompt)
                    f.write("\n\n=== USER PROMPT ===\n")
                    f.write(user_prompt)
                logger.info(f"Saved synthesis prompt to: {debug_file}")
            except Exception as e:
                logger.warning(f"Could not save debug file: {e}")
        
        try:
            # Use configured synthesis_max_tokens or None (will use client default)
            response = self.azure_client.generate_structured_output(
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.synthesis_max_tokens
            )
            
            if not response or not response.strip():
                logger.error("Empty response from API for narrative synthesis")
                logger.error(f"Total prompt size was: {total_prompt_size} chars")
                
                # Save empty response debug info
                if self.cache_dir:
                    error_file = self.cache_dir.parent / "synthesis_error_debug.txt"
                    with open(error_file, 'w') as f:
                        f.write(f"Empty response received at {logger.name}\n")
                        f.write(f"Prompt size: {total_prompt_size} chars\n")
                        f.write(f"Number of chunks: {len(chunk_analyses)}\n")
                        f.write(f"Synopsis size: {len(synopsis)} chars\n")
                    logger.info(f"Saved error debug info to: {error_file}")
                
                raise Exception("Empty response from API for narrative synthesis")
            
            logger.info(f"Successfully parsed JSON response: {len(response)} chars")
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in narrative synthesis: {e}")
            logger.error(f"Response content: {response[:500] if response else 'EMPTY'}")
            
            # Save failed response
            if self.cache_dir:
                error_file = self.cache_dir.parent / "synthesis_json_error.txt"
                with open(error_file, 'w') as f:
                    f.write(f"JSON Decode Error: {e}\n\n")
                    f.write("=== RESPONSE CONTENT ===\n")
                    f.write(response if response else "EMPTY")
                logger.info(f"Saved failed response to: {error_file}")
            
            raise Exception(f"Failed to parse JSON response in narrative synthesis: {e}")
    
    def _stage3_generate_emotional_arc(self, scene_timeline: List[Dict]) -> List[Dict]:
        """
        Stage 3: Generate emotional intensity arc from scene timeline.
        
        Args:
            scene_timeline: Scenes from stage 2
        
        Returns:
            Emotional arc with scene_id, emotion, intensity
        """
        if not scene_timeline:
            return []
        
        # Build concise scene summary
        scene_summary = "\n".join([
            f"Scene {s['scene_id']}: {s.get('dominant_emotion', 'neutral')} - {s['summary'][:100]}"
            for s in scene_timeline
        ])
        
        system_prompt = """You are a narrative analyst. Generate an emotional intensity curve for the movie."""
        
        user_prompt = f"""Based on these scenes, assign emotion and intensity (0-1 scale):

{scene_summary}

Return JSON array:
[
  {{"scene_id": 1, "emotion": "calm", "intensity": 0.3}},
  {{"scene_id": 2, "emotion": "tense", "intensity": 0.7}}
]

Consider narrative pacing and emotional flow."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.azure_client.generate_structured_output(
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.synthesis_max_tokens
            )
            
            if not response or not response.strip():
                logger.warning("Empty response from API for emotional arc - returning empty arc")
                return []
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in emotional arc: {e} - returning empty arc")
            return []
    
    def _clear_cache(self):
        """Clear chunk analysis cache."""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("chunk_*.json"):
                cache_file.unlink()
            logger.info(f"Cleared cache directory: {self.cache_dir}")
    
    def validate_story_graph(self, story_graph: Dict) -> bool:
        """
        Validate story graph structure.
        
        Args:
            story_graph: Story graph dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'title',
            'logline',
            'characters',
            'major_themes',
            'plot_structure',
            'scene_timeline',
            'emotional_arc'
        ]
        
        # Check top-level fields
        for field in required_fields:
            if field not in story_graph:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check plot structure
        plot_fields = ['setup', 'inciting_incident', 'rising_action', 'climax', 'resolution']
        for field in plot_fields:
            if field not in story_graph['plot_structure']:
                logger.error(f"Missing plot structure field: {field}")
                return False
        
        # Check scene timeline has entries
        if not story_graph['scene_timeline']:
            logger.error("Scene timeline is empty")
            return False
        
        # Check scene structure (sample first scene)
        scene_fields = [
            'scene_id', 'start_time', 'end_time', 'summary',
            'key_events', 'characters_present', 'dominant_emotion',
            'genre_indicators', 'visual_inferences'
        ]
        
        if story_graph['scene_timeline']:
            scene = story_graph['scene_timeline'][0]
            for field in scene_fields:
                if field not in scene:
                    logger.error(f"Scene missing field: {field}")
                    return False
        
        logger.info("Story graph validation passed")
        return True
