"""
Embedding generation for semantic scene retrieval.

Generates vector embeddings for:
1. Scenes/shots from shot metadata (visual + audio + narrative context)
2. Beats from beat sheet embedding prompts

Uses Azure OpenAI embeddings API for dense vector representations.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for scenes and beats using Azure OpenAI."""
    
    def __init__(
        self,
        azure_client,
        embedding_model: str = "text-embedding-ada-002",
        batch_size: int = 10,
        max_input_tokens: int = 8000
    ):
        """
        Initialize embedding generator.
        
        Args:
            azure_client: Azure OpenAI client instance
            embedding_model: Embedding model name
            batch_size: Number of texts to embed per API call
            max_input_tokens: Maximum tokens per input text
        """
        self.azure_client = azure_client
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        
    def generate_scene_embeddings(
        self,
        shot_metadata: List[Dict],
        story_graph: Optional[Dict] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generate embeddings for all scenes/shots.
        
        Creates rich text representations combining:
        - Scene caption/description
        - Visual attributes (suspense, darkness, intensity, etc.)
        - Audio features (energy, spectral qualities)
        - Story graph context (characters, emotions, genre indicators)
        
        Args:
            shot_metadata: List of shot dictionaries from shot_metadata.json
            story_graph: Optional story graph for enriched context
        
        Returns:
            Tuple of (embeddings array, shot_ids list)
        """
        logger.info(f"Generating embeddings for {len(shot_metadata)} shots...")
        
        # Build scene timeline mapping if story graph available
        scene_context = {}
        if story_graph and 'scene_timeline' in story_graph:
            scene_context = self._build_scene_context_map(story_graph)
        
        # Generate text representations for each shot
        texts = []
        shot_ids = []
        
        for shot in shot_metadata:
            shot_id = shot['shot_id']
            text = self._build_scene_text(shot, scene_context)
            texts.append(text)
            shot_ids.append(shot_id)
        
        # Generate embeddings in batches
        embeddings = self._generate_embeddings_batch(texts)
        
        logger.info(f"Generated {len(embeddings)} scene embeddings (dim={embeddings.shape[1]})")
        return embeddings, shot_ids
    
    def generate_beat_embeddings(
        self,
        beats: List[Dict]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for trailer beats.
        
        Uses the rich embedding_prompt from each beat, which includes:
        - Visual requirements
        - Emotional tone
        - Narrative context
        - Genre-specific keywords
        
        Args:
            beats: List of beat dictionaries from beats.json
        
        Returns:
            Tuple of (embeddings array, beat_ids list)
        """
        logger.info(f"Generating embeddings for {len(beats)} beats...")
        
        texts = []
        beat_ids = []
        
        for beat in beats:
            beat_id = beat['id']
            # Use the pre-crafted embedding_prompt
            text = beat.get('embedding_prompt', '')
            
            # Fallback: build from beat components if no prompt
            if not text:
                text = self._build_beat_text(beat)
            
            texts.append(text)
            beat_ids.append(beat_id)
        
        # Generate embeddings in batches
        embeddings = self._generate_embeddings_batch(texts)
        
        logger.info(f"Generated {len(embeddings)} beat embeddings (dim={embeddings.shape[1]})")
        return embeddings, beat_ids
    
    def _build_scene_text(
        self,
        shot: Dict,
        scene_context: Dict
    ) -> str:
        """
        Build rich text representation for a shot/scene.
        
        Combines multiple information sources:
        1. Visual analysis (caption, attributes)
        2. Audio features (energy, mood)
        3. Story context (characters, emotions, events)
        4. Temporal position in narrative
        """
        parts = []
        
        # 1. Scene caption/description
        if 'caption' in shot:
            parts.append(shot['caption'])
        
        # 2. Visual attributes
        if 'attributes' in shot:
            attrs = shot['attributes']
            # High-value attributes (>0.5)
            high_attrs = [
                f"{k}: {v:.2f}"
                for k, v in attrs.items()
                if isinstance(v, (int, float)) and v > 0.5
            ]
            if high_attrs:
                parts.append(f"Visual qualities: {', '.join(high_attrs)}")
        
        # 3. Audio characteristics
        if 'audio_features' in shot:
            audio = shot['audio_features']
            audio_desc = []
            
            # Energy level
            if 'rms_energy_mean' in audio:
                energy = audio['rms_energy_mean']
                if energy > 0.05:
                    audio_desc.append("high energy")
                elif energy < 0.02:
                    audio_desc.append("quiet/ambient")
            
            # Spectral characteristics
            if 'spectral_centroid_mean' in audio:
                centroid = audio['spectral_centroid_mean']
                if centroid > 3000:
                    audio_desc.append("bright/sharp sounds")
                elif centroid < 1500:
                    audio_desc.append("dark/low tones")
            
            if audio_desc:
                parts.append(f"Audio: {', '.join(audio_desc)}")
        
        # 4. Story graph context
        start_time = shot.get('start_time', 0)
        if start_time in scene_context:
            ctx = scene_context[start_time]
            
            # Characters present
            if ctx.get('characters_present'):
                chars = ', '.join(ctx['characters_present'])
                parts.append(f"Characters: {chars}")
            
            # Key events
            if ctx.get('key_events'):
                events = '. '.join(ctx['key_events'][:2])  # Top 2 events
                parts.append(f"Events: {events}")
            
            # Dominant emotion
            if ctx.get('dominant_emotion'):
                parts.append(f"Emotion: {ctx['dominant_emotion']}")
            
            # Genre indicators
            if ctx.get('genre_indicators'):
                indicators = ', '.join(ctx['genre_indicators'][:3])
                parts.append(f"Genre markers: {indicators}")
            
            # Visual inferences
            if ctx.get('visual_inferences'):
                visuals = ', '.join(ctx['visual_inferences'])
                parts.append(f"Setting: {visuals}")
        
        # 5. Temporal context
        duration = shot.get('duration', 0)
        if duration > 0:
            parts.append(f"Duration: {duration:.1f}s")
        
        return '. '.join(parts)
    
    def _build_beat_text(self, beat: Dict) -> str:
        """
        Build text representation for a beat (fallback if no embedding_prompt).
        """
        parts = []
        
        # Beat description
        if beat.get('description'):
            parts.append(beat['description'])
        
        # Target emotion
        if beat.get('target_emotion'):
            parts.append(f"Emotion: {beat['target_emotion']}")
        
        # Visual requirements
        if beat.get('visual_requirements'):
            visuals = '. '.join(beat['visual_requirements'])
            parts.append(f"Visual: {visuals}")
        
        # Audio cue
        if beat.get('audio_cue'):
            parts.append(f"Audio: {beat['audio_cue']}")
        
        # Voiceover
        if beat.get('voiceover'):
            parts.append(f"VO: {beat['voiceover']}")
        
        return '. '.join(parts)
    
    def _build_scene_context_map(self, story_graph: Dict) -> Dict:
        """
        Build mapping from timestamp to scene context.
        
        Returns:
            Dict mapping start_time (seconds) to scene info
        """
        context_map = {}
        
        for scene in story_graph.get('scene_timeline', []):
            # Parse timestamp (HH:MM:SS or MM:SS)
            start_time_str = scene.get('start_time', '00:00:00')
            start_seconds = self._parse_timestamp(start_time_str)
            
            context_map[start_seconds] = {
                'scene_id': scene.get('scene_id'),
                'summary': scene.get('summary', ''),
                'key_events': scene.get('key_events', []),
                'characters_present': scene.get('characters_present', []),
                'dominant_emotion': scene.get('dominant_emotion', ''),
                'genre_indicators': scene.get('genre_indicators', []),
                'visual_inferences': scene.get('visual_inferences', [])
            }
        
        return context_map
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Convert HH:MM:SS or MM:SS to seconds."""
        parts = timestamp.split(':')
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        else:
            return float(parts[0])
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts in batches.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                # Call Azure OpenAI embeddings API
                response = self.azure_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated embeddings for batch {i//self.batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                # Fallback: zero vectors
                embedding_dim = 1536  # Default for ada-002
                batch_embeddings = [np.zeros(embedding_dim).tolist() for _ in batch]
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List,
        output_path: Path,
        metadata: Optional[Dict] = None
    ):
        """
        Save embeddings to pickle file.
        
        Args:
            embeddings: Numpy array of embeddings
            ids: List of IDs (shot_ids or beat_ids)
            output_path: Path to save pickle file
            metadata: Optional metadata dictionary
        """
        data = {
            'embeddings': embeddings,
            'ids': ids,
            'metadata': metadata or {}
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved embeddings to {output_path}")
    
    def load_embeddings(self, path: Path) -> Tuple[np.ndarray, List]:
        """
        Load embeddings from pickle file.
        
        Returns:
            Tuple of (embeddings array, ids list)
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data['ids']


def generate_embeddings(
    output_dir: Path,
    shot_metadata_path: Path,
    beats_path: Path,
    story_graph_path: Optional[Path],
    azure_client,
    config: Dict
) -> Tuple[Path, Path]:
    """
    Main function to generate and save embeddings.
    
    Args:
        output_dir: Output directory for embeddings
        shot_metadata_path: Path to shot_metadata.json
        beats_path: Path to beats.json
        story_graph_path: Optional path to story_graph.json
        azure_client: Azure OpenAI client
        config: Configuration dictionary
    
    Returns:
        Tuple of (scene_embeddings_path, beat_embeddings_path)
    """
    # Load data
    with open(shot_metadata_path) as f:
        shot_metadata = json.load(f)
    
    with open(beats_path) as f:
        beats_data = json.load(f)
        beats = beats_data.get('beats', [])
    
    story_graph = None
    if story_graph_path and story_graph_path.exists():
        with open(story_graph_path) as f:
            story_graph = json.load(f)
    
    # Initialize generator
    embedding_config = config.get('embedding', {})
    generator = EmbeddingGenerator(
        azure_client=azure_client,
        embedding_model=embedding_config.get('model'),
        batch_size=embedding_config.get('batch_size'),
        max_input_tokens=embedding_config.get('max_input_tokens')
    )
    
    # Generate scene embeddings
    scene_embeddings, shot_ids = generator.generate_scene_embeddings(
        shot_metadata=shot_metadata,
        story_graph=story_graph
    )
    
    scene_embeddings_path = output_dir / 'scene_embeddings.pkl'
    generator.save_embeddings(
        embeddings=scene_embeddings,
        ids=shot_ids,
        output_path=scene_embeddings_path,
        metadata={
            'source': 'shot_metadata.json',
            'model': embedding_config.get('model'),
            'count': len(shot_ids)
        }
    )
    
    # Generate beat embeddings
    beat_embeddings, beat_ids = generator.generate_beat_embeddings(beats=beats)
    
    beat_embeddings_path = output_dir / 'beat_embeddings.pkl'
    generator.save_embeddings(
        embeddings=beat_embeddings,
        ids=beat_ids,
        output_path=beat_embeddings_path,
        metadata={
            'source': 'beats.json',
            'model': embedding_config.get('model'),
            'count': len(beat_ids)
        }
    )
    
    logger.info(f"Embedding generation complete:")
    logger.info(f"  Scenes: {len(shot_ids)} embeddings")
    logger.info(f"  Beats: {len(beat_ids)} embeddings")
    
    return scene_embeddings_path, beat_embeddings_path
