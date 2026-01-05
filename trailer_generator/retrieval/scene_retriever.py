"""
Scene retrieval module for semantic beat-to-scene matching.

Uses FAISS for efficient similarity search and multi-factor scoring:
1. Semantic similarity (cosine distance from embeddings)
2. Emotional alignment (beat emotion vs scene emotion)
3. Visual attribute matching (beat requirements vs scene attributes)
4. Original genre penalty (penalize scenes that match original genre too closely)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
    faiss = None

class SceneRetriever:
    """Retrieve best-matching scenes for each trailer beat using semantic search."""
    
    def __init__(
        self,
        scene_embeddings: np.ndarray,
        scene_ids: List[int],
        shot_metadata: List[Dict],
        scoring_weights: Optional[Dict] = None
    ):
        """
        Initialize scene retriever.
        
        Args:
            scene_embeddings: Scene embedding vectors (n_scenes, embedding_dim)
            scene_ids: List of shot IDs aligned with embeddings
            shot_metadata: Full shot metadata for scoring
            scoring_weights: Weights for multi-factor scoring
        """
        if faiss is None:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        self.scene_embeddings = scene_embeddings.astype('float32')
        self.scene_ids = scene_ids
        self.shot_metadata = {shot['id']: shot for shot in shot_metadata}
        
        # Default scoring weights
        self.weights = scoring_weights or {
            'semantic_similarity': 0.50,      # Primary: embedding similarity
            'emotional_alignment': 0.25,      # Secondary: emotion match
            'visual_match': 0.20,             # Tertiary: visual attributes
            'original_genre_penalty': 0.05    # Small penalty for original genre
        }
        
        # Build FAISS index
        self.index = self._build_faiss_index()
        
        logger.info(f"Scene retriever initialized with {len(scene_ids)} scenes")
    
    def _build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for efficient similarity search.
        
        Uses IndexFlatIP (inner product) which is equivalent to cosine similarity
        when vectors are L2-normalized.
        
        Returns:
            faiss.Index: FAISS index configured for inner product similarity search.
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.scene_embeddings)
        
        # Create index
        dimension = self.scene_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
        index.add(self.scene_embeddings)
        
        logger.debug(f"Built FAISS index with {index.ntotal} vectors, dim={dimension}")
        return index
    
    def retrieve_scenes_for_beat(
        self,
        beat: Dict,
        beat_embedding: np.ndarray,
        top_k: int = 10,
        target_genre: str = 'thriller'
    ) -> List[Dict]:
        """
        Retrieve and rank scenes for a single beat.
        
        Args:
            beat: Beat dictionary with requirements
            beat_embedding: Beat embedding vector
            top_k: Number of candidates to retrieve
            target_genre: Target genre for scoring
        
        Returns:
            List of candidate dictionaries sorted by score (highest first)
        """
        # Step 1: FAISS semantic search
        beat_embedding_norm = beat_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(beat_embedding_norm)
        
        # Retrieve more candidates than needed for reranking
        search_k = min(top_k * 3, len(self.scene_ids))
        similarities, indices = self.index.search(beat_embedding_norm, search_k)
        
        # Step 2: Multi-factor scoring
        candidates = []
        for idx, similarity in zip(indices[0], similarities[0]):
            shot_id = self.scene_ids[idx]
            shot = self.shot_metadata.get(shot_id)
            
            if not shot:
                continue
            
            # Calculate multi-factor score
            score = self._calculate_score(
                beat=beat,
                shot=shot,
                semantic_similarity=float(similarity),
                target_genre=target_genre
            )
            
            candidates.append({
                'shot_id': shot_id,
                'score': score,
                'semantic_similarity': float(similarity),
                'start_time': shot.get('start_time', 0),
                'end_time': shot.get('end_time', 0),
                'duration': shot.get('duration', 0),
                'caption': shot.get('caption', ''),
                'attributes': shot.get('attributes', {}),
                'shot_path': shot.get('shot_path', '')
            })
        
        # Step 3: Sort by total score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]
    
    def retrieve_all_beats(
        self,
        beats: List[Dict],
        beat_embeddings: np.ndarray,
        top_k: int = 10,
        target_genre: str = 'thriller'
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve scenes for all beats.
        
        Args:
            beats: List of beat dictionaries
            beat_embeddings: Beat embedding array
            top_k: Number of candidates per beat
            target_genre: Target genre
        
        Returns:
            Dictionary mapping beat_id to list of candidate scenes
        """
        results = {}
        
        for i, beat in enumerate(beats):
            beat_id = beat['id']
            beat_embedding = beat_embeddings[i]
            
            candidates = self.retrieve_scenes_for_beat(
                beat=beat,
                beat_embedding=beat_embedding,
                top_k=top_k,
                target_genre=target_genre
            )
            
            results[beat_id] = candidates
            
            logger.debug(
                f"Beat {beat_id}: Retrieved {len(candidates)} candidates "
                f"(top score: {candidates[0]['score']:.3f})"
            )
        
        logger.info(f"Retrieved scenes for {len(beats)} beats")
        return results
    
    def _calculate_score(
        self,
        beat: Dict,
        shot: Dict,
        semantic_similarity: float,
        target_genre: str
    ) -> float:
        """Calculate multi-factor score for beat-scene match.
        
        Score = w1*semantic + w2*emotion + w3*visual - w4*genre_penalty
        
        Args:
            beat: Beat dictionary containing target emotion and visual requirements.
            shot: Shot dictionary containing attributes and metadata.
            semantic_similarity: Pre-computed cosine similarity from FAISS search.
            target_genre: Target genre for the trailer transformation.
        
        Returns:
            float: Weighted composite score between 0 and 1.
        """
        # Factor 1: Semantic similarity (from FAISS)
        semantic_score = semantic_similarity
        
        # Factor 2: Emotional alignment
        emotional_score = self._calculate_emotional_alignment(
            beat.get('target_emotion', ''),
            shot.get('attributes', {})
        )
        
        # Factor 3: Visual attribute matching
        visual_score = self._calculate_visual_match(
            beat.get('visual_requirements', []),
            shot.get('attributes', {})
        )
        
        # Factor 4: Original genre penalty
        genre_penalty = self._calculate_genre_penalty(
            shot.get('attributes', {}),
            target_genre
        )
        
        # Weighted combination
        total_score = (
            self.weights['semantic_similarity'] * semantic_score +
            self.weights['emotional_alignment'] * emotional_score +
            self.weights['visual_match'] * visual_score -
            self.weights['original_genre_penalty'] * genre_penalty
        )
        
        return total_score
    
    def _calculate_emotional_alignment(
        self,
        target_emotion: str,
        attributes: Dict
    ) -> float:
        """Calculate how well scene emotion matches beat emotion.
        
        Maps target emotions to relevant attribute values using a predefined
        emotion-to-attribute mapping.
        
        Args:
            target_emotion: Desired emotional tone for the beat (e.g., 'suspense', 'fear').
            attributes: Scene attribute dictionary with numeric scores.
        
        Returns:
            float: Emotional alignment score between 0 and 1, with 0.5 as neutral.
        """
        if not target_emotion or not attributes:
            return 0.5  # Neutral score
        
        # Emotion-to-attribute mapping
        emotion_map = {
            'suspense': ['suspense', 'tension', 'mystery'],
            'fear': ['darkness', 'suspense', 'horror'],
            'terror': ['horror', 'darkness', 'intensity'],
            'excitement': ['intensity', 'motion', 'energy'],
            'tension': ['tension', 'suspense', 'conflict'],
            'wonder': ['beauty', 'scale', 'mystery'],
            'sadness': ['emotional_weight', 'darkness'],
            'joy': ['brightness', 'warmth'],
            'curiosity': ['mystery', 'intrigue'],
            'dread': ['darkness', 'suspense', 'foreboding'],
            'relief': ['brightness', 'calm'],
            'shock': ['intensity', 'surprise'],
            'unease': ['suspense', 'tension', 'darkness']
        }
        
        relevant_attrs = emotion_map.get(target_emotion.lower(), [])
        if not relevant_attrs:
            return 0.5
        
        # Average relevant attribute scores
        scores = []
        for attr in relevant_attrs:
            if attr in attributes:
                scores.append(attributes[attr])
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_visual_match(
        self,
        visual_requirements: List[str],
        attributes: Dict
    ) -> float:
        """Calculate how well scene attributes match visual requirements.
        
        Performs keyword matching between requirements and attribute names,
        then averages the matched attribute scores.
        
        Args:
            visual_requirements: List of visual requirement strings from the beat.
            attributes: Scene attribute dictionary with numeric scores.
        
        Returns:
            float: Visual match score between 0 and 1, with 0.5 as neutral.
        """
        if not visual_requirements or not attributes:
            return 0.5
        
        # Extract keywords from requirements
        keywords = set()
        for req in visual_requirements:
            # Simple keyword extraction (lowercase, split on spaces)
            words = req.lower().replace(',', '').split()
            keywords.update(words)
        
        # Match against attribute names
        relevant_attrs = []
        for attr_name, attr_value in attributes.items():
            # Check if any keyword appears in attribute name
            attr_words = attr_name.lower().replace('_', ' ').split()
            if keywords.intersection(attr_words):
                if isinstance(attr_value, (int, float)):
                    relevant_attrs.append(attr_value)
        
        # Average matched attribute scores
        if relevant_attrs:
            return np.mean(relevant_attrs)
        
        # Fallback: use high-value attributes
        high_attrs = [v for v in attributes.values() if isinstance(v, (int, float)) and v > 0.6]
        return np.mean(high_attrs) if high_attrs else 0.5
    
    def _calculate_genre_penalty(
        self,
        attributes: Dict,
        target_genre: str
    ) -> float:
        """Penalize scenes that strongly match the original (non-target) genre.
        
        This helps select scenes that feel "off" or reinterpretable by avoiding
        scenes with strong indicators of genres other than the target.
        
        Args:
            attributes: Scene attribute dictionary with numeric scores.
            target_genre: Target genre for the trailer transformation.
        
        Returns:
            float: Genre penalty score between 0 and 1, where higher means more penalty.
        """
        # Genre indicator attributes (these often match original genre)
        genre_indicators = {
            'comedy': ['humor', 'lightheartedness', 'brightness'],
            'drama': ['emotional_weight', 'character_focus'],
            'action': ['motion', 'intensity', 'energy'],
            'horror': ['horror', 'darkness', 'fear'],
            'thriller': ['suspense', 'tension', 'mystery'],
            'romance': ['warmth', 'intimacy', 'beauty'],
            'scifi': ['futurism', 'technology', 'scale']
        }
        
        # We want to avoid strong indicators of OTHER genres
        penalty = 0.0
        for genre, indicators in genre_indicators.items():
            if genre == target_genre:
                continue  # Don't penalize target genre
            
            # Check how strongly scene matches this other genre
            genre_scores = []
            for indicator in indicators:
                if indicator in attributes:
                    genre_scores.append(attributes[indicator])
            
            if genre_scores:
                # Penalize if scene strongly matches another genre
                genre_strength = np.mean(genre_scores)
                if genre_strength > 0.7:
                    penalty += genre_strength
        
        return min(penalty, 1.0)  # Cap at 1.0

def retrieve_scenes(
    embeddings_dir: Path,
    beats_path: Path,
    shot_metadata_path: Path,
    output_path: Path,
    target_genre: str,
    top_k: int = 10,
    scoring_weights: Optional[Dict] = None
) -> Dict:
    """
    Main function for scene retrieval.
    
    Args:
        embeddings_dir: Directory containing embedding files
        beats_path: Path to beats.json
        shot_metadata_path: Path to shot_metadata.json
        output_path: Path to save selected_scenes.json
        target_genre: Target trailer genre
        top_k: Number of candidates per beat
        scoring_weights: Optional custom scoring weights
    
    Returns:
        Dictionary of beat_id -> candidate scenes
    """
    # Load embeddings
    scene_emb_path = embeddings_dir / 'scene_embeddings.pkl'
    beat_emb_path = embeddings_dir / 'beat_embeddings.pkl'
    
    with open(scene_emb_path, 'rb') as f:
        scene_data = pickle.load(f)
        scene_embeddings = scene_data['embeddings']
        scene_ids = scene_data['ids']
    
    with open(beat_emb_path, 'rb') as f:
        beat_data = pickle.load(f)
        beat_embeddings = beat_data['embeddings']
        beat_ids = beat_data['ids']
    
    # Load metadata
    with open(shot_metadata_path) as f:
        shot_metadata_json = json.load(f)
        # Handle both formats: direct list or wrapped in "shots" key
        if isinstance(shot_metadata_json, dict) and 'shots' in shot_metadata_json:
            shot_metadata = shot_metadata_json['shots']
        elif isinstance(shot_metadata_json, list):
            shot_metadata = shot_metadata_json
        else:
            raise ValueError(f"Unexpected shot_metadata format: {type(shot_metadata_json)}")
    
    with open(beats_path) as f:
        beats_data = json.load(f)
        beats = beats_data['beats']
    
    # Initialize retriever
    retriever = SceneRetriever(
        scene_embeddings=scene_embeddings,
        scene_ids=scene_ids,
        shot_metadata=shot_metadata,
        scoring_weights=scoring_weights
    )
    
    # Retrieve scenes for all beats
    results = retriever.retrieve_all_beats(
        beats=beats,
        beat_embeddings=beat_embeddings,
        top_k=top_k,
        target_genre=target_genre
    )
    
    # Save results
    output_data = {
        'target_genre': target_genre,
        'beats': beats,
        'selected_scenes': results,
        'metadata': {
            'total_beats': len(beats),
            'candidates_per_beat': top_k,
            'scoring_weights': retriever.weights
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved scene selections to {output_path}")
    
    # Log statistics
    total_candidates = sum(len(candidates) for candidates in results.values())
    avg_top_score = np.mean([
        candidates[0]['score'] for candidates in results.values() if candidates
    ])
    
    logger.info(f"Scene retrieval statistics:")
    logger.info(f"  Total candidates: {total_candidates}")
    logger.info(f"  Avg top score: {avg_top_score:.3f}")
    
    return results
