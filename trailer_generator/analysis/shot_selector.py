"""
Shot selection utilities for choosing top candidates for trailers.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ShotSelector:
    """
    Selects top-scoring shots for trailer generation.
    Supports various selection strategies.
    """
    
    def __init__(self, target_count: int = 60):
        """Initialize shot selector.
        
        Args:
            target_count: Target number of shots to select.
        """
        self.target_count = target_count
    
    def select_top_shots(self, shots: List[Dict]) -> List[Dict]:
        """Select top N shots by score.
        
        Args:
            shots: List of scored shots with 'score' key.
            
        Returns:
            List[Dict]: Top N shots, sorted by score descending.
        """
        # Filter shots with valid scores
        valid_shots = [s for s in shots if 'score' in s and s['score'] > 0]
        
        if len(valid_shots) == 0:
            logger.warning("No valid scored shots found")
            return []
        
        # Sort by score descending
        sorted_shots = sorted(valid_shots, key=lambda x: x['score'], reverse=True)
        
        # Take top N
        selected = sorted_shots[:self.target_count]
        
        logger.info(f"Selected {len(selected)} top shots from {len(shots)} total")
        
        if selected:
            logger.info(f"Top shot score: {selected[0]['score']:.3f}")
            logger.info(f"Lowest selected score: {selected[-1]['score']:.3f}")
        
        return selected
    
    def select_diverse_shots(self, shots: List[Dict], 
                            temporal_diversity: float = 0.3) -> List[Dict]:
        """Select shots with temporal diversity to avoid clustering.
        
        Args:
            shots: List of scored shots with 'score' and 'start_time' keys.
            temporal_diversity: Minimum time gap as a fraction of total duration.
            
        Returns:
            List[Dict]: Selected shots with temporal spacing, sorted by start time.
        """
        valid_shots = [s for s in shots if 'score' in s and s['score'] > 0]
        
        if not valid_shots:
            return []
        
        # Sort by score
        sorted_shots = sorted(valid_shots, key=lambda x: x['score'], reverse=True)
        
        # Get total video duration
        max_time = max(s.get('end_time', 0) for s in sorted_shots)
        min_gap = max_time * temporal_diversity
        
        selected = []
        
        for shot in sorted_shots:
            if len(selected) >= self.target_count:
                break
            
            # Check if shot is temporally diverse from selected shots
            shot_time = shot.get('start_time', 0)
            is_diverse = all(
                abs(shot_time - s.get('start_time', 0)) > min_gap
                for s in selected
            )
            
            if is_diverse or len(selected) == 0:
                selected.append(shot)
        
        # Sort selected shots by time for narrative flow
        selected.sort(key=lambda x: x.get('start_time', 0))
        
        logger.info(f"Selected {len(selected)} diverse shots (min gap: {min_gap:.1f}s)")
        
        return selected
    
    def select_by_narrative_arc(self, shots: List[Dict], 
                                arc_stages: List[str]) -> Dict[str, List[Dict]]:
        """Select shots organized by narrative arc stages.
        
        Args:
            shots: List of scored shots with 'score' and 'start_time' keys.
            arc_stages: List of arc stage names (e.g., ['intro', 'rising', 'climax']).
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping stage names to shot lists.
        """
        valid_shots = [s for s in shots if 'score' in s and s['score'] > 0]
        
        if not valid_shots:
            return {stage: [] for stage in arc_stages}
        
        # Sort by time
        sorted_shots = sorted(valid_shots, key=lambda x: x.get('start_time', 0))
        
        # Divide shots into temporal sections
        num_stages = len(arc_stages)
        shots_per_stage = len(sorted_shots) // num_stages
        
        arc_shots = {}
        
        for i, stage in enumerate(arc_stages):
            start_idx = i * shots_per_stage
            end_idx = (i + 1) * shots_per_stage if i < num_stages - 1 else len(sorted_shots)
            
            stage_shots = sorted_shots[start_idx:end_idx]
            
            # Select top shots from this stage
            stage_count = self.target_count // num_stages
            stage_sorted = sorted(stage_shots, key=lambda x: x['score'], reverse=True)
            arc_shots[stage] = stage_sorted[:stage_count]
        
        logger.info(f"Selected shots by narrative arc: {[(s, len(arc_shots[s])) for s in arc_stages]}")
        
        return arc_shots
    
    def filter_by_duration(self, shots: List[Dict], 
                          min_duration: float = 0.5,
                          max_duration: float = 10.0) -> List[Dict]:
        """Filter shots by duration constraints.
        
        Args:
            shots: List of shots with 'duration' key.
            min_duration: Minimum shot duration in seconds.
            max_duration: Maximum shot duration in seconds.
            
        Returns:
            List[Dict]: Filtered shots within the duration range.
        """
        filtered = [
            s for s in shots
            if min_duration <= s.get('duration', 0) <= max_duration
        ]
        
        logger.info(f"Duration filter: {len(filtered)}/{len(shots)} shots kept")
        
        return filtered
    
    def balance_attributes(self, shots: List[Dict], 
                          target_attributes: Dict[str, tuple]) -> List[Dict]:
        """Select shots to balance specific attribute ranges.
        
        Args:
            shots: List of scored shots with 'score' and 'analysis' keys.
            target_attributes: Dict mapping attribute name to (min, max) target range.
            
        Returns:
            List[Dict]: Balanced selection of shots matching attribute criteria.
        """
        valid_shots = [s for s in shots if 'score' in s]
        
        if not valid_shots:
            return []
        
        selected = []
        
        # Sort by score
        sorted_shots = sorted(valid_shots, key=lambda x: x['score'], reverse=True)
        
        for shot in sorted_shots:
            if len(selected) >= self.target_count:
                break
            
            analysis = shot.get('analysis', {})
            attributes = analysis.get('attributes', {})
            
            # Check if shot fits target attribute ranges
            fits_criteria = True
            for attr_name, (min_val, max_val) in target_attributes.items():
                attr_value = attributes.get(attr_name, 0.0)
                if not (min_val <= attr_value <= max_val):
                    fits_criteria = False
                    break
            
            if fits_criteria:
                selected.append(shot)
        
        logger.info(f"Attribute balance: selected {len(selected)} shots matching criteria")
        
        return selected
