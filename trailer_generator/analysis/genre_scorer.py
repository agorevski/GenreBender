"""
Genre-configurable scoring system for ranking shots.
Computes weighted scores based on genre-specific attributes.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class GenreScorer:
    """
    Computes genre-specific scores for shots based on analyzed attributes.
    Weights are configurable via genre profiles.
    """
    
    def __init__(self, genre_weights: Dict[str, float]):
        """
        Initialize genre scorer.
        
        Args:
            genre_weights: Dictionary mapping attribute names to weights
                Example: {'suspense': 0.35, 'darkness': 0.25, ...}
        """
        self.genre_weights = genre_weights
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weights sum to approximately 1.0.
        
        Logs a warning if the sum of genre weights deviates from 1.0
        by more than 0.01.
        """
        total = sum(self.genre_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Genre weights sum to {total:.3f}, expected 1.0")
    
    def score_shot(self, shot: Dict) -> float:
        """Compute genre-specific score for a single shot.
        
        Calculates a weighted score by multiplying each attribute value
        by its corresponding genre weight and summing the results.
        
        Args:
            shot: Shot dictionary with 'analysis' containing attributes.
                Expected structure: {'id': str, 'analysis': {'attributes': {...}}}
            
        Returns:
            float: Weighted score clamped between 0.0 and 1.0.
        """
        analysis = shot.get('analysis', {})
        attributes = analysis.get('attributes', {})
        
        if not attributes:
            logger.warning(f"No attributes found for shot {shot.get('id')}")
            return 0.0
        
        score = 0.0
        
        for attr_name, weight in self.genre_weights.items():
            attr_value = attributes.get(attr_name, 0.0)
            score += attr_value * weight
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def score_shots(self, shots: List[Dict]) -> List[Dict]:
        """Compute scores for multiple shots.
        
        Iterates through all shots, computing genre-specific scores and
        logging the score distribution statistics.
        
        Args:
            shots: List of shot dictionaries to score.
            
        Returns:
            list: The same shots list with 'score' field added to each shot.
        """
        logger.info(f"Scoring {len(shots)} shots with genre weights")
        
        for shot in shots:
            shot['score'] = self.score_shot(shot)
        
        # Log score distribution
        scores = [s['score'] for s in shots if 'score' in s]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            logger.info(f"Score distribution: min={min_score:.3f}, avg={avg_score:.3f}, max={max_score:.3f}")
        
        return shots
    
    def get_score_distribution(self, shots: List[Dict], bins: int = 10) -> Dict:
        """Get distribution of scores across shots.
        
        Creates a histogram of score values to analyze the distribution
        of scores across all shots.
        
        Args:
            shots: List of scored shots containing 'score' field.
            bins: Number of histogram bins. Defaults to 10.
            
        Returns:
            dict: Dictionary with distribution statistics containing:
                - bins (list): Bin edge values.
                - counts (list): Count of shots in each bin.
                - min (float): Minimum score.
                - max (float): Maximum score.
                - mean (float): Average score.
                - total_shots (int): Total number of scored shots.
        """
        scores = [s.get('score', 0.0) for s in shots]
        
        if not scores:
            return {'bins': [], 'counts': []}
        
        # Create histogram
        min_score = min(scores)
        max_score = max(scores)
        bin_width = (max_score - min_score) / bins if max_score > min_score else 1.0
        
        histogram = [0] * bins
        bin_edges = []
        
        for i in range(bins):
            bin_edges.append(min_score + i * bin_width)
        
        for score in scores:
            if score == max_score:
                histogram[-1] += 1
            else:
                bin_idx = int((score - min_score) / bin_width)
                histogram[bin_idx] += 1
        
        return {
            'bins': bin_edges,
            'counts': histogram,
            'min': min_score,
            'max': max_score,
            'mean': sum(scores) / len(scores),
            'total_shots': len(scores)
        }
    
    def get_top_attributes(self, shot: Dict) -> List[tuple]:
        """Get top contributing attributes for a shot's score.
        
        Analyzes which attributes contributed most to a shot's final score
        by computing the weighted contribution of each attribute.
        
        Args:
            shot: Shot dictionary with analysis containing attributes.
            
        Returns:
            list: List of (attribute_name, contribution) tuples, sorted
                by contribution in descending order.
        """
        analysis = shot.get('analysis', {})
        attributes = analysis.get('attributes', {})
        
        contributions = []
        
        for attr_name, weight in self.genre_weights.items():
            attr_value = attributes.get(attr_name, 0.0)
            contribution = attr_value * weight
            contributions.append((attr_name, contribution))
        
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        return contributions
