"""
AI-powered transition selection for trailers.
Intelligently selects transition types based on shot analysis and pacing.
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TransitionSelector:
    """
    Selects optimal transitions between shots using AI or rule-based logic.
    """
    
    # Transition types supported by FFmpeg xfade filter
    TRANSITION_TYPES = [
        'fade',           # Simple crossfade
        'wipeleft',       # Wipe from left
        'wiperight',      # Wipe from right
        'wipeup',         # Wipe from bottom
        'wipedown',       # Wipe from top
        'slideleft',      # Slide from left
        'slideright',     # Slide from right
        'circlecrop',     # Circle crop
        'rectcrop',       # Rectangle crop
        'dissolve'        # Dissolve
    ]
    
    def __init__(self, azure_client, genre_profile: Dict, enable_ai: bool = True):
        """
        Initialize transition selector.
        
        Args:
            azure_client: Azure OpenAI client instance (can be None if enable_ai=False)
            genre_profile: Genre profile with pacing information
            enable_ai: Whether to use AI for transition selection
        """
        self.azure_client = azure_client
        self.genre_profile = genre_profile
        self.enable_ai = enable_ai
        self.pacing = genre_profile.get('pacing', 'measured')
    
    def select_transitions(self, timeline: Dict, shot_metadata: List[Dict]) -> List[Dict]:
        """
        Select transitions for all shots in timeline.
        
        Args:
            timeline: Timeline dictionary with shot sequence
            shot_metadata: Full shot metadata with analysis
            
        Returns:
            List of transition dictionaries with type and duration
        """
        timeline_shots = timeline.get('timeline', [])
        if len(timeline_shots) < 2:
            return []
        
        transitions = []
        
        for i in range(len(timeline_shots) - 1):
            current_shot = timeline_shots[i]
            next_shot = timeline_shots[i + 1]
            
            # Get full analysis for both shots
            curr_analysis = self._get_shot_analysis(current_shot['shot_id'], shot_metadata)
            next_analysis = self._get_shot_analysis(next_shot['shot_id'], shot_metadata)
            
            # Select transition
            if self.enable_ai and self.azure_client and i < 5:  # Use AI for first few transitions to save costs
                try:
                    transition = self._select_ai_transition(
                        i, current_shot, next_shot, curr_analysis, next_analysis
                    )
                except Exception as e:
                    logger.warning(f"AI transition selection failed for shot {i}: {e}")
                    transition = self._select_rule_based_transition(
                        current_shot, next_shot, curr_analysis, next_analysis
                    )
            else:
                transition = self._select_rule_based_transition(
                    current_shot, next_shot, curr_analysis, next_analysis
                )
            
            transitions.append({
                'shot_index': i,
                'type': transition,
                'duration': self._get_transition_duration(),
                'offset': self._calculate_transition_offset(current_shot, next_shot)
            })
        
        logger.info(f"Selected {len(transitions)} transitions")
        return transitions
    
    def _select_ai_transition(self, index: int, current_shot: Dict, next_shot: Dict,
                             curr_analysis: Dict, next_analysis: Dict) -> str:
        """
        Use AI to select optimal transition.
        
        Args:
            index: Transition index
            current_shot: Current shot data
            next_shot: Next shot data
            curr_analysis: Current shot analysis
            next_analysis: Next shot analysis
            
        Returns:
            Transition type name
        """
        # Build concise prompt
        prompt = f"""Select the best transition between these two trailer shots:

Shot {index + 1}:
- Duration: {current_shot.get('duration', 2)}s
- Caption: {curr_analysis.get('caption', 'N/A')[:100]}
- Intensity: {curr_analysis.get('attributes', {}).get('intensity', 'medium')}

Shot {index + 2}:
- Duration: {next_shot.get('duration', 2)}s
- Caption: {next_analysis.get('caption', 'N/A')[:100]}
- Intensity: {next_analysis.get('attributes', {}).get('intensity', 'medium')}

Trailer pacing: {self.pacing}

Choose from: {', '.join(self.TRANSITION_TYPES[:6])}  # Limit choices

Consider:
- Pacing and rhythm
- Emotional continuity
- Visual style match

Return ONLY the transition name, nothing else."""
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional trailer editor. Return only the transition name."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Generate with low token limit
        response = self.azure_client.generate_structured_output(
            messages=messages,
            max_completion_tokens=50
        )
        
        # Parse and validate
        transition = response.strip().lower()
        if transition in self.TRANSITION_TYPES:
            logger.debug(f"AI selected transition: {transition}")
            return transition
        else:
            logger.warning(f"AI returned invalid transition '{transition}', using fallback")
            return 'fade'
    
    def _select_rule_based_transition(self, current_shot: Dict, next_shot: Dict,
                                     curr_analysis: Dict, next_analysis: Dict) -> str:
        """
        Select transition using rule-based logic.
        
        Args:
            current_shot: Current shot data
            next_shot: Next shot data
            curr_analysis: Current shot analysis
            next_analysis: Next shot analysis
            
        Returns:
            Transition type name
        """
        # Get intensity values
        curr_intensity = curr_analysis.get('attributes', {}).get('intensity', 0.5)
        next_intensity = next_analysis.get('attributes', {}).get('intensity', 0.5)
        
        # Convert to numeric if string
        if isinstance(curr_intensity, str):
            intensity_map = {'low': 0.3, 'medium': 0.5, 'high': 0.8}
            curr_intensity = intensity_map.get(curr_intensity, 0.5)
        if isinstance(next_intensity, str):
            intensity_map = {'low': 0.3, 'medium': 0.5, 'high': 0.8}
            next_intensity = intensity_map.get(next_intensity, 0.5)
        
        intensity_change = abs(next_intensity - curr_intensity)
        
        # Rule-based selection based on pacing
        if self.pacing in ['fast_throughout', 'quick_and_punchy']:
            # Fast pacing: prefer cuts (no transition) or very short fades
            if intensity_change > 0.3:
                return 'fade'  # Quick fade for big changes
            else:
                return 'fade'  # Standard fade
        
        elif self.pacing in ['slow_build_to_fast', 'slow_build_with_shocks']:
            # Build tension: use dissolves and fades
            if intensity_change > 0.4:
                return 'dissolve'  # Dramatic change
            else:
                return 'fade'
        
        elif self.pacing in ['measured', 'flowing']:
            # Smooth pacing: varied transitions
            if intensity_change > 0.3:
                return 'wipeleft'  # Visual variety
            else:
                return 'fade'
        
        else:
            # Default: simple fade
            return 'fade'
    
    def _get_shot_analysis(self, shot_id: int, shot_metadata: List[Dict]) -> Dict:
        """
        Get analysis for specific shot.
        
        Args:
            shot_id: Shot ID to find
            shot_metadata: List of all shot metadata
            
        Returns:
            Shot analysis dictionary
        """
        for shot in shot_metadata:
            if shot.get('id') == shot_id:
                return shot.get('analysis', {})
        return {}
    
    def _get_transition_duration(self) -> float:
        """
        Get transition duration based on pacing.
        
        Returns:
            Duration in seconds
        """
        if self.pacing in ['fast_throughout', 'quick_and_punchy']:
            return 0.3
        elif self.pacing in ['slow_build_to_fast', 'slow_build_with_shocks']:
            return 0.5
        else:
            return 0.4
    
    def _calculate_transition_offset(self, current_shot: Dict, next_shot: Dict) -> float:
        """
        Calculate offset for transition timing.
        
        Args:
            current_shot: Current shot data
            next_shot: Next shot data
            
        Returns:
            Offset in seconds
        """
        # Transition should start slightly before shot ends
        duration = self._get_transition_duration()
        return max(0, current_shot.get('duration', 2) - duration)
