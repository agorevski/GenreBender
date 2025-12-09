"""
Timeline construction module for building shot-level trailer sequences.

Takes beat-matched scenes and constructs a precise timeline with:
- Pacing rules (cold open, rising action, montage, stinger)
- Shot duration assignments
- Transition specifications
- Voiceover timing
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TimelineConstructor:
    """Build deterministic shot-level timelines from beat-scene matches."""
    
    def __init__(
        self,
        target_duration: int = 90,
        pacing_rules: Optional[Dict] = None
    ):
        """
        Initialize timeline constructor.
        
        Args:
            target_duration: Target trailer length in seconds
            pacing_rules: Optional custom pacing rules
        """
        self.target_duration = target_duration
        
        # Default pacing rules by beat type
        self.pacing_rules = pacing_rules or {
            'cold_open': {'duration': 3.0, 'shots': 1},
            'hook': {'duration': 5.0, 'shots': 2},
            'setup': {'duration': 8.0, 'shots': 3},
            'conflict_intro': {'duration': 6.0, 'shots': 2},
            'rising_action': {'duration': 10.0, 'shots': 4},
            'montage': {'duration': 15.0, 'shots': 8},
            'escalation': {'duration': 12.0, 'shots': 5},
            'climax_tease': {'duration': 8.0, 'shots': 3},
            'climax': {'duration': 6.0, 'shots': 2},
            'final_beat': {'duration': 4.0, 'shots': 2},
            'stinger': {'duration': 2.0, 'shots': 1}
        }
    
    def construct_timeline(
        self,
        beats: List[Dict],
        selected_scenes: Dict[str, List[Dict]],
        genre: str = 'thriller'
    ) -> Dict:
        """
        Construct complete trailer timeline from beats and scenes.
        
        Args:
            beats: List of beat dictionaries
            selected_scenes: Dict mapping beat_id to candidate scenes
            genre: Target genre for pacing adjustments
        
        Returns:
            Timeline dictionary with shots, transitions, and metadata
        """
        logger.info(f"Constructing timeline for {len(beats)} beats...")
        
        # Step 1: Allocate time budget per beat
        beat_budgets = self._allocate_time_budget(beats)
        
        # Step 2: Select shots for each beat
        timeline_shots = []
        current_time = 0.0
        
        for i, beat in enumerate(beats):
            beat_id = beat['id']
            budget = beat_budgets[beat_id]
            candidates = selected_scenes.get(beat_id, [])
            
            if not candidates:
                logger.warning(f"No candidates for beat {beat_id}, skipping")
                continue
            
            # Select and time shots for this beat
            beat_shots = self._select_shots_for_beat(
                beat=beat,
                candidates=candidates,
                budget=budget,
                start_time=current_time,
                beat_index=i
            )
            
            timeline_shots.extend(beat_shots)
            current_time += budget['duration']
        
        # Step 3: Add transitions
        timeline_shots = self._add_transitions(timeline_shots, genre)
        
        # Step 4: Build final timeline structure
        timeline = {
            'target_genre': genre,
            'target_duration': self.target_duration,
            'actual_duration': current_time,
            'total_shots': len(timeline_shots),
            'shots': timeline_shots,
            'metadata': {
                'beat_count': len(beats),
                'pacing_profile': self._get_pacing_profile(timeline_shots)
            }
        }
        
        logger.info(f"Timeline constructed: {len(timeline_shots)} shots, {current_time:.1f}s")
        return timeline
    
    def _allocate_time_budget(self, beats: List[Dict]) -> Dict[str, Dict]:
        """
        Allocate time budget to each beat based on position and type.
        
        Uses standard trailer structure:
        - First 10%: Hook (fast)
        - 10-40%: Setup (moderate)
        - 40-70%: Escalation (fast)
        - 70-90%: Climax (intense, quick cuts)
        - 90-100%: Stinger (very fast)
        """
        total_beats = len(beats)
        budgets = {}
        
        # Base duration per beat
        base_duration = self.target_duration / total_beats
        
        for i, beat in enumerate(beats):
            beat_id = beat['id']
            position = i / total_beats  # 0.0 to 1.0
            
            # Adjust duration based on position
            if position < 0.1:  # Opening (10%)
                duration_factor = 0.8  # Shorter, punchy
                shot_count = 1
            elif position < 0.4:  # Setup (30%)
                duration_factor = 1.2  # Longer, establish
                shot_count = 3
            elif position < 0.7:  # Escalation (30%)
                duration_factor = 1.5  # Montage, quick cuts
                shot_count = 5
            elif position < 0.9:  # Climax (20%)
                duration_factor = 1.0  # Intense, varied
                shot_count = 3
            else:  # Stinger (10%)
                duration_factor = 0.6  # Very short
                shot_count = 1
            
            budgets[beat_id] = {
                'duration': base_duration * duration_factor,
                'shot_count': shot_count,
                'position': position
            }
        
        # Normalize to hit target duration
        total_allocated = sum(b['duration'] for b in budgets.values())
        scale_factor = self.target_duration / total_allocated
        
        for beat_id in budgets:
            budgets[beat_id]['duration'] *= scale_factor
        
        return budgets
    
    def _select_shots_for_beat(
        self,
        beat: Dict,
        candidates: List[Dict],
        budget: Dict,
        start_time: float,
        beat_index: int
    ) -> List[Dict]:
        """
        Select and time shots for a single beat.
        
        Strategies:
        - Cold open: 1 establishing shot
        - Setup: 2-3 longer shots
        - Montage: 6-10 quick cuts
        - Climax: 2-3 intense shots
        - Stinger: 1 punchy shot
        """
        shots = []
        shot_count = budget['shot_count']
        total_duration = budget['duration']
        
        # Ensure we have enough candidates
        if len(candidates) < shot_count:
            logger.warning(
                f"Beat {beat['id']}: Only {len(candidates)} candidates "
                f"for {shot_count} shots, using available"
            )
            shot_count = len(candidates)
        
        # Select top shots by score
        selected = candidates[:shot_count]
        
        # Allocate time per shot
        shot_durations = self._allocate_shot_durations(
            shot_count=shot_count,
            total_duration=total_duration,
            position=budget['position']
        )
        
        # Build shot entries
        current_time = start_time
        for i, (candidate, duration) in enumerate(zip(selected, shot_durations)):
            # Calculate actual shot timing
            source_duration = candidate['duration']
            
            # Trim or use full shot
            if source_duration <= duration:
                # Use full shot
                use_start = 0.0
                use_end = source_duration
                actual_duration = source_duration
            else:
                # Trim to fit duration (use middle portion)
                trim_amount = source_duration - duration
                use_start = trim_amount / 2
                use_end = source_duration - (trim_amount / 2)
                actual_duration = duration
            
            shot_entry = {
                'shot_id': candidate['shot_id'],
                'beat_id': beat['id'],
                'beat_index': beat_index,
                'timeline_start': current_time,
                'timeline_end': current_time + actual_duration,
                'timeline_duration': actual_duration,
                'source_shot_start': candidate['start_time'],
                'source_shot_end': candidate['end_time'],
                'use_start_offset': use_start,
                'use_end_offset': use_end,
                'use_duration': actual_duration,
                'caption': candidate.get('caption', ''),
                'score': candidate.get('score', 0.0),
                'shot_path': candidate.get('shot_path', ''),
                'voiceover': beat.get('voiceover') if i == 0 else None  # VO on first shot
            }
            
            shots.append(shot_entry)
            current_time += actual_duration
        
        return shots
    
    def _allocate_shot_durations(
        self,
        shot_count: int,
        total_duration: float,
        position: float
    ) -> List[float]:
        """
        Allocate duration to individual shots within a beat.
        
        Varies by position:
        - Opening: Longer establishing shots
        - Middle: Varied pacing
        - Climax: Quick cuts
        - Stinger: Very short
        """
        durations = []
        
        if position < 0.1:  # Opening
            # Single establishing shot
            durations = [total_duration]
        
        elif position < 0.4:  # Setup
            # Descending durations (establish then tighten)
            base = total_duration / shot_count
            for i in range(shot_count):
                factor = 1.3 - (i * 0.2)  # 1.3, 1.1, 0.9, 0.7...
                durations.append(base * factor)
        
        elif position < 0.7:  # Montage/Escalation
            # Quick, uniform cuts
            durations = [total_duration / shot_count] * shot_count
        
        elif position < 0.9:  # Climax
            # Accelerating (getting faster)
            base = total_duration / shot_count
            for i in range(shot_count):
                factor = 1.2 - (i * 0.3)  # Gets shorter
                durations.append(base * factor)
        
        else:  # Stinger
            # Single punchy shot
            durations = [total_duration]
        
        # Normalize to exact total
        current_total = sum(durations)
        durations = [d * (total_duration / current_total) for d in durations]
        
        return durations
    
    def _add_transitions(
        self,
        shots: List[Dict],
        genre: str
    ) -> List[Dict]:
        """
        Add transition specifications between shots.
        
        Transition types:
        - cut: Hard cut (default)
        - dissolve: Cross-dissolve (0.5s)
        - fade_black: Fade to black (1.0s)
        - smash_cut: Abrupt cut with audio spike
        """
        genre_transitions = {
            'thriller': {
                'default': 'cut',
                'emphasis': 'smash_cut',
                'reset': 'fade_black'
            },
            'action': {
                'default': 'cut',
                'emphasis': 'smash_cut',
                'reset': 'cut'
            },
            'drama': {
                'default': 'dissolve',
                'emphasis': 'cut',
                'reset': 'fade_black'
            },
            'horror': {
                'default': 'cut',
                'emphasis': 'smash_cut',
                'reset': 'fade_black'
            },
            'scifi': {
                'default': 'dissolve',
                'emphasis': 'cut',
                'reset': 'dissolve'
            },
            'comedy': {
                'default': 'cut',
                'emphasis': 'smash_cut',
                'reset': 'cut'
            },
            'romance': {
                'default': 'dissolve',
                'emphasis': 'dissolve',
                'reset': 'fade_black'
            }
        }
        
        transitions = genre_transitions.get(genre, genre_transitions['thriller'])
        
        for i, shot in enumerate(shots):
            if i == len(shots) - 1:
                # Last shot: no transition
                shot['transition_out'] = None
            elif shot['beat_index'] != shots[i+1]['beat_index']:
                # Beat boundary: emphasis transition
                shot['transition_out'] = transitions['emphasis']
            else:
                # Within beat: default transition
                shot['transition_out'] = transitions['default']
        
        return shots
    
    def _get_pacing_profile(self, shots: List[Dict]) -> Dict:
        """Calculate pacing statistics for the timeline."""
        durations = [s['timeline_duration'] for s in shots]
        
        return {
            'avg_shot_duration': np.mean(durations),
            'min_shot_duration': np.min(durations),
            'max_shot_duration': np.max(durations),
            'total_shots': len(shots),
            'shots_per_minute': len(shots) / (sum(durations) / 60.0)
        }

def construct_timeline(
    selected_scenes_path: Path,
    output_path: Path,
    target_duration: int = 90,
    genre: str = 'thriller'
) -> Dict:
    """
    Main function for timeline construction.
    
    Args:
        selected_scenes_path: Path to selected_scenes.json
        output_path: Path to save trailer_timeline.json
        target_duration: Target trailer length in seconds
        genre: Target genre
    
    Returns:
        Timeline dictionary
    """
    # Load selected scenes
    with open(selected_scenes_path) as f:
        data = json.load(f)
        beats = data['beats']
        selected_scenes = data['selected_scenes']
    
    # Initialize constructor
    constructor = TimelineConstructor(target_duration=target_duration)
    
    # Build timeline
    timeline = constructor.construct_timeline(
        beats=beats,
        selected_scenes=selected_scenes,
        genre=genre
    )
    
    # Save timeline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(timeline, f, indent=2)
    
    logger.info(f"Saved timeline to {output_path}")
    logger.info(f"Timeline: {timeline['total_shots']} shots, {timeline['actual_duration']:.1f}s")
    
    return timeline
