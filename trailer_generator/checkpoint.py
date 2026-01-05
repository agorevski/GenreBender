"""
Checkpoint management for pipeline resumption.
Allows resuming from any stage of the trailer generation pipeline.
Supports both genre-agnostic stages (shared) and genre-dependent stages (per-genre).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages pipeline checkpoints to enable resuming from any stage.
    Supports genre-agnostic stages (run once) and genre-dependent stages (per-genre).
    """
    
    # Genre-agnostic stages (run once, shared across all genre outputs)
    GENRE_AGNOSTIC_STAGES = [
        'shot_detection',
        'keyframe_extraction',
        'audio_extraction',
        'subtitle_management',
        'remote_analysis',
        'story_graph_generation',
    ]
    
    # Genre-dependent stages (run once per genre)
    GENRE_DEPENDENT_STAGES = [
        'beat_sheet_generation',
        'embedding_generation',
        'scene_retrieval',
        'timeline_construction',
        'video_assembly',
        'audio_mixing',
    ]
    
    # Legacy stages (for backward compatibility with old pipelines)
    LEGACY_STAGES = [
        'genre_scoring',
        'shot_selection',
        'narrative_generation',
    ]
    
    # All stages combined (for backward compatibility)
    STAGES = GENRE_AGNOSTIC_STAGES + LEGACY_STAGES + GENRE_DEPENDENT_STAGES
    
    def __init__(self, checkpoint_path: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to the checkpoint file for storing pipeline state.
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load checkpoint from disk.

        Returns:
            Dict: The checkpoint data loaded from disk, or an empty checkpoint
                structure if loading fails or file doesn't exist.
        """
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
                # Migrate old checkpoint format if needed
                data = self._migrate_checkpoint(data)
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return self._create_empty()
        return self._create_empty()
    
    def _migrate_checkpoint(self, data: Dict) -> Dict:
        """Migrate old checkpoint format to new format with per-genre support.

        Args:
            data: The checkpoint data dictionary to migrate.

        Returns:
            Dict: The migrated checkpoint data with updated version and structure.
        """
        version = data.get('version', '1.0')
        
        if version == '1.0':
            # Add genre_stages section if not present
            if 'genre_stages' not in data:
                data['genre_stages'] = {}
            data['version'] = '2.0'
            logger.info("Migrated checkpoint from v1.0 to v2.0")
        
        return data
    
    def _create_empty(self) -> Dict:
        """Create empty checkpoint structure.

        Returns:
            Dict: A new empty checkpoint dictionary with initialized fields
                including version, timestamps, and empty stage tracking.
        """
        return {
            'version': '2.0',
            'created_at': datetime.now().isoformat(),
            'last_completed_stage': None,
            'input_file': None,
            'genre': None,  # Primary genre (for backward compatibility)
            'stages': {stage: {'completed': False} for stage in self.STAGES},
            'genre_stages': {}  # Per-genre completion tracking
        }
    
    def save(self):
        """Save checkpoint to disk.

        Updates the 'updated_at' timestamp and writes the checkpoint data
        to the configured checkpoint path as JSON.
        """
        try:
            self.data['updated_at'] = datetime.now().isoformat()
            with open(self.checkpoint_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.debug(f"Saved checkpoint to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def reload(self):
        """Reload checkpoint from disk to get latest state.

        Useful in parallel execution scenarios where the checkpoint file
        may have been updated by another process.
        """
        self.data = self._load()
        logger.debug(f"Reloaded checkpoint from {self.checkpoint_path}")
    
    def mark_stage_completed(self, stage: str, metadata: Optional[Dict] = None, genre: Optional[str] = None):
        """Mark a stage as completed.

        Args:
            stage: Name of the stage to mark as completed.
            metadata: Optional metadata about stage completion to store.
            genre: Genre for genre-dependent stages. Use None for genre-agnostic stages.
        """
        # Determine if this is a genre-dependent stage
        is_genre_dependent = stage in self.GENRE_DEPENDENT_STAGES
        
        if is_genre_dependent and genre:
            # Per-genre completion
            self._mark_genre_stage_completed(stage, genre, metadata)
        else:
            # Genre-agnostic or legacy completion
            if stage not in self.STAGES:
                logger.warning(f"Unknown stage: {stage}")
                return
            
            self.data['stages'][stage]['completed'] = True
            self.data['stages'][stage]['timestamp'] = datetime.now().isoformat()
            
            if metadata:
                self.data['stages'][stage].update(metadata)
            
            self.data['last_completed_stage'] = stage
        
        self.save()
        
        if genre:
            logger.info(f"✓ Checkpoint: Completed stage '{stage}' for genre '{genre}'")
        else:
            logger.info(f"✓ Checkpoint: Completed stage '{stage}'")
    
    def _mark_genre_stage_completed(self, stage: str, genre: str, metadata: Optional[Dict] = None):
        """Mark a genre-dependent stage as completed for a specific genre.

        Args:
            stage: Name of the stage to mark as completed.
            genre: Target genre (will be lowercased).
            metadata: Optional metadata to store with the completion record.
        """
        genre = genre.lower()
        
        # Initialize genre section if needed
        if genre not in self.data['genre_stages']:
            self.data['genre_stages'][genre] = {}
        
        if stage not in self.data['genre_stages'][genre]:
            self.data['genre_stages'][genre][stage] = {}
        
        self.data['genre_stages'][genre][stage]['completed'] = True
        self.data['genre_stages'][genre][stage]['timestamp'] = datetime.now().isoformat()
        
        if metadata:
            self.data['genre_stages'][genre][stage].update(metadata)
    
    def is_stage_completed(self, stage: str, genre: Optional[str] = None) -> bool:
        """Check if a stage is completed.

        Args:
            stage: Name of the stage to check.
            genre: Genre for genre-dependent stages. Use None for genre-agnostic stages.

        Returns:
            bool: True if the stage is completed, False otherwise.
        """
        # Check if this is a genre-dependent stage
        is_genre_dependent = stage in self.GENRE_DEPENDENT_STAGES
        
        if is_genre_dependent and genre:
            return self._is_genre_stage_completed(stage, genre)
        
        # Genre-agnostic or legacy stage
        if stage not in self.STAGES:
            return False
        
        # Initialize stage if it doesn't exist in checkpoint data
        if stage not in self.data['stages']:
            self.data['stages'][stage] = {'completed': False}
        
        return self.data['stages'][stage].get('completed', False)
    
    def _is_genre_stage_completed(self, stage: str, genre: str) -> bool:
        """Check if a genre-dependent stage is completed for a specific genre.

        Args:
            stage: Name of the stage to check.
            genre: Target genre to check (will be lowercased).

        Returns:
            bool: True if the stage is completed for this genre, False otherwise.
        """
        genre = genre.lower()
        
        if genre not in self.data.get('genre_stages', {}):
            return False
        
        if stage not in self.data['genre_stages'][genre]:
            return False
        
        return self.data['genre_stages'][genre][stage].get('completed', False)
    
    def get_completed_genres(self, stage: str) -> List[str]:
        """Get list of genres that have completed a specific stage.

        Args:
            stage: Name of the stage to check.

        Returns:
            List[str]: List of genre names that have completed this stage.
        """
        completed = []
        for genre, stages in self.data.get('genre_stages', {}).items():
            if stage in stages and stages[stage].get('completed', False):
                completed.append(genre)
        return completed
    
    def get_incomplete_genres(self, stage: str, target_genres: List[str]) -> List[str]:
        """Get list of genres that have NOT completed a specific stage.

        Args:
            stage: Name of the stage to check.
            target_genres: List of genres to check against.

        Returns:
            List[str]: List of genre names that have NOT completed this stage.
        """
        completed = set(self.get_completed_genres(stage))
        return [g for g in target_genres if g.lower() not in completed]
    
    def get_last_completed_stage(self) -> Optional[str]:
        """Get the last completed stage.

        Returns:
            Optional[str]: The name of the last completed stage, or None if
                no stages have been completed.
        """
        return self.data.get('last_completed_stage')
    
    def get_resume_stage(self, requested_stage: Optional[str] = None) -> Optional[str]:
        """Determine which stage to resume from.

        If a specific stage is requested, validates and returns it. Otherwise,
        auto-detects the next uncompleted stage after the last completed one.

        Args:
            requested_stage: User-requested stage to resume from. If None,
                auto-detection is used.

        Returns:
            Optional[str]: Stage name to resume from, or None to start from beginning.
        """
        if requested_stage:
            if requested_stage not in self.STAGES:
                logger.error(f"Invalid stage: {requested_stage}")
                return None
            return requested_stage
        
        # Auto-detect: resume from next uncompleted stage
        last_completed = self.get_last_completed_stage()
        if not last_completed:
            return None  # Start from beginning
        
        # Find next stage after last completed
        try:
            last_idx = self.STAGES.index(last_completed)
            if last_idx + 1 < len(self.STAGES):
                next_stage = self.STAGES[last_idx + 1]
                logger.info(f"Auto-detected resume point: {next_stage}")
                return next_stage
        except ValueError:
            pass
        
        return None
    
    def should_skip_stage(self, stage: str, force: bool = False, genre: Optional[str] = None) -> bool:
        """Determine if a stage should be skipped.

        Args:
            stage: Name of the stage to check.
            force: If True, never skip (force re-run even if completed).
            genre: Genre for genre-dependent stages.

        Returns:
            bool: True if the stage should be skipped, False otherwise.
        """
        if force:
            return False
        return self.is_stage_completed(stage, genre)
    
    def validate_resume(self, resume_from: str, input_file: str, genre: Optional[str] = None) -> bool:
        """Validate that we can resume from the specified stage.

        Checks that the stage is valid, the input file matches (with warning if not),
        and all prerequisite stages have been completed.

        Args:
            resume_from: Name of the stage to resume from.
            input_file: Current input file path to validate against checkpoint.
            genre: Current genre (optional, for logging purposes).

        Returns:
            bool: True if resume is valid and can proceed, False otherwise.
        """
        if resume_from not in self.STAGES:
            logger.error(f"Invalid stage: {resume_from}")
            return False
        
        # Check if input file matches
        checkpoint_input = self.data.get('input_file')
        if checkpoint_input and checkpoint_input != input_file:
            logger.warning(f"Input file mismatch: checkpoint={checkpoint_input}, current={input_file}")
            logger.warning("Proceeding anyway, but results may be inconsistent")
        
        # Check if all prerequisite stages are completed
        resume_idx = self.STAGES.index(resume_from)
        for i in range(resume_idx):
            prereq_stage = self.STAGES[i]
            # Skip legacy stages in prerequisite check
            if prereq_stage in self.LEGACY_STAGES:
                continue
            if not self.is_stage_completed(prereq_stage):
                logger.error(f"Cannot resume from '{resume_from}': prerequisite stage '{prereq_stage}' not completed")
                return False
        
        logger.info(f"✓ Resume validation passed for stage '{resume_from}'")
        return True
    
    def set_metadata(self, input_file: str, genre: Optional[str] = None):
        """Set checkpoint metadata.

        Args:
            input_file: Path to the input video file.
            genre: Trailer genre (optional for genre-agnostic stages).
        """
        self.data['input_file'] = input_file
        if genre:
            self.data['genre'] = genre
        self.save()
    
    def reset(self):
        """Reset checkpoint to empty state.

        Clears all checkpoint data and saves a fresh empty checkpoint
        to disk.
        """
        self.data = self._create_empty()
        self.save()
        logger.info("Checkpoint reset")
    
    def reset_genre(self, genre: str):
        """Reset all stages for a specific genre.

        Removes all completion records for the specified genre and saves
        the checkpoint.

        Args:
            genre: Genre to reset (will be lowercased).
        """
        genre = genre.lower()
        if genre in self.data.get('genre_stages', {}):
            del self.data['genre_stages'][genre]
            self.save()
            logger.info(f"Reset checkpoint for genre '{genre}'")
    
    def get_stats(self, genre: Optional[str] = None) -> Dict:
        """Get checkpoint statistics.

        Args:
            genre: If provided, include genre-specific completion stats.

        Returns:
            Dict: Dictionary containing checkpoint statistics including:
                - total_stages: Total number of stages
                - completed_stages: Number of completed stages
                - progress_percent: Completion percentage
                - last_completed: Name of last completed stage
                - completed_list: List of completed stage names
                - agnostic_completed: List of completed genre-agnostic stages
                - genre_completed: List of completed genre-dependent stages
                - genre: The genre these stats are for
        """
        # Count genre-agnostic completed stages
        agnostic_completed = [s for s in self.GENRE_AGNOSTIC_STAGES if self.is_stage_completed(s)]
        
        # Count genre-dependent completed stages for the specified genre
        genre_completed = []
        if genre:
            genre_completed = [s for s in self.GENRE_DEPENDENT_STAGES 
                             if self.is_stage_completed(s, genre)]
        
        all_completed = agnostic_completed + genre_completed
        total = len(self.GENRE_AGNOSTIC_STAGES) + len(self.GENRE_DEPENDENT_STAGES)
        
        return {
            'total_stages': total,
            'completed_stages': len(all_completed),
            'progress_percent': (len(all_completed) / total) * 100 if total > 0 else 0,
            'last_completed': self.get_last_completed_stage(),
            'completed_list': all_completed,
            'agnostic_completed': agnostic_completed,
            'genre_completed': genre_completed if genre else [],
            'genre': genre
        }
    
    def get_all_genre_stats(self, target_genres: List[str]) -> Dict:
        """Get completion statistics for all target genres.

        Args:
            target_genres: List of genre names to check.

        Returns:
            Dict: Dictionary mapping genre name to completion stats containing:
                - completed_stages: Number of completed stages for this genre
                - total_stages: Total number of genre-dependent stages
                - completed_list: List of completed stage names
                - is_complete: True if all genre-dependent stages are complete
        """
        stats = {}
        for genre in target_genres:
            genre = genre.lower()
            completed = [s for s in self.GENRE_DEPENDENT_STAGES 
                        if self.is_stage_completed(s, genre)]
            stats[genre] = {
                'completed_stages': len(completed),
                'total_stages': len(self.GENRE_DEPENDENT_STAGES),
                'completed_list': completed,
                'is_complete': len(completed) == len(self.GENRE_DEPENDENT_STAGES)
            }
        return stats


def load_shots_from_metadata(metadata_path: Path) -> List[Dict]:
    """Load shots from saved metadata file.

    Args:
        metadata_path: Path to the shot_metadata.json file.

    Returns:
        List[Dict]: List of shot dictionaries, or empty list if loading fails.
    """
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return []
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        shots = metadata.get('shots', [])
        logger.info(f"Loaded {len(shots)} shots from {metadata_path}")
        return shots
    except Exception as e:
        logger.error(f"Failed to load shots metadata: {e}")
        return []


def save_shots_to_metadata(shots: List[Dict], metadata_path: Path, video_path: str):
    """Save shots to metadata file.

    Args:
        shots: List of shot dictionaries to save.
        metadata_path: Path to the shot_metadata.json file.
        video_path: Source video path to record in metadata.
    """
    metadata = {
        'source_video': video_path,
        'total_shots': len(shots),
        'shots': shots,
        'updated_at': datetime.now().isoformat()
    }
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved {len(shots)} shots to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save shots metadata: {e}")
