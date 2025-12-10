"""
Checkpoint management for pipeline resumption.
Allows resuming from any stage of the trailer generation pipeline.
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
    """
    
    STAGES = [
        'shot_detection',
        'keyframe_extraction',
        'audio_extraction',
        'subtitle_management',
        'remote_analysis',
        'genre_scoring',
        'shot_selection',
        'narrative_generation',
        'video_assembly',
        'audio_mixing',
        # Semantic pipeline stages (11-15)
        'story_graph_generation',
        'beat_sheet_generation',
        'embedding_generation',
        'scene_retrieval',
        'timeline_construction'
    ]
    
    def __init__(self, checkpoint_path: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load checkpoint from disk."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return self._create_empty()
        return self._create_empty()
    
    def _create_empty(self) -> Dict:
        """Create empty checkpoint structure."""
        return {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'last_completed_stage': None,
            'input_file': None,
            'genre': None,
            'stages': {stage: {'completed': False} for stage in self.STAGES}
        }
    
    def save(self):
        """Save checkpoint to disk."""
        try:
            self.data['updated_at'] = datetime.now().isoformat()
            with open(self.checkpoint_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.debug(f"Saved checkpoint to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def mark_stage_completed(self, stage: str, metadata: Optional[Dict] = None):
        """
        Mark a stage as completed.
        
        Args:
            stage: Stage name
            metadata: Optional metadata about stage completion
        """
        if stage not in self.STAGES:
            logger.warning(f"Unknown stage: {stage}")
            return
        
        self.data['stages'][stage]['completed'] = True
        self.data['stages'][stage]['timestamp'] = datetime.now().isoformat()
        
        if metadata:
            self.data['stages'][stage].update(metadata)
        
        self.data['last_completed_stage'] = stage
        self.save()
        
        logger.info(f"✓ Checkpoint: Completed stage '{stage}'")
    
    def is_stage_completed(self, stage: str) -> bool:
        """
        Check if a stage is completed.
        
        Args:
            stage: Stage name
            
        Returns:
            True if stage is completed
        """
        if stage not in self.STAGES:
            return False
        
        # Initialize stage if it doesn't exist in checkpoint data
        if stage not in self.data['stages']:
            self.data['stages'][stage] = {'completed': False}
        
        return self.data['stages'][stage].get('completed', False)
    
    def get_last_completed_stage(self) -> Optional[str]:
        """Get the last completed stage."""
        return self.data.get('last_completed_stage')
    
    def get_resume_stage(self, requested_stage: Optional[str] = None) -> Optional[str]:
        """
        Determine which stage to resume from.
        
        Args:
            requested_stage: User-requested stage to resume from
            
        Returns:
            Stage name to resume from, or None to start from beginning
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
    
    def should_skip_stage(self, stage: str, force: bool = False) -> bool:
        """
        Determine if a stage should be skipped.
        
        Args:
            stage: Stage name
            force: Force re-run even if completed
            
        Returns:
            True if stage should be skipped
        """
        if force:
            return False
        return self.is_stage_completed(stage)
    
    def validate_resume(self, resume_from: str, input_file: str, genre: str) -> bool:
        """
        Validate that we can resume from the specified stage.
        
        Args:
            resume_from: Stage to resume from
            input_file: Current input file
            genre: Current genre
            
        Returns:
            True if resume is valid
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
            if not self.is_stage_completed(prereq_stage):
                logger.error(f"Cannot resume from '{resume_from}': prerequisite stage '{prereq_stage}' not completed")
                return False
        
        logger.info(f"✓ Resume validation passed for stage '{resume_from}'")
        return True
    
    def set_metadata(self, input_file: str, genre: str):
        """
        Set checkpoint metadata.
        
        Args:
            input_file: Input video file
            genre: Trailer genre
        """
        self.data['input_file'] = input_file
        self.data['genre'] = genre
        self.save()
    
    def reset(self):
        """Reset checkpoint to empty state."""
        self.data = self._create_empty()
        self.save()
        logger.info("Checkpoint reset")
    
    def get_stats(self) -> Dict:
        """Get checkpoint statistics."""
        completed_stages = [s for s in self.STAGES if self.is_stage_completed(s)]
        return {
            'total_stages': len(self.STAGES),
            'completed_stages': len(completed_stages),
            'progress_percent': (len(completed_stages) / len(self.STAGES)) * 100,
            'last_completed': self.get_last_completed_stage(),
            'completed_list': completed_stages
        }


def load_shots_from_metadata(metadata_path: Path) -> List[Dict]:
    """
    Load shots from saved metadata file.
    
    Args:
        metadata_path: Path to shot_metadata.json
        
    Returns:
        List of shot dictionaries
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
    """
    Save shots to metadata file.
    
    Args:
        shots: List of shot dictionaries
        metadata_path: Path to shot_metadata.json
        video_path: Source video path
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
