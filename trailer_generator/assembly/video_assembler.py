"""
Video assembly engine for trailer generation.
Handles video concatenation, color grading, transitions, and title overlays using FFmpeg.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from .title_generator import TitleGenerator
from .transition_selector import TransitionSelector

logger = logging.getLogger(__name__)

class VideoAssembler:
    """
    Assembles final trailer video from timeline using FFmpeg.
    """
    
    def __init__(self, config: Dict, genre_profile: Dict, output_dir: Path,
                 enable_color_grading: bool = True, enable_transitions: bool = True):
        """
        Initialize video assembler.
        
        Args:
            config: Global configuration dictionary
            genre_profile: Genre-specific configuration
            output_dir: Base output directory
            enable_color_grading: Whether to apply color grading
            enable_transitions: Whether to add transitions between shots
        """
        self.config = config
        self.genre_profile = genre_profile
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / 'temp'
        self.enable_color_grading = enable_color_grading
        self.enable_transitions = enable_transitions
        
        # Video settings from config
        self.video_config = config.get('video', {})
        self.resolution = self.video_config.get('resolution', '1920x1080')
        self.fps = self.video_config.get('fps', 24)
        self.codec = self.video_config.get('codec', 'libx264')
        self.bitrate = self.video_config.get('bitrate', '5000k')
        self.preset = self.video_config.get('preset', 'medium')
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def assemble_video(self, timeline: Dict, shots_dir: Path, 
                      output_path: Path, shot_metadata: List[Dict] = None,
                      azure_client = None) -> str:
        """
        Main video assembly orchestrator.
        
        Args:
            timeline: Timeline dictionary with shot sequence
            shots_dir: Directory containing shot video files
            output_path: Path for output video
            shot_metadata: Full shot metadata (needed for transitions)
            azure_client: Azure OpenAI client for AI features (optional)
            
        Returns:
            Path to assembled video
        """
        logger.info("Starting video assembly...")
        
        # Step 1: Validate inputs
        self._validate_timeline(timeline, shots_dir)
        
        # Step 2: Generate title cards (if AI enabled)
        titles = []
        if azure_client and self.video_config.get('ai_title_generation', False):
            try:
                title_gen = TitleGenerator(azure_client, self.genre_profile.get('genre', 'thriller'))
                titles = title_gen.generate_titles(timeline)
                logger.info(f"Generated {len(titles)} title cards")
            except Exception as e:
                logger.warning(f"Title generation failed: {e}")
        
        # Step 3: Select transitions (if enabled)
        transitions = []
        if self.enable_transitions and shot_metadata:
            try:
                transition_selector = TransitionSelector(
                    azure_client, 
                    self.genre_profile,
                    enable_ai=self.video_config.get('ai_transition_selection', False)
                )
                transitions = transition_selector.select_transitions(timeline, shot_metadata)
                logger.info(f"Selected {len(transitions)} transitions")
            except Exception as e:
                logger.warning(f"Transition selection failed: {e}")
                transitions = []
        
        # Step 4: Build video based on complexity
        if self.enable_transitions and transitions:
            # Complex: Use filter_complex for transitions and color grading
            output = self._assemble_with_transitions(
                timeline, shots_dir, output_path, transitions, titles
            )
        else:
            # Simple: Concatenate with color grading only
            output = self._assemble_simple(
                timeline, shots_dir, output_path, titles
            )
        
        logger.info(f"Video assembly complete: {output}")
        return str(output)
    
    def _validate_timeline(self, timeline: Dict, shots_dir: Path):
        """
        Validate timeline and shot files exist.
        
        Args:
            timeline: Timeline dictionary
            shots_dir: Directory containing shots
            
        Raises:
            FileNotFoundError: If shot files missing
        """
        timeline_shots = timeline.get('timeline', [])
        if not timeline_shots:
            raise ValueError("Timeline contains no shots")
        
        missing_shots = []
        missing_durations = []
        for shot_data in timeline_shots:
            shot_id = shot_data.get('shot_id')
            shot_file = shots_dir / f"shot_{shot_id:04d}.mp4"
            if not shot_file.exists():
                missing_shots.append(shot_id)
            if 'duration' not in shot_data:
                missing_durations.append(shot_id)
        
        if missing_shots:
            raise FileNotFoundError(f"Missing shot files: {missing_shots}")
        if missing_durations:
            raise ValueError(f"Timeline missing duration for shots: {missing_durations}")
        
        logger.info(f"Validated {len(timeline_shots)} shots")
    
    def _assemble_simple(self, timeline: Dict, shots_dir: Path, 
                        output_path: Path, titles: List[Dict]) -> Path:
        """
        Simple assembly: concatenate shots with color grading.
        
        Args:
            timeline: Timeline dictionary
            shots_dir: Directory containing shots
            output_path: Output path
            titles: Title cards (not implemented in simple mode)
            
        Returns:
            Path to output video
        """
        logger.info("Using simple concatenation method with per-shot trim...")
        
        timeline_shots = timeline.get('timeline', [])
        
        # Build input list
        inputs = []
        for shot_data in timeline_shots:
            shot_id = shot_data.get('shot_id')
            shot_file = shots_dir / f"shot_{shot_id:04d}.mp4"
            inputs.extend(['-i', str(shot_file)])
        
        # Build filter_complex to trim video/audio to timeline durations
        filters = []
        video_labels = []
        audio_labels = []
        color_filter = self.genre_profile.get('color_grade', {}).get('filter', '') if self.enable_color_grading else ''
        
        for idx, shot_data in enumerate(timeline_shots):
            duration = shot_data.get('duration')
            vf_chain = color_filter if color_filter else 'null'
            filters.append(
                f"[{idx}:v]{vf_chain},trim=duration={duration},setpts=PTS-STARTPTS[v{idx}]"
            )
            filters.append(
                f"[{idx}:a]atrim=duration={duration},asetpts=PTS-STARTPTS[a{idx}]"
            )
            video_labels.append(f"v{idx}")
            audio_labels.append(f"a{idx}")
        
        pair_labels = ''.join([f"[{v}][{a}]" for v, a in zip(video_labels, audio_labels)])
        filters.append(
            f"{pair_labels}concat=n={len(timeline_shots)}:v=1:a=1[outv][outa]"
        )
        
        filter_complex = ';'.join(filters)
        
        cmd = ['ffmpeg']
        cmd.extend(inputs)
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', self.codec,
            '-preset', self.preset,
            '-b:v', self.bitrate,
            '-c:a', 'aac',
            '-b:a', '192k',
            '-r', str(self.fps),
            '-s', self.resolution,
            '-y',
            str(output_path)
        ])
        
        logger.info("Executing FFmpeg filter_complex for simple concat...")
        logger.debug(f"Filter complex: {filter_complex}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            logger.info("FFmpeg concatenation with trim successful")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise
        
        return output_path
    
    def _assemble_with_transitions(self, timeline: Dict, shots_dir: Path,
                                   output_path: Path, transitions: List[Dict],
                                   titles: List[Dict]) -> Path:
        """
        Complex assembly: shots with transitions using filter_complex.
        
        Args:
            timeline: Timeline dictionary
            shots_dir: Directory containing shots
            output_path: Output path
            transitions: List of transition specifications
            titles: Title cards to overlay
            
        Returns:
            Path to output video
        """
        logger.info("Using complex filter with transitions...")
        
        timeline_shots = timeline.get('timeline', [])
        
        # Build filter_complex string
        filter_complex = self._build_filter_complex(
            timeline_shots, shots_dir, transitions
        )
        
        # Build input list
        inputs = []
        for shot_data in timeline_shots:
            shot_id = shot_data.get('shot_id')
            shot_file = shots_dir / f"shot_{shot_id:04d}.mp4"
            inputs.extend(['-i', str(shot_file)])
        
        # Build FFmpeg command
        cmd = ['ffmpeg']
        cmd.extend(inputs)
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '[outa]',  # Map audio output
            '-c:v', self.codec,
            '-preset', self.preset,
            '-b:v', self.bitrate,
            '-c:a', 'aac',  # Encode audio as AAC
            '-b:a', '192k',  # Audio bitrate
            '-r', str(self.fps),
            '-s', self.resolution,
            '-y',
            str(output_path)
        ])
        
        # Execute FFmpeg
        logger.info("Executing FFmpeg with filter_complex...")
        logger.debug(f"Filter complex: {filter_complex}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("FFmpeg filter_complex execution successful")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            # Save filter complex for debugging
            debug_file = self.temp_dir / 'filter_complex_debug.txt'
            with open(debug_file, 'w') as f:
                f.write(filter_complex)
            logger.error(f"Filter complex saved to: {debug_file}")
            raise
        
        return output_path
    
    def _build_filter_complex(self, timeline_shots: List[Dict], 
                             shots_dir: Path, transitions: List[Dict]) -> str:
        """
        Build FFmpeg filter_complex string for transitions and color grading.
        
        Args:
            timeline_shots: List of shots in timeline
            shots_dir: Directory containing shots
            transitions: List of transition specifications
            
        Returns:
            filter_complex string
        """
        filters = []
        
        # Step 1: Apply color grading (if enabled) and trim each input to timeline duration
        color_filter = self.genre_profile.get('color_grade', {}).get('filter', '') if self.enable_color_grading else ''
        for i, shot_data in enumerate(timeline_shots):
            duration = shot_data.get('duration')
            vf_chain = color_filter if color_filter else 'null'
            filters.append(f"[{i}:v]{vf_chain},trim=duration={duration},setpts=PTS-STARTPTS[v{i}]")
            filters.append(f"[{i}:a]atrim=duration={duration},asetpts=PTS-STARTPTS[a{i}]")
        
        # Step 2: Apply transitions between consecutive shots
        if not transitions:
            # No transitions: just concatenate trimmed video and audio
            pair_labels = ''.join([f"[v{i}][a{i}]" for i in range(len(timeline_shots))])
            filters.append(f"{pair_labels}concat=n={len(timeline_shots)}:v=1:a=1[outv][outa]")
        else:
            # Build xfade filter chain for video
            current_label = "v0"
            
            for i, trans in enumerate(transitions):
                next_idx = i + 1
                trans_type = trans.get('type', 'fade')
                trans_duration = trans.get('duration', 0.5)
                offset = trans.get('offset', 0)
                
                # Calculate offset (when transition starts)
                # offset = cumulative duration of all previous shots - transition duration
                if i == 0:
                    # First transition
                    first_shot_dur = timeline_shots[0].get('duration', 2)
                    transition_offset = first_shot_dur - trans_duration
                else:
                    # Subsequent transitions
                    prev_total = sum(timeline_shots[j].get('duration', 2) for j in range(next_idx))
                    transition_offset = prev_total - trans_duration
                
                output_label = f"v{next_idx}_{i}" if i < len(transitions) - 1 else "outv"
                
                # xfade filter
                filters.append(
                    f"[{current_label}][v{next_idx}]xfade=transition={trans_type}:"
                    f"duration={trans_duration}:offset={transition_offset}[{output_label}]"
                )
                
                current_label = output_label
            
            # Concatenate trimmed audio streams (no transitions for audio)
            audio_labels = ''.join([f"[a{i}]" for i in range(len(timeline_shots))])
            filters.append(f"{audio_labels}concat=n={len(timeline_shots)}:v=0:a=1[outa]")
        
        # Combine all filters
        filter_string = ';'.join(filters)
        
        return filter_string
    
    def _get_color_grade_filter(self) -> str:
        """
        Get color grading filter from genre profile.
        
        Returns:
            FFmpeg filter string
        """
        return self.genre_profile.get('color_grade', {}).get('filter', '')
