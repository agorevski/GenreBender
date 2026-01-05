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
        """Initialize video assembler.
        
        Args:
            config: Global configuration dictionary containing video settings.
            genre_profile: Genre-specific configuration with color grading options.
            output_dir: Base output directory for assembled videos.
            enable_color_grading: Whether to apply color grading. Defaults to True.
            enable_transitions: Whether to add transitions between shots. Defaults to True.
        """
        self.config = config
        self.genre_profile = genre_profile
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / 'temp'
        self.enable_color_grading = enable_color_grading
        self.enable_transitions = enable_transitions
        
        # Video settings from config
        self.video_config = config.get('video', {})
        self.resolution = self.video_config.get('resolution')
        self.fps = self.video_config.get('fps')
        self.codec = self.video_config.get('codec')
        self.bitrate = self.video_config.get('bitrate')
        self.preset = self.video_config.get('preset')
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def assemble_video(self, timeline: Dict, shots_dir: Path, 
                      output_path: Path, shot_metadata: List[Dict] = None,
                      azure_client = None) -> str:
        """Orchestrate the main video assembly process.
        
        Coordinates title generation, transition selection, and video assembly
        using either simple concatenation or complex filter-based transitions.
        
        Args:
            timeline: Timeline dictionary with shot sequence and durations.
            shots_dir: Directory containing shot video files.
            output_path: Path for the output video file.
            shot_metadata: Full shot metadata needed for transition selection.
                Defaults to None.
            azure_client: Azure OpenAI client for AI-powered features.
                Defaults to None.
            
        Returns:
            str: Path to the assembled video file.
            
        Raises:
            FileNotFoundError: If required shot files are missing.
            ValueError: If timeline contains no shots or missing durations.
            subprocess.CalledProcessError: If FFmpeg execution fails.
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
        """Validate timeline structure and verify shot files exist.
        
        Args:
            timeline: Timeline dictionary containing shot sequence.
            shots_dir: Directory containing shot video files.
            
        Raises:
            ValueError: If timeline contains no shots or shots missing durations.
            FileNotFoundError: If required shot video files are missing.
        """
        timeline_shots = timeline.get('shots', [])
        if not timeline_shots:
            raise ValueError("Timeline contains no shots")
        
        missing_shots = []
        missing_durations = []
        for shot_data in timeline_shots:
            shot_id = shot_data.get('shot_id')
            shot_file = shots_dir / f"shot_{shot_id:04d}.mp4"
            if not shot_file.exists():
                missing_shots.append(shot_id)
            # Check for either 'duration' or 'timeline_duration'
            if 'duration' not in shot_data and 'timeline_duration' not in shot_data:
                missing_durations.append(shot_id)
        
        if missing_shots:
            raise FileNotFoundError(f"Missing shot files: {missing_shots}")
        if missing_durations:
            raise ValueError(f"Timeline missing duration for shots: {missing_durations}")
        
        logger.info(f"Validated {len(timeline_shots)} shots")
    
    def _assemble_simple(self, timeline: Dict, shots_dir: Path, 
                        output_path: Path, titles: List[Dict]) -> Path:
        """Assemble video using simple concatenation with color grading.
        
        Uses FFmpeg filter_complex to trim each shot to its timeline duration
        and apply color grading before concatenating.
        
        Args:
            timeline: Timeline dictionary containing shot sequence and durations.
            shots_dir: Directory containing shot video files.
            output_path: Path for the output video file.
            titles: Title cards to overlay. Not implemented in simple mode.
            
        Returns:
            Path: Path to the output video file.
            
        Raises:
            subprocess.CalledProcessError: If FFmpeg execution fails.
        """
        logger.info("Using simple concatenation method with per-shot trim...")
        
        timeline_shots = timeline.get('shots', [])
        
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
            duration = shot_data.get('timeline_duration') or shot_data.get('duration')
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
        """Assemble video with transitions using FFmpeg filter_complex.
        
        Applies color grading, trims shots to timeline durations, and uses
        xfade filters to create smooth transitions between consecutive shots.
        
        Args:
            timeline: Timeline dictionary containing shot sequence and durations.
            shots_dir: Directory containing shot video files.
            output_path: Path for the output video file.
            transitions: List of transition specifications with type, duration,
                and offset for each transition.
            titles: Title cards to overlay on the video.
            
        Returns:
            Path: Path to the output video file.
            
        Raises:
            subprocess.CalledProcessError: If FFmpeg execution fails. Filter
                complex is saved to temp directory for debugging.
        """
        logger.info("Using complex filter with transitions...")
        
        timeline_shots = timeline.get('shots', [])
        
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
        """Build FFmpeg filter_complex string for transitions and color grading.
        
        Constructs a complete filter_complex string that applies color grading,
        trims each shot to its timeline duration, and chains xfade transitions
        between consecutive shots.
        
        Args:
            timeline_shots: List of shot dictionaries from the timeline,
                each containing shot_id and duration information.
            shots_dir: Directory containing the shot video files.
            transitions: List of transition specifications, each containing
                type (e.g., 'fade', 'wipe'), duration, and offset values.
            
        Returns:
            str: Complete FFmpeg filter_complex string ready for execution.
        """
        filters = []
        
        # Step 1: Apply color grading (if enabled) and trim each input to timeline duration
        color_filter = self.genre_profile.get('color_grade', {}).get('filter', '') if self.enable_color_grading else ''
        for i, shot_data in enumerate(timeline_shots):
            duration = shot_data.get('timeline_duration') or shot_data.get('duration')
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
                    first_shot_dur = timeline_shots[0].get('timeline_duration') or timeline_shots[0].get('duration', 2)
                    transition_offset = first_shot_dur - trans_duration
                else:
                    # Subsequent transitions
                    prev_total = sum(timeline_shots[j].get('timeline_duration') or timeline_shots[j].get('duration', 2) for j in range(next_idx))
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
        """Get color grading filter string from genre profile.
        
        Returns:
            str: FFmpeg filter string for color grading, or empty string
                if no color grading is configured.
        """
        return self.genre_profile.get('color_grade', {}).get('filter', '')
