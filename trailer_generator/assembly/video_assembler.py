"""
Video assembly engine for trailer generation.
Handles video concatenation, color grading, transitions, and title overlays using FFmpeg.
"""

import json
import logging
import math
import subprocess
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
        self.enable_color_grading = enable_color_grading
        self.enable_transitions = enable_transitions
        
        # Video settings from config
        self.video_config = config.get('video', {})
        self.resolution = self.video_config.get('resolution')
        self.fps = self.video_config.get('fps')
        self.codec = self.video_config.get('codec')
        self.bitrate = self.video_config.get('bitrate')
        self.preset = self.video_config.get('preset')
        
        self._media_info = {}
    
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
                genre = (self.genre_profile.get('name') or self.genre_profile.get('genre')
                         or timeline.get('target_genre', 'thriller'))
                title_gen = TitleGenerator(azure_client, genre)
                titles = title_gen.generate_titles(timeline)
                logger.info(f"Generated {len(titles)} title cards")
            except Exception as e:
                logger.warning(f"Title generation failed: {e}")
        
        # Step 3: Select transitions (if enabled)
        transitions = []
        if self.enable_transitions:
            transitions = self._timeline_transitions(timeline['shots'])
        if (self.enable_transitions and shot_metadata and azure_client
                and self.video_config.get('ai_transition_selection', False)):
            try:
                transition_selector = TransitionSelector(
                    azure_client, 
                    self.genre_profile,
                    enable_ai=self.video_config.get('ai_transition_selection', False)
                )
                # The selector still uses the legacy Stage 8 timeline schema.
                selector_timeline = {
                    **timeline,
                    'timeline': [
                        {**shot, 'duration': self._duration(shot)}
                        for shot in timeline['shots']
                    ]
                }
                transitions = transition_selector.select_transitions(selector_timeline, shot_metadata)
                logger.info(f"Selected {len(transitions)} transitions")
            except Exception as e:
                logger.warning(f"Transition selection failed: {e}")
        
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
        
        if not Path(output).is_file() or Path(output).stat().st_size == 0:
            raise ValueError(f"FFmpeg did not produce a non-empty video: {output}")
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
        timeline_shots = timeline.get('shots', []) if isinstance(timeline, dict) else []
        if not isinstance(timeline_shots, list) or not timeline_shots:
            raise ValueError("Timeline contains no shots")
        fps = self._number(self.fps, 'Video fps')
        if fps <= 0:
            raise ValueError("Video fps must be positive")

        current_time = 0.0
        seen_ids, seen_paths, seen_ranges = set(), set(), set()
        files = []
        for shot_data in timeline_shots:
            if not isinstance(shot_data, dict):
                raise ValueError("Timeline shots must be objects")
            shot_id = shot_data.get('shot_id')
            if isinstance(shot_id, bool) or not isinstance(shot_id, int) or shot_id < 0:
                raise ValueError("Shot IDs must be non-negative integers")
            duration = self._duration(shot_data)
            if duration < 1 / fps - 1e-9:
                raise ValueError(f"Shot {shot_id} is shorter than one output frame")
            offset = self._number(shot_data.get('use_start_offset', 0), 'source offset')
            if offset < 0:
                raise ValueError(f"Negative source offset for shot {shot_id}")
            expected = {
                'timeline_start': current_time,
                'timeline_end': current_time + duration,
                'use_end_offset': offset + duration,
                'use_duration': duration,
            }
            for key, value in expected.items():
                if key in shot_data and not math.isclose(
                        self._number(shot_data[key], key), value, abs_tol=1e-6):
                    raise ValueError(f"Inconsistent {key} for shot {shot_id}")
            available = self._source_duration(shot_data)
            if available is not None and offset + duration > available + 1e-6:
                raise ValueError(f"Source range exceeded for shot {shot_id}")
            shot_file = self._shot_file(shot_data, shots_dir).resolve()
            if not shot_file.is_file():
                raise FileNotFoundError(f"Missing shot file: {shot_file}")
            if shot_id in seen_ids or shot_file in seen_paths:
                raise ValueError(f"Duplicate shot content in timeline: {shot_id}")
            if 'source_shot_start' in shot_data:
                source_range = (shot_data['source_shot_start'], shot_data['source_shot_end'])
                if source_range in seen_ranges:
                    raise ValueError(f"Duplicate source range in timeline: {shot_id}")
                seen_ranges.add(source_range)
            seen_ids.add(shot_id)
            seen_paths.add(shot_file)
            files.append(shot_file)
            current_time += duration
        if 'actual_duration' in timeline and not math.isclose(
                self._number(timeline['actual_duration'], 'actual_duration'),
                current_time, abs_tol=1e-6):
            raise ValueError("Timeline actual_duration does not match its shots")

        # Check the files, not just metadata: FFmpeg otherwise silently shortens
        # out-of-range trims and can render a trailer very different from its plan.
        self._media_info = {}
        for shot_data, shot_file in zip(timeline_shots, files):
            info = self._probe_media(shot_file)
            self._media_info[shot_file] = info
            end = shot_data.get('use_start_offset', 0) + self._duration(shot_data)
            if end > info['duration'] + 1 / fps:
                raise ValueError(f"Source range exceeds video duration: {shot_file}")
        logger.info(f"Validated {len(timeline_shots)} shots")

    @staticmethod
    def _number(value, name: str) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    @classmethod
    def _duration(cls, shot: Dict) -> float:
        value = shot.get('timeline_duration', shot.get('duration'))
        duration = cls._number(value, 'Shot duration')
        if duration <= 0:
            raise ValueError("Shot duration must be positive")
        return duration

    @classmethod
    def _source_duration(cls, shot: Dict) -> Optional[float]:
        available = None
        if 'source_shot_start' in shot or 'source_shot_end' in shot:
            start = cls._number(shot.get('source_shot_start'), 'source_shot_start')
            end = cls._number(shot.get('source_shot_end'), 'source_shot_end')
            if start < 0 or end <= start:
                raise ValueError("Invalid source shot range")
            available = end - start
        if 'source_duration' in shot:
            declared = cls._number(shot['source_duration'], 'source_duration')
            if declared <= 0:
                raise ValueError("Source duration must be positive")
            available = min(available, declared) if available is not None else declared
        return available

    @staticmethod
    def _shot_file(shot: Dict, shots_dir: Path) -> Path:
        path = shot.get('shot_path')
        if path is not None and not isinstance(path, (str, Path)):
            raise ValueError("Shot path must be a filename")
        if path:
            return Path(path)
        return Path(shots_dir) / f"shot_{shot['shot_id']:04d}.mp4"

    @staticmethod
    def _probe_media(path: Path) -> Dict:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_streams', '-show_format',
             '-of', 'json', str(path)],
            check=True, capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        video = next((s for s in streams if s.get('codec_type') == 'video'), None)
        if video is None:
            raise ValueError(f"Shot has no video stream: {path}")
        duration = None
        for value in (video.get('duration'), data.get('format', {}).get('duration')):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed > 0:
                duration = parsed
                break
        if duration is None:
            raise ValueError(f"Cannot determine video duration: {path}")
        return {
            'duration': duration,
            'has_audio': any(s.get('codec_type') == 'audio' for s in streams)
        }

    @staticmethod
    def _timeline_transitions(shots: List[Dict]) -> List[Dict]:
        mapping = {'dissolve': ('dissolve', 0.5), 'fade_black': ('fadeblack', 1.0)}
        return [
            {'shot_index': i, 'type': mapping.get(shot.get('transition_out'), ('cut', 0))[0],
             'duration': mapping.get(shot.get('transition_out'), ('cut', 0))[1]}
            for i, shot in enumerate(shots[:-1])
        ]
    
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
        return self._render(timeline, shots_dir, output_path, [])
    
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
                complex is saved beside the output for debugging.
        """
        logger.info("Using complex filter with transitions...")
        return self._render(timeline, shots_dir, output_path, transitions)

    def _render(self, timeline: Dict, shots_dir: Path, output_path: Path,
                transitions: List[Dict]) -> Path:
        timeline_shots = timeline.get('shots', [])
        filter_complex = self._build_filter_complex(
            timeline_shots, shots_dir, transitions
        )
        inputs = []
        for shot_data in timeline_shots:
            shot_file = self._shot_file(shot_data, shots_dir)
            inputs.extend(['-i', str(shot_file)])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ['ffmpeg', '-nostdin']
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
            debug_file = output_path.with_suffix('.filter_complex.txt')
            with open(debug_file, 'w') as f:
                f.write(filter_complex)
            logger.error(f"Filter complex saved to: {debug_file}")
            raise
        
        return output_path
    
    def _build_filter_complex(self, timeline_shots: List[Dict], 
                             shots_dir: Path, transitions: List[Dict]) -> str:
        """Build FFmpeg filter_complex string for transitions and color grading.
        
        Trim offsets are relative to the extracted shot, not the original movie.
        Transitions use available outgoing source handles so both video and audio
        retain the scheduled duration. Boundaries without handles remain cuts.
        
        Args:
            timeline_shots: List of shot dictionaries from the timeline,
                each containing shot_id and duration information.
            shots_dir: Directory containing the shot video files.
            transitions: List of transition specifications, each containing
                type, duration, and optionally shot_index. Offsets are derived
                from the timeline rather than trusting selector estimates.
            
        Returns:
            str: Complete FFmpeg filter_complex string ready for execution.
        """
        if not timeline_shots:
            raise ValueError("Timeline contains no shots")
        durations = [self._duration(shot) for shot in timeline_shots]
        boundary_transitions = {}
        seen_boundaries = set()
        supported = set(TransitionSelector.TRANSITION_TYPES) | {'fadeblack'}
        for index, transition in enumerate(transitions):
            boundary = transition.get('shot_index', index)
            if (isinstance(boundary, bool) or not isinstance(boundary, int)
                    or not 0 <= boundary < len(timeline_shots) - 1
                    or boundary in seen_boundaries):
                raise ValueError("Invalid or repeated transition boundary")
            seen_boundaries.add(boundary)
            trans_type = transition.get('type', 'cut')
            requested = self._number(transition.get('duration', 0), 'Transition duration')
            if requested < 0:
                raise ValueError("Transition duration cannot be negative")
            if trans_type in ('cut', 'smash_cut') or requested == 0:
                continue
            if trans_type not in supported:
                raise ValueError(f"Unsupported transition: {trans_type}")
            shot = timeline_shots[boundary]
            info = self._media_info.get(self._shot_file(shot, shots_dir).resolve(), {})
            available = self._source_duration(shot)
            if info.get('duration') is not None:
                available = min(available, info['duration']) if available is not None else info['duration']
            end = shot.get('use_start_offset', 0) + durations[boundary]
            handle = max(0.0, available - end) if available is not None else 0.0
            # Use natural outgoing source handles instead of shortening the
            # timeline (or freezing frames). Without handles, preserve a cut.
            overlap = min(requested, handle, durations[boundary] / 2,
                          durations[boundary + 1] / 2)
            overlap = math.floor((overlap + 1e-9) * float(self.fps)) / float(self.fps)
            if overlap > 0:
                boundary_transitions[boundary] = (trans_type, overlap)

        filters = []
        width, height = self.resolution.split('x')
        color = self._get_color_grade_filter() if self.enable_color_grading else ''
        for i, shot in enumerate(timeline_shots):
            duration = durations[i] + boundary_transitions.get(i, ('cut', 0))[1]
            start = shot.get('use_start_offset', 0)
            filters.append(
                f"[{i}:v]trim=start={start}:duration={duration},setpts=PTS-STARTPTS,"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                f"fps={self.fps},format=yuv420p,settb=AVTB"
                f"{',' + color if color else ''}[v{i}]"
            )
            info = self._media_info.get(self._shot_file(shot, shots_dir).resolve(), {})
            if info.get('has_audio', True):
                filters.append(
                    f"[{i}:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS,"
                    f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"apad,atrim=duration={duration}[a{i}]"
                )
            else:
                filters.append(
                    f"anullsrc=r=48000:cl=stereo,atrim=duration={duration}[a{i}]"
                )

        if not boundary_transitions:
            pairs = ''.join(f"[v{i}][a{i}]" for i in range(len(timeline_shots)))
            filters.append(f"{pairs}concat=n={len(timeline_shots)}:v=1:a=1[outv][outa]")
        else:
            video, audio = 'v0', 'a0'
            boundary_time = durations[0]
            for i in range(len(timeline_shots) - 1):
                next_video = 'outv' if i == len(timeline_shots) - 2 else f"joinedv{i}"
                next_audio = 'outa' if i == len(timeline_shots) - 2 else f"joineda{i}"
                if i in boundary_transitions:
                    trans_type, overlap = boundary_transitions[i]
                    filters.append(
                        f"[{video}][v{i + 1}]xfade=transition={trans_type}:"
                        f"duration={overlap}:offset={boundary_time}[{next_video}]"
                    )
                    filters.append(
                        f"[{audio}][a{i + 1}]acrossfade=d={overlap}:c1=tri:c2=tri[{next_audio}]"
                    )
                else:
                    filters.append(
                        f"[{video}][v{i + 1}]concat=n=2:v=1:a=0,"
                        f"fps={self.fps},settb=AVTB[{next_video}]"
                    )
                    filters.append(f"[{audio}][a{i + 1}]concat=n=2:v=0:a=1[{next_audio}]")
                video, audio = next_video, next_audio
                boundary_time += durations[i + 1]
        return ';'.join(filters)
    
    def _get_color_grade_filter(self) -> str:
        """Get color grading filter string from genre profile.
        
        Returns:
            str: FFmpeg filter string for color grading, or empty string
                if no color grading is configured.
        """
        return self.genre_profile.get('color_grade', {}).get('filter', '')
