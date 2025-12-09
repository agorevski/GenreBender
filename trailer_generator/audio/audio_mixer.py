"""
Audio mixing engine for trailer generation.
Handles music mixing, audio ducking, normalization, and final audio assembly.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from .music_selector import MusicSelector

logger = logging.getLogger(__name__)

class AudioMixer:
    """
    Mixes audio for trailer including music, original audio, and effects.
    """
    
    def __init__(self, config: Dict, genre_profile: Dict, output_dir: Path,
                 azure_client=None, enable_ducking: bool = True):
        """
        Initialize audio mixer.
        
        Args:
            config: Global configuration dictionary
            genre_profile: Genre-specific configuration
            output_dir: Base output directory
            azure_client: Azure OpenAI client for AI features (optional)
            enable_ducking: Whether to apply audio ducking
        """
        self.config = config
        self.genre_profile = genre_profile
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / 'temp'
        self.azure_client = azure_client
        self.enable_ducking = enable_ducking
        
        # Audio settings from config
        self.audio_config = config.get('audio', {})
        self.sample_rate = self.audio_config.get('sample_rate')
        self.bitrate = self.audio_config.get('bitrate')
        self.output_format = self.audio_config.get('output_format')
        self.ducking_threshold = self.audio_config.get('ducking_threshold')
        self.ducking_ratio = self.audio_config.get('ducking_ratio')
        self.normalization_target = self.audio_config.get('normalization_target')
        
        # Initialize music selector
        music_lib_path = self.audio_config.get('music_library_path')
        self.music_selector = MusicSelector(
            azure_client=azure_client,
            genre_profile=genre_profile,
            music_library_path=music_lib_path,
            enable_ai=self.audio_config.get('ai_music_selection')
        )
        
        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def mix_audio(self, timeline: Dict, video_path: Path, output_path: Path,
                  music_file: Optional[str] = None) -> str:
        """
        Main audio mixing orchestrator.
        
        Args:
            timeline: Timeline dictionary
            video_path: Path to assembled video (without final audio)
            output_path: Path for final output
            music_file: Optional specific music file to use
            
        Returns:
            Path to final video with mixed audio
        """
        logger.info("Starting audio mixing...")
        
        # Step 1: Select music
        logger.info("Selecting music track...")
        music_selection = self.music_selector.select_music(timeline, music_file)
        
        if not music_selection.get('file'):
            # No music available - use original audio only
            logger.warning("No music available, using original audio only")
            return self._use_original_audio_only(video_path, output_path)
        
        logger.info(f"Selected music: {music_selection['file']}")
        logger.info(f"Music source: {music_selection.get('source', 'unknown')}")
        
        # Step 2: Extract original audio from video
        logger.info("Extracting original audio from video...")
        original_audio = self._extract_audio(video_path)
        
        # Step 3: Build audio mix
        logger.info("Building audio mix...")
        if self.enable_ducking and original_audio:
            # Mix with ducking (lower music when dialogue present)
            mixed_audio = self._mix_with_ducking(
                music_file=music_selection['file'],
                original_audio=original_audio,
                timeline=timeline
            )
        else:
            # Simple mix without ducking
            mixed_audio = self._mix_simple(
                music_file=music_selection['file'],
                original_audio=original_audio,
                timeline=timeline
            )
        
        # Step 4: Normalize audio
        logger.info("Normalizing audio...")
        normalized_audio = self._normalize_audio(mixed_audio)
        
        # Step 5: Mux audio with video
        logger.info("Muxing audio with video...")
        final_video = self._mux_audio_video(video_path, normalized_audio, output_path)
        
        # Cleanup temp files
        self._cleanup_temp_files([original_audio, mixed_audio, normalized_audio])
        
        logger.info(f"Audio mixing complete: {final_video}")
        return str(final_video)
    
    def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """
        Extract audio track from video.
        
        Args:
            video_path: Input video path
            
        Returns:
            Path to extracted audio file or None if no audio
        """
        output_audio = self.temp_dir / 'original_audio.wav'
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM for processing
            '-ar', str(self.sample_rate),
            '-ac', '2',  # Stereo
            '-y',
            str(output_audio)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Extracted audio to: {output_audio}")
            return output_audio
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to extract audio: {e.stderr}")
            return None
    
    def _mix_with_ducking(self, music_file: str, original_audio: Optional[Path],
                          timeline: Dict) -> Path:
        """
        Mix music and original audio with ducking.
        
        Args:
            music_file: Path to music file
            original_audio: Path to original audio
            timeline: Timeline dictionary
            
        Returns:
            Path to mixed audio file
        """
        output_audio = self.temp_dir / 'mixed_ducked.wav'
        
        # Build filter_complex for ducking
        # sidechaincompress lowers music volume when original audio is present
        if original_audio and original_audio.exists():
            filter_complex = (
                f"[0:a]volume=0.3[music];"  # Lower music base volume
                f"[music][1:a]sidechaincompress="
                f"threshold={self.ducking_threshold}dB:"
                f"ratio={self.ducking_ratio}:"
                f"attack=200:"
                f"release=1000[mixed]"
            )
            
            cmd = [
                'ffmpeg',
                '-i', music_file,
                '-i', str(original_audio),
                '-filter_complex', filter_complex,
                '-map', '[mixed]',
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '2',
                '-y',
                str(output_audio)
            ]
        else:
            # No original audio, just use music
            cmd = [
                'ffmpeg',
                '-i', music_file,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '2',
                '-af', 'volume=0.5',  # Lower volume
                '-y',
                str(output_audio)
            ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Mixed audio with ducking: {output_audio}")
            return output_audio
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio mixing failed: {e.stderr}")
            raise
    
    def _mix_simple(self, music_file: str, original_audio: Optional[Path],
                    timeline: Dict) -> Path:
        """
        Simple audio mix without ducking.
        
        Args:
            music_file: Path to music file
            original_audio: Path to original audio
            timeline: Timeline dictionary
            
        Returns:
            Path to mixed audio file
        """
        output_audio = self.temp_dir / 'mixed_simple.wav'
        
        if original_audio and original_audio.exists():
            # Mix both tracks
            cmd = [
                'ffmpeg',
                '-i', music_file,
                '-i', str(original_audio),
                '-filter_complex',
                '[0:a]volume=0.3[a1];[1:a]volume=0.7[a2];[a1][a2]amix=inputs=2:duration=first[mixed]',
                '-map', '[mixed]',
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '2',
                '-y',
                str(output_audio)
            ]
        else:
            # Music only
            cmd = [
                'ffmpeg',
                '-i', music_file,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '2',
                '-af', 'volume=0.5',
                '-y',
                str(output_audio)
            ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Mixed audio (simple): {output_audio}")
            return output_audio
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio mixing failed: {e.stderr}")
            raise
    
    def _normalize_audio(self, audio_path: Path) -> Path:
        """
        Normalize audio to target LUFS.
        
        Args:
            audio_path: Input audio path
            
        Returns:
            Path to normalized audio
        """
        output_audio = self.temp_dir / 'normalized.wav'
        
        # Use loudnorm filter for LUFS normalization
        cmd = [
            'ffmpeg',
            '-i', str(audio_path),
            '-af', f'loudnorm=I={self.normalization_target}:TP=-1.5:LRA=11',
            '-ar', str(self.sample_rate),
            '-y',
            str(output_audio)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Normalized audio: {output_audio}")
            return output_audio
        except subprocess.CalledProcessError as e:
            logger.warning(f"Normalization failed: {e.stderr}, using original")
            return audio_path
    
    def _mux_audio_video(self, video_path: Path, audio_path: Path,
                         output_path: Path) -> Path:
        """
        Mux audio with video to create final output.
        
        Args:
            video_path: Input video path
            audio_path: Audio path to mux
            output_path: Output path
            
        Returns:
            Path to final video
        """
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',  # Copy video without re-encoding
            '-c:a', self.output_format,
            '-b:a', self.bitrate,
            '-map', '0:v:0',  # Video from first input
            '-map', '1:a:0',  # Audio from second input
            '-shortest',  # End when shortest stream ends
            '-y',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Muxed final video: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Muxing failed: {e.stderr}")
            raise
    
    def _use_original_audio_only(self, video_path: Path, output_path: Path) -> str:
        """
        Fallback: use video's original audio without music.
        
        Args:
            video_path: Input video path
            output_path: Output path
            
        Returns:
            Path to output video
        """
        logger.info("Using original audio only (no music)")
        
        # Simply copy the video
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-c', 'copy',
            '-y',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Copy failed: {e.stderr}")
            raise
    
    def _cleanup_temp_files(self, file_paths: List[Optional[Path]]):
        """
        Clean up temporary audio files.
        
        Args:
            file_paths: List of temporary file paths to delete
        """
        for path in file_paths:
            if path and path.exists():
                try:
                    path.unlink()
                    logger.debug(f"Deleted temp file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")
