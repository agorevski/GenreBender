"""
Audio feature extraction module.
Extracts audio features (MFCC, spectral, etc.) from video shots for multimodal analysis.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

# Lazy import librosa (only if available)
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Audio extraction will be disabled.")


class AudioExtractor:
    """
    Extracts audio features from video shots for multimodal analysis.
    Uses librosa for audio processing and feature extraction.
    """
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        """
        Initialize audio extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        if not LIBROSA_AVAILABLE:
            logger.error("librosa is not installed. Install with: pip install librosa soundfile")
    
    def extract_audio_features(self, video_path: str, shots: List[Dict]) -> List[Dict]:
        """
        Extract audio features for all shots in the video.
        
        Args:
            video_path: Path to source video
            shots: List of shot dictionaries with timing information
            
        Returns:
            Updated shots list with audio features
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("Skipping audio extraction (librosa not available)")
            for shot in shots:
                shot['audio_features'] = None
            return shots
        
        logger.info(f"Extracting audio features from {len(shots)} shots...")
        
        # Extract full audio from video
        audio_data, sr = self._extract_audio_from_video(video_path)
        
        if audio_data is None:
            logger.error("Failed to extract audio from video")
            for shot in shots:
                shot['audio_features'] = None
            return shots
        
        # Extract features for each shot
        for i, shot in enumerate(shots, start=1):
            features = self._extract_shot_features(
                audio_data, sr, shot['start_time'], shot['end_time']
            )
            shot['audio_features'] = features
            
            if i % 50 == 0:
                logger.info(f"Extracted audio features for {i}/{len(shots)} shots")
        
        successful = sum(1 for s in shots if s.get('audio_features'))
        logger.info(f"Successfully extracted audio features for {successful}/{len(shots)} shots")
        
        return shots
    
    def _extract_audio_from_video(self, video_path: str) -> tuple[Optional[np.ndarray], Optional[int]]:
        """
        Extract audio track from video file using ffmpeg.
        
        Args:
            video_path: Path to source video
            
        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if failed
        """
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_audio_path = tmp_file.name
            
            # Extract audio using ffmpeg
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM audio
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                tmp_audio_path
            ]
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr.decode()}")
                return None, None
            
            # Load audio with librosa
            audio_data, sr = librosa.load(tmp_audio_path, sr=self.sample_rate, mono=True)
            
            # Clean up temp file
            os.unlink(tmp_audio_path)
            
            return audio_data, sr
            
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timeout while extracting audio")
            return None, None
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None, None
    
    def _extract_shot_features(self, audio_data: np.ndarray, sr: int,
                               start_time: float, end_time: float) -> Optional[Dict]:
        """
        Extract audio features for a specific time segment.
        
        Args:
            audio_data: Full audio array
            sr: Sample rate
            start_time: Shot start time in seconds
            end_time: Shot end time in seconds
            
        Returns:
            Dictionary of audio features or None if failed
        """
        try:
            # Extract audio segment for this shot
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = audio_data[start_sample:end_sample]
            
            # Handle empty or very short segments
            if len(segment) < sr * 0.1:  # Less than 0.1 seconds
                return self._empty_features()
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfccs, axis=1).tolist()
            mfcc_std = np.std(mfccs, axis=1).tolist()
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
            
            # Extract temporal features
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
            rms_energy = librosa.feature.rms(y=segment)
            
            # Extract chroma features (musical content)
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            chroma_mean = np.mean(chroma, axis=1).tolist()
            
            # Compute tempo (if segment is long enough)
            tempo = None
            if len(segment) >= sr * 2:  # At least 2 seconds
                try:
                    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
                    tempo = float(tempo)
                except:
                    tempo = None
            
            features = {
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'rms_energy_mean': float(np.mean(rms_energy)),
                'rms_energy_std': float(np.std(rms_energy)),
                'chroma_mean': chroma_mean,
                'tempo': tempo,
                'duration': end_time - start_time
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract features for shot: {e}")
            return self._empty_features()
    
    def _empty_features(self) -> Dict:
        """
        Return empty feature structure for failed extractions.
        
        Returns:
            Dictionary with default/zero values
        """
        return {
            'mfcc_mean': [0.0] * self.n_mfcc,
            'mfcc_std': [0.0] * self.n_mfcc,
            'spectral_centroid_mean': 0.0,
            'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0,
            'spectral_bandwidth_mean': 0.0,
            'zero_crossing_rate_mean': 0.0,
            'rms_energy_mean': 0.0,
            'rms_energy_std': 0.0,
            'chroma_mean': [0.0] * 12,
            'tempo': None,
            'duration': 0.0
        }
    
    def extract_audio_summary(self, audio_features: Dict) -> Dict:
        """
        Create a compact summary of audio features for analysis.
        
        Args:
            audio_features: Full audio feature dictionary
            
        Returns:
            Compact summary dictionary
        """
        if not audio_features:
            return {'audio_type': 'silent', 'intensity': 0.0}
        
        # Classify audio type based on features
        energy = audio_features.get('rms_energy_mean', 0)
        zcr = audio_features.get('zero_crossing_rate_mean', 0)
        spectral_centroid = audio_features.get('spectral_centroid_mean', 0)
        
        # Simple heuristics for audio classification
        audio_type = 'unknown'
        if energy < 0.01:
            audio_type = 'silent'
        elif zcr > 0.15:
            audio_type = 'high_frequency'  # Dialog, effects
        elif spectral_centroid > 3000:
            audio_type = 'bright'  # Music, action
        elif spectral_centroid < 1000:
            audio_type = 'dark'  # Low rumble, suspense
        else:
            audio_type = 'balanced'
        
        return {
            'audio_type': audio_type,
            'intensity': float(energy),
            'brightness': float(spectral_centroid / 4000),  # Normalized
            'tempo': audio_features.get('tempo')
        }
