"""Preprocesses audio for optimal diarization and transcription."""
import logging
import numpy as np
from typing import List, Tuple, Optional
import scipy.signal as signal

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocesses audio for optimal speech recognition and diarization.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        target_db: float = -20.0,
        enable_vad: bool = True
    ):
        """
        Initialize preprocessor with target parameters.
        
        Parameters:
            sample_rate: Audio sample rate in Hz
            target_db: Target loudness in dB
            enable_vad: Enable Voice Activity Detection
        """
        self.sample_rate = sample_rate
        self.target_db = target_db
        self.enable_vad = enable_vad
        
        # VAD thresholds
        self.silence_threshold_db = -40.0
        self.min_speech_duration_ms = 300
        self.min_silence_duration_ms = 200
        
        logger.info(f"AudioPreprocessor initialized: {sample_rate}Hz, target_db={target_db}")
    
    def normalize(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Normalizes audio to target dB level.
        
        Parameters:
            audio_chunk: Input audio as numpy array
        
        Returns:
            numpy array: Normalized audio
        """
        if len(audio_chunk) == 0:
            return audio_chunk
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        if rms < 1e-10:  # Avoid division by zero
            return audio_chunk
        
        # Convert to dB
        current_db = 20 * np.log10(rms)
        
        # Calculate gain needed
        gain_db = self.target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain and clip
        normalized = audio_chunk * gain_linear
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def remove_silence(
        self,
        audio_chunk: np.ndarray,
        threshold_db: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Removes silent portions and returns speech segments.
        
        Parameters:
            audio_chunk: Input audio as numpy array
            threshold_db: Silence threshold in dB (None uses default)
        
        Returns:
            tuple: (audio with silence removed, list of (start, end) speech segments)
        """
        if threshold_db is None:
            threshold_db = self.silence_threshold_db
        
        # Calculate frame energy
        frame_length = int(0.02 * self.sample_rate)  # 20ms frames
        hop_length = frame_length // 2
        
        frames = []
        for i in range(0, len(audio_chunk) - frame_length, hop_length):
            frame = audio_chunk[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            
            if rms > 0:
                db = 20 * np.log10(rms)
            else:
                db = -100
            
            frames.append((i, db > threshold_db))
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        start_idx = 0
        
        for i, (pos, is_speech) in enumerate(frames):
            if is_speech and not in_speech:
                start_idx = pos
                in_speech = True
            elif not is_speech and in_speech:
                end_idx = pos
                duration_ms = (end_idx - start_idx) / self.sample_rate * 1000
                
                if duration_ms >= self.min_speech_duration_ms:
                    speech_segments.append((start_idx, end_idx))
                
                in_speech = False
        
        # Handle last segment
        if in_speech:
            speech_segments.append((start_idx, len(audio_chunk)))
        
        # Concatenate speech segments
        if speech_segments:
            speech_audio = np.concatenate([
                audio_chunk[start:end] for start, end in speech_segments
            ])
        else:
            speech_audio = np.array([])
        
        return speech_audio, speech_segments
    
    def apply_vad(self, audio_chunk: np.ndarray) -> List[Tuple[int, int]]:
        """
        Applies Voice Activity Detection.
        
        Parameters:
            audio_chunk: Input audio as numpy array
        
        Returns:
            list of tuples: (start_ms, end_ms) speech segments in milliseconds
        """
        _, segments = self.remove_silence(audio_chunk)
        
        # Convert to milliseconds
        segments_ms = [
            (int(start / self.sample_rate * 1000), int(end / self.sample_rate * 1000))
            for start, end in segments
        ]
        
        return segments_ms
    
    def apply_highpass_filter(
        self,
        audio_chunk: np.ndarray,
        cutoff_hz: float = 80.0
    ) -> np.ndarray:
        """
        Apply highpass filter to remove low-frequency noise.
        
        Parameters:
            audio_chunk: Input audio
            cutoff_hz: Cutoff frequency in Hz
        
        Returns:
            numpy array: Filtered audio
        """
        if len(audio_chunk) == 0:
            return audio_chunk
        
        # Design butterworth highpass filter
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist
        
        b, a = signal.butter(5, normalized_cutoff, btype='high')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio_chunk)
        
        return filtered
    
    def resample(self, audio_chunk: np.ndarray, target_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Parameters:
            audio_chunk: Input audio
            target_rate: Target sample rate in Hz
        
        Returns:
            numpy array: Resampled audio
        """
        if self.sample_rate == target_rate:
            return audio_chunk
        
        # Calculate resampling ratio
        ratio = target_rate / self.sample_rate
        num_samples = int(len(audio_chunk) * ratio)
        
        # Resample using scipy
        resampled = signal.resample(audio_chunk, num_samples)
        
        return resampled
    
    def preprocess(
        self,
        audio_chunk: np.ndarray,
        normalize: bool = True,
        remove_silence: bool = False,
        highpass: bool = True
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Parameters:
            audio_chunk: Input audio
            normalize: Apply normalization
            remove_silence: Remove silent portions
            highpass: Apply highpass filter
        
        Returns:
            numpy array: Preprocessed audio
        """
        processed = audio_chunk.copy()
        
        # Highpass filter
        if highpass:
            processed = self.apply_highpass_filter(processed)
        
        # Remove silence
        if remove_silence:
            processed, _ = self.remove_silence(processed)
        
        # Normalize
        if normalize and len(processed) > 0:
            processed = self.normalize(processed)
        
        return processed
    
    def get_audio_stats(self, audio_chunk: np.ndarray) -> dict:
        """
        Calculate audio statistics.
        
        Parameters:
            audio_chunk: Input audio
        
        Returns:
            dict: Audio statistics
        """
        if len(audio_chunk) == 0:
            return {
                'rms': 0,
                'db': -100,
                'peak': 0,
                'duration_ms': 0,
                'is_speech': False
            }
        
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        db = 20 * np.log10(rms) if rms > 0 else -100
        peak = np.max(np.abs(audio_chunk))
        duration_ms = len(audio_chunk) / self.sample_rate * 1000
        is_speech = db > self.silence_threshold_db
        
        return {
            'rms': float(rms),
            'db': float(db),
            'peak': float(peak),
            'duration_ms': float(duration_ms),
            'is_speech': bool(is_speech)
        }
