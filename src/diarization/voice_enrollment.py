"""Enrolls known speakers for identification."""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .speaker_embeddings import SpeakerEmbeddingExtractor

logger = logging.getLogger(__name__)


class VoiceEnrollment:
    """
    Enrolls known speakers for identification.
    """
    
    def __init__(
        self,
        embedding_extractor: SpeakerEmbeddingExtractor,
        storage_path: str = "./models/speaker_embedding/enrolled"
    ):
        """
        Initializes enrollment system.
        
        Parameters:
            embedding_extractor: Speaker embedding extractor instance
            storage_path: Path to store enrolled speaker data
        """
        self.embedding_extractor = embedding_extractor
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._enrolled_speakers: Dict[str, Dict] = {}
        self._speaker_count = 0
        
        # Load existing enrollments
        self.load_enrolled_speakers()
        
        logger.info(f"VoiceEnrollment initialized: {len(self._enrolled_speakers)} enrolled speakers")
    
    def enroll_speaker(
        self,
        audio_samples: List[np.ndarray],
        speaker_name: str,
        sample_rate: int = 16000
    ) -> Tuple[str, np.ndarray]:
        """
        Enrolls speaker from multiple audio samples.
        
        Parameters:
            audio_samples: List of audio samples for enrollment
            speaker_name: Name of the speaker
            sample_rate: Audio sample rate
        
        Returns:
            tuple: (speaker_id, averaged_embedding)
        """
        if len(audio_samples) < 1:
            raise ValueError("At least one audio sample required for enrollment")
        
        # Extract embeddings from all samples
        embeddings = []
        for sample in audio_samples:
            if len(sample) >= sample_rate * 0.5:  # Minimum 0.5 seconds
                emb = self.embedding_extractor.extract_embedding(sample, sample_rate)
                if np.any(emb):
                    embeddings.append(emb)
        
        if len(embeddings) == 0:
            raise ValueError("Could not extract valid embeddings from samples")
        
        # Average the embeddings
        averaged_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(averaged_embedding)
        if norm > 0:
            averaged_embedding = averaged_embedding / norm
        
        # Generate speaker ID
        self._speaker_count += 1
        speaker_id = f"enrolled_{self._speaker_count:03d}"
        
        # Store enrollment
        self._enrolled_speakers[speaker_id] = {
            'name': speaker_name,
            'embedding': averaged_embedding,
            'sample_count': len(embeddings),
            'enrolled_at': str(np.datetime64('now'))
        }
        
        # Save to disk
        self._save_enrollment(speaker_id)
        
        logger.info(f"Enrolled speaker '{speaker_name}' as {speaker_id} ({len(embeddings)} samples)")
        
        return speaker_id, averaged_embedding
    
    def load_enrolled_speakers(self) -> Dict[str, Dict]:
        """
        Loads previously enrolled speakers.
        
        Returns:
            dict: Mapping of speaker_id to embedding and name
        """
        self._enrolled_speakers.clear()
        
        # Load metadata
        metadata_file = self.storage_path / "enrollments.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                for speaker_id, info in metadata.items():
                    emb_file = self.storage_path / f"{speaker_id}.npy"
                    if emb_file.exists():
                        embedding = np.load(emb_file)
                        self._enrolled_speakers[speaker_id] = {
                            'name': info['name'],
                            'embedding': embedding,
                            'sample_count': info.get('sample_count', 1),
                            'enrolled_at': info.get('enrolled_at', '')
                        }
                
                logger.info(f"Loaded {len(self._enrolled_speakers)} enrolled speakers")
                
            except Exception as e:
                logger.error(f"Error loading enrollments: {e}")
        
        return {
            sid: {'name': info['name'], 'embedding': info['embedding']}
            for sid, info in self._enrolled_speakers.items()
        }
    
    def match_to_enrolled(
        self,
        embedding: np.ndarray,
        threshold: float = 0.75
    ) -> Tuple[Optional[str], float]:
        """
        Matches embedding against enrolled speakers.
        
        Parameters:
            embedding: Speaker embedding to match
            threshold: Minimum similarity threshold
        
        Returns:
            tuple: (speaker_name or None, confidence)
        """
        best_match = None
        best_similarity = 0.0
        
        for speaker_id, info in self._enrolled_speakers.items():
            similarity = self.embedding_extractor.compare_embeddings(
                embedding,
                info['embedding']
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = info['name']
        
        if best_similarity >= threshold:
            logger.debug(f"Matched enrolled speaker: {best_match} ({best_similarity:.3f})")
            return best_match, best_similarity
        
        return None, best_similarity
    
    def get_enrolled_speakers(self) -> List[Dict]:
        """
        Get list of all enrolled speakers.
        
        Returns:
            list: List of enrolled speaker info
        """
        return [
            {
                'speaker_id': sid,
                'name': info['name'],
                'sample_count': info['sample_count'],
                'enrolled_at': info['enrolled_at']
            }
            for sid, info in self._enrolled_speakers.items()
        ]
    
    def remove_speaker(self, speaker_id: str) -> bool:
        """
        Remove an enrolled speaker.
        
        Parameters:
            speaker_id: Speaker ID to remove
        
        Returns:
            bool: Success
        """
        if speaker_id not in self._enrolled_speakers:
            return False
        
        # Remove from memory
        del self._enrolled_speakers[speaker_id]
        
        # Remove files
        emb_file = self.storage_path / f"{speaker_id}.npy"
        if emb_file.exists():
            emb_file.unlink()
        
        # Update metadata
        self._save_metadata()
        
        logger.info(f"Removed enrolled speaker: {speaker_id}")
        return True
    
    def update_speaker(
        self,
        speaker_id: str,
        new_samples: List[np.ndarray],
        sample_rate: int = 16000
    ) -> bool:
        """
        Update enrolled speaker with additional samples.
        
        Parameters:
            speaker_id: Speaker ID to update
            new_samples: Additional audio samples
            sample_rate: Audio sample rate
        
        Returns:
            bool: Success
        """
        if speaker_id not in self._enrolled_speakers:
            return False
        
        # Extract new embeddings
        new_embeddings = []
        for sample in new_samples:
            if len(sample) >= sample_rate * 0.5:
                emb = self.embedding_extractor.extract_embedding(sample, sample_rate)
                if np.any(emb):
                    new_embeddings.append(emb)
        
        if not new_embeddings:
            return False
        
        # Combine with existing
        existing_emb = self._enrolled_speakers[speaker_id]['embedding']
        existing_count = self._enrolled_speakers[speaker_id]['sample_count']
        
        # Weighted average
        total_count = existing_count + len(new_embeddings)
        new_avg = np.mean(new_embeddings, axis=0)
        
        combined = (existing_emb * existing_count + new_avg * len(new_embeddings)) / total_count
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        # Update
        self._enrolled_speakers[speaker_id]['embedding'] = combined
        self._enrolled_speakers[speaker_id]['sample_count'] = total_count
        
        # Save
        self._save_enrollment(speaker_id)
        
        logger.info(f"Updated {speaker_id} with {len(new_embeddings)} new samples")
        return True
    
    def _save_enrollment(self, speaker_id: str) -> None:
        """Save enrollment to disk."""
        try:
            info = self._enrolled_speakers[speaker_id]
            
            # Save embedding
            emb_file = self.storage_path / f"{speaker_id}.npy"
            np.save(emb_file, info['embedding'])
            
            # Save metadata
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error saving enrollment: {e}")
    
    def _save_metadata(self) -> None:
        """Save enrollment metadata."""
        try:
            metadata = {
                sid: {
                    'name': info['name'],
                    'sample_count': info['sample_count'],
                    'enrolled_at': info['enrolled_at']
                }
                for sid, info in self._enrolled_speakers.items()
            }
            
            metadata_file = self.storage_path / "enrollments.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
