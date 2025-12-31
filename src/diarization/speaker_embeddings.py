"""Extracts and manages speaker voice embeddings."""
import logging
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeakerEmbeddingExtractor:
    """
    Extracts voice embeddings from audio segments.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 192,
        device: str = "cpu"
    ):
        """
        Loads speaker embedding model.
        
        Parameters:
            model_path: Path to pre-trained model or model name
            embedding_dim: Dimension of embedding vectors
            device: Device to run model on (cpu/cuda)
        """
        self.model_path = model_path or "speechbrain/spkrec-ecapa-voxceleb"
        self.embedding_dim = embedding_dim
        self.device = device
        self._model = None
        self._initialized = False
        
        logger.info(f"SpeakerEmbeddingExtractor initialized: dim={embedding_dim}")
    
    def _load_model(self) -> None:
        """Load the embedding model lazily."""
        if self._initialized:
            return
        
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            self._model = EncoderClassifier.from_hparams(
                source=self.model_path,
                savedir=f"./models/speaker_embedding/{Path(self.model_path).name}",
                run_opts={"device": self.device}
            )
            self._initialized = True
            logger.info(f"Loaded speaker embedding model: {self.model_path}")
            
        except ImportError:
            logger.warning("SpeechBrain not available, using fallback embedding")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self._initialized = True
    
    def extract_embedding(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts voice embedding from audio.
        
        Parameters:
            audio_segment: Audio data as numpy array
            sample_rate: Sample rate of audio
        
        Returns:
            numpy array: Embedding vector
        """
        self._load_model()
        
        if len(audio_segment) < sample_rate * 0.5:  # Minimum 0.5 seconds
            logger.warning("Audio segment too short for embedding")
            return np.zeros(self.embedding_dim)
        
        try:
            if self._model is not None:
                import torch
                
                # Convert to tensor
                audio_tensor = torch.tensor(audio_segment).unsqueeze(0).float()
                
                # Extract embedding
                with torch.no_grad():
                    embedding = self._model.encode_batch(audio_tensor)
                    embedding = embedding.squeeze().cpu().numpy()
                
                return embedding
            else:
                # Fallback: simple spectral features
                return self._extract_fallback_embedding(audio_segment, sample_rate)
                
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def _extract_fallback_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Fallback embedding using spectral features."""
        try:
            import librosa
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            
            # Compute statistics
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Combine features
            features = np.concatenate([mfcc_mean, mfcc_std])
            
            # Pad or truncate to embedding_dim
            if len(features) < self.embedding_dim:
                features = np.pad(features, (0, self.embedding_dim - len(features)))
            else:
                features = features[:self.embedding_dim]
            
            return features
            
        except Exception as e:
            logger.error(f"Fallback embedding failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Computes cosine similarity between embeddings.
        
        Parameters:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            float: Similarity score 0-1
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2
        
        # Cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
