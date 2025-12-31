"""Central configuration management with validation."""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class AppConfig:
    """Application configuration."""
    
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    openai_base_url: str = "https://api.openai.com/v1"
    
    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-pro"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    # Cartesia TTS Configuration
    cartesia_api_key: str = ""
    cartesia_voice_id: str = "default"
    cartesia_model: str = "sonic-english"
    
    # LLM Provider Priority
    llm_provider_chain: List[str] = field(default_factory=lambda: ["openai", "ollama"])
    
    # Audio Configuration
    sample_rate: int = 16000
    chunk_size: int = 1024
    buffer_duration: float = 5.0
    audio_device_index: int = -1
    
    # Wake Word Configuration
    wake_phrases: List[str] = field(default_factory=lambda: ["Hey Expert", "Okay Assistant"])
    wake_word_sensitivity: float = 0.5
    wake_word_model_path: str = "./models/wake_word/"
    
    # Speaker Diarization Configuration
    max_speakers: int = 10
    speaker_similarity_threshold: float = 0.75
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    
    # Transcription Configuration
    stt_model: str = "base.en"
    stt_language: str = "en"
    stt_compute_type: str = "float32"
    
    # Expert Configuration
    knowledge_base_path: str = "./knowledge/documents"
    embeddings_path: str = "./knowledge/embeddings"
    max_context_tokens: int = 4000
    max_knowledge_chunks: int = 5
    
    # Meeting Configuration
    meeting_display_name: str = "AI Expert"
    zoom_sdk_key: str = ""
    zoom_sdk_secret: str = ""
    
    # Performance Configuration
    max_latency_ms: int = 500
    enable_gpu: bool = False
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/voice_assistant.log"
    
    # Output Configuration
    save_transcripts: bool = True
    transcript_output_dir: str = "./outputs"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        
        if self.chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {self.chunk_size}")
        
        if not 0 <= self.wake_word_sensitivity <= 1:
            raise ValueError(f"wake_word_sensitivity must be between 0 and 1")
        
        if self.max_speakers < 1:
            raise ValueError(f"max_speakers must be at least 1")
        
        if not 0 <= self.speaker_similarity_threshold <= 1:
            raise ValueError(f"speaker_similarity_threshold must be between 0 and 1")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        
        # Create required directories
        Path(self.knowledge_base_path).mkdir(parents=True, exist_ok=True)
        Path(self.embeddings_path).mkdir(parents=True, exist_ok=True)
        Path(self.transcript_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.wake_word_model_path).mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load and validate configuration from environment and files.
    
    Parameters:
        config_path: Optional path to .env file
    
    Returns:
        AppConfig: Validated configuration object
    """
    # Load environment variables
    if config_path:
        load_dotenv(config_path)
    else:
        load_dotenv()
    
    # Parse LLM provider chain
    provider_chain_str = os.getenv("LLM_PROVIDER_CHAIN", "openai,ollama")
    llm_provider_chain = [p.strip() for p in provider_chain_str.split(",")]
    
    # Parse wake phrases
    wake_phrases_str = os.getenv("WAKE_PHRASES", "Hey Expert,Okay Assistant")
    wake_phrases = [p.strip() for p in wake_phrases_str.split(",")]
    
    # Create configuration
    config = AppConfig(
        # OpenAI
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        
        # Gemini
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-pro"),
        
        # Ollama
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama2"),
        
        # Cartesia TTS
        cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
        cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", "default"),
        cartesia_model=os.getenv("CARTESIA_MODEL", "sonic-english"),
        
        # LLM Provider Chain
        llm_provider_chain=llm_provider_chain,
        
        # Audio
        sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
        buffer_duration=float(os.getenv("BUFFER_DURATION", "5.0")),
        audio_device_index=int(os.getenv("AUDIO_DEVICE_INDEX", "-1")),
        
        # Wake Word
        wake_phrases=wake_phrases,
        wake_word_sensitivity=float(os.getenv("WAKE_WORD_SENSITIVITY", "0.5")),
        wake_word_model_path=os.getenv("WAKE_WORD_MODEL_PATH", "./models/wake_word/"),
        
        # Speaker Diarization
        max_speakers=int(os.getenv("MAX_SPEAKERS", "10")),
        speaker_similarity_threshold=float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75")),
        speaker_embedding_model=os.getenv("SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"),
        
        # Transcription
        stt_model=os.getenv("STT_MODEL", "base.en"),
        stt_language=os.getenv("STT_LANGUAGE", "en"),
        stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "float32"),
        
        # Expert
        knowledge_base_path=os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge/documents"),
        embeddings_path=os.getenv("EMBEDDINGS_PATH", "./knowledge/embeddings"),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4000")),
        max_knowledge_chunks=int(os.getenv("MAX_KNOWLEDGE_CHUNKS", "5")),
        
        # Meeting
        meeting_display_name=os.getenv("MEETING_DISPLAY_NAME", "AI Expert"),
        zoom_sdk_key=os.getenv("ZOOM_SDK_KEY", ""),
        zoom_sdk_secret=os.getenv("ZOOM_SDK_SECRET", ""),
        
        # Performance
        max_latency_ms=int(os.getenv("MAX_LATENCY_MS", "500")),
        enable_gpu=os.getenv("ENABLE_GPU", "false").lower() == "true",
        
        # Logging
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "./logs/voice_assistant.log"),
        
        # Output
        save_transcripts=os.getenv("SAVE_TRANSCRIPTS", "true").lower() == "true",
        transcript_output_dir=os.getenv("TRANSCRIPT_OUTPUT_DIR", "./outputs"),
    )
    
    return config
