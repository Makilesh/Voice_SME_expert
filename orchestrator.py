"""Coordinates all components for real-time operation."""
import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime
import time

from config import AppConfig
from src.audio import (
    MicrophoneCapture,
    VirtualAudioCapture,
    AudioPreprocessor,
    AudioBuffer
)
from src.utils import MemoryMonitor, PerformanceTracker, ThreadManager

logger = logging.getLogger(__name__)


class MeetingOrchestrator:
    """
    Coordinates all components for the voice assistant.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initializes all components.
        
        Parameters:
            config: Application configuration
        """
        self.config = config
        self._is_running = False
        self._start_time: Optional[datetime] = None
        
        # Initialize utilities
        self.memory_monitor = MemoryMonitor()
        self.performance_tracker = PerformanceTracker()
        self.thread_manager = ThreadManager()
        
        # Initialize audio components
        self.audio_capture: Optional[MicrophoneCapture] = None
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=config.sample_rate,
            target_db=-20.0
        )
        self.audio_buffer = AudioBuffer(
            max_duration_seconds=config.buffer_duration,
            sample_rate=config.sample_rate
        )
        
        # Placeholder for other components (to be initialized when implemented)
        self.speaker_diarizer = None
        self.stt_handler = None
        self.wake_word_detector = None
        self.expert_agent = None
        self.tts_handler = None
        self.meeting_connector = None
        
        # Stats
        self._audio_chunks_processed = 0
        self._wake_word_detections = 0
        self._expert_queries = 0
        
        logger.info("MeetingOrchestrator initialized")
    
    async def start(self, audio_source: str) -> None:
        """
        Starts the orchestrator.
        
        Parameters:
            audio_source: Audio source specification (e.g., "microphone:default")
        """
        if self._is_running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info(f"Starting orchestrator with audio source: {audio_source}")
        self._start_time = datetime.now()
        
        try:
            # Parse audio source
            source_type, *source_args = audio_source.split(":")
            
            # Initialize audio capture based on source
            if source_type == "microphone":
                device = source_args[0] if source_args else None
                device_idx = None if device == "default" else int(device) if device.isdigit() else None
                
                self.audio_capture = MicrophoneCapture(
                    sample_rate=self.config.sample_rate,
                    chunk_size=self.config.chunk_size,
                    device_index=device_idx
                )
                logger.info(f"Using microphone device: {device}")
            
            elif source_type == "virtual":
                application = source_args[0] if source_args else None
                
                self.audio_capture = VirtualAudioCapture(
                    sample_rate=self.config.sample_rate,
                    chunk_size=self.config.chunk_size,
                    loopback_device=application
                )
                logger.info(f"Using virtual audio capture for: {application}")
            
            else:
                logger.error(f"Unsupported audio source type: {source_type}")
                return
            
            # Start audio processing
            self._is_running = True
            
            # Start background task for audio processing
            self.thread_manager.start_background_task(
                self.process_audio_stream,
                "audio_processing"
            )
            
            logger.info("=== System Ready ===")
            logger.info(f"Wake phrase: {', '.join(self.config.wake_phrases)}")
            logger.info(f"Mode: {source_type}")
            logger.info("==================")
        
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            self._is_running = False
            raise
    
    async def stop(self) -> None:
        """
        Stops all components gracefully.
        """
        if not self._is_running:
            return
        
        logger.info("Stopping orchestrator...")
        self._is_running = False
        
        try:
            # Stop audio capture
            if self.audio_capture and self.audio_capture.is_capturing():
                self.audio_capture.stop_capture()
            
            # Stop all background tasks
            await self.thread_manager.stop_all(timeout=5.0)
            
            # Print final stats
            self._print_session_summary()
            
            logger.info("Orchestrator stopped successfully")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def process_audio_stream(self) -> None:
        """
        Main processing loop for audio.
        """
        logger.info("Starting audio processing loop")
        
        try:
            async for audio_chunk in self.audio_capture.get_audio_stream():
                if not self._is_running:
                    break
                
                # Start performance timing
                self.performance_tracker.start_timer("audio_processing")
                
                # Preprocess audio
                processed_audio = self.audio_preprocessor.preprocess(
                    audio_chunk,
                    normalize=True,
                    remove_silence=False,
                    highpass=True
                )
                
                # Add to buffer
                self.audio_buffer.write(processed_audio, time.time())
                
                # Update stats
                self._audio_chunks_processed += 1
                
                # TODO: Pass to speaker diarization
                # TODO: Pass to wake word detection
                # TODO: Pass to transcription
                
                # Stop timer
                elapsed = self.performance_tracker.stop_timer("audio_processing")
                
                # Periodic memory check
                if self._audio_chunks_processed % 100 == 0:
                    mem_stats = self.memory_monitor.check_memory()
                    logger.debug(f"Memory: {mem_stats['process_mb']:.1f}MB")
        
        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}")
        
        finally:
            logger.info("Audio processing loop stopped")
    
    async def handle_wake_word_detected(self, audio_after_wake: bytes) -> None:
        """
        Handles wake word detection event.
        
        Parameters:
            audio_after_wake: Audio data after wake word
        """
        logger.info("[Wake Word Detected] Processing query...")
        self._wake_word_detections += 1
        
        # TODO: Capture and transcribe query
        # TODO: Call expert agent
        # TODO: Speak response
        
        logger.info("[Wake Word] Ready for next query")
    
    async def handle_expert_query(self, query: str, speaker_id: str) -> None:
        """
        Processes expert query and speaks response.
        
        Parameters:
            query: User query text
            speaker_id: Speaker identifier
        """
        logger.info(f"[Expert] Query from {speaker_id}: {query}")
        self._expert_queries += 1
        
        # TODO: Get meeting context
        # TODO: Call expert agent
        # TODO: Stream response to TTS
        
        logger.info("[Expert] Response complete")
    
    def get_meeting_status(self) -> Dict:
        """
        Returns current meeting status.
        
        Returns:
            dict: Status information
        """
        if not self._start_time:
            return {"status": "not_started"}
        
        runtime = (datetime.now() - self._start_time).total_seconds()
        
        status = {
            "status": "running" if self._is_running else "stopped",
            "runtime_seconds": runtime,
            "audio_chunks_processed": self._audio_chunks_processed,
            "wake_word_detections": self._wake_word_detections,
            "expert_queries": self._expert_queries,
            "audio_buffer_duration": self.audio_buffer.get_duration(),
            "memory": self.memory_monitor.check_memory(),
            "active_tasks": self.thread_manager.get_active_count()
        }
        
        return status
    
    def _print_session_summary(self) -> None:
        """Print session summary."""
        if not self._start_time:
            return
        
        runtime = (datetime.now() - self._start_time).total_seconds()
        minutes = int(runtime // 60)
        seconds = int(runtime % 60)
        
        logger.info("=" * 50)
        logger.info("Meeting Session Complete")
        logger.info(f"Duration: {minutes} min {seconds} sec")
        logger.info("=" * 50)
        logger.info(f"Audio chunks processed: {self._audio_chunks_processed}")
        logger.info(f"Wake word detections: {self._wake_word_detections}")
        logger.info(f"Expert queries: {self._expert_queries}")
        logger.info("=" * 50)
        
        # Performance summary
        perf_summary = self.performance_tracker.get_summary()
        if perf_summary:
            logger.info("Performance Metrics:")
            for name, stats in perf_summary.items():
                logger.info(f"  {name}: {stats['mean']*1000:.2f}ms avg")
        
        logger.info("=" * 50)
