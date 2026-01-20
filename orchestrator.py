"""Coordinates all components for real-time operation with full-duplex support."""
import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime
import time
import threading

from config import AppConfig
from src.audio import (
    MicrophoneCapture,
    VirtualAudioCapture,
    AudioPreprocessor,
    AudioBuffer
)
from src.utils import MemoryMonitor, PerformanceTracker, ThreadManager

logger = logging.getLogger(__name__)


# Turn timing stats for performance monitoring
class TurnStats:
    """Track conversation turn timing."""
    __slots__ = ('stt_time', 'llm_time', 'tts_time', 'total_time', 'was_barge_in')
    
    def __init__(self):
        self.stt_time: float = 0.0
        self.llm_time: float = 0.0
        self.tts_time: float = 0.0
        self.total_time: float = 0.0
        self.was_barge_in: bool = False


class MeetingOrchestrator:
    """
    Coordinates all components for the voice assistant with full-duplex support.
    Supports barge-in detection for natural conversation flow.
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
        self._shutdown_event = threading.Event()
        
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
        self._barge_in_count = 0
        self._turn_stats: list = []
        self._consecutive_errors = 0
        
        logger.info("MeetingOrchestrator initialized with full-duplex support")
    
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
        Stops all components gracefully with proper shutdown sequence.
        """
        if not self._is_running:
            return
        
        logger.info("Stopping orchestrator...")
        self._is_running = False
        self._shutdown_event.set()
        
        try:
            # Stop TTS handler first (cleanest shutdown)
            if self.tts_handler and hasattr(self.tts_handler, 'shutdown'):
                try:
                    self.tts_handler.shutdown()
                except Exception as e:
                    logger.warning(f"TTS shutdown error: {e}")
            
            # Stop STT handler
            if self.stt_handler and hasattr(self.stt_handler, 'stop'):
                try:
                    self.stt_handler.stop()
                except Exception as e:
                    logger.warning(f"STT shutdown error: {e}")
            
            # Stop LLM handler
            if self.expert_agent and hasattr(self.expert_agent, 'shutdown'):
                try:
                    await self.expert_agent.shutdown()
                except Exception as e:
                    logger.warning(f"LLM shutdown error: {e}")
            
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
        Processes expert query and speaks response with full-duplex support.
        
        Parameters:
            query: User query text
            speaker_id: Speaker identifier
        """
        turn_stats = TurnStats()
        turn_start = time.time()
        
        logger.info(f"[Expert] Query from {speaker_id}: {query}")
        self._expert_queries += 1
        
        try:
            # Step 1: Get LLM response
            llm_start = time.time()
            # TODO: Get meeting context and call expert agent
            # response = await self.expert_agent.process_with_history(query, context)
            turn_stats.llm_time = time.time() - llm_start
            
            # Step 2: Speak response with barge-in monitoring
            if self.tts_handler:
                tts_start = time.time()
                # Clear STT buffer before speaking (critical for barge-in)
                if self.stt_handler and hasattr(self.stt_handler, 'clear_realtime_text'):
                    self.stt_handler.clear_realtime_text()
                
                # TODO: self.tts_handler.speak(response, enable_barge_in=True)
                # completed = self.tts_handler.wait_for_completion(timeout=30.0)
                
                turn_stats.tts_time = time.time() - tts_start
                
                # Step 3: Handle barge-in if detected
                if self.tts_handler.was_barge_in():
                    turn_stats.was_barge_in = True
                    self._barge_in_count += 1
                    logger.info("🎤 Barge-in detected - processing interruption")
                    
                    # Get interruption text from real-time STT
                    if self.stt_handler and hasattr(self.stt_handler, 'get_realtime_text'):
                        interruption_text = self.stt_handler.get_realtime_text()
                        if interruption_text and len(interruption_text) > 2:
                            logger.info(f"📝 Interruption: {interruption_text}")
                            # Recursively process the interruption
                            await self.handle_expert_query(interruption_text, speaker_id)
            
            turn_stats.total_time = time.time() - turn_start
            self._turn_stats.append(turn_stats)
            
            # Log turn timing
            logger.info(f"⏱ Turn timing: LLM={turn_stats.llm_time*1000:.0f}ms, "
                       f"TTS={turn_stats.tts_time*1000:.0f}ms, "
                       f"Total={turn_stats.total_time*1000:.0f}ms"
                       f"{' [BARGE-IN]' if turn_stats.was_barge_in else ''}")
            
            if turn_stats.total_time > 3.0:
                logger.warning(f"⚠️ Slow turn: {turn_stats.total_time:.1f}s")
            
            # Reset error counter on success
            self._consecutive_errors = 0
            
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(f"[Expert] Error processing query: {e}")
            
            # Check for circuit breaker
            max_errors = getattr(self.config, 'max_consecutive_errors', 3)
            if self._consecutive_errors >= max_errors:
                logger.error(f"❌ Circuit breaker: {self._consecutive_errors} consecutive errors")
        
        logger.info("[Expert] Response complete")
    
    def get_meeting_status(self) -> Dict:
        """
        Returns current meeting status with barge-in stats.
        
        Returns:
            dict: Status information
        """
        if not self._start_time:
            return {"status": "not_started"}
        
        runtime = (datetime.now() - self._start_time).total_seconds()
        
        # Calculate turn statistics
        avg_turn_time = 0.0
        if self._turn_stats:
            avg_turn_time = sum(t.total_time for t in self._turn_stats) / len(self._turn_stats)
        
        status = {
            "status": "running" if self._is_running else "stopped",
            "runtime_seconds": runtime,
            "audio_chunks_processed": self._audio_chunks_processed,
            "wake_word_detections": self._wake_word_detections,
            "expert_queries": self._expert_queries,
            "barge_in_count": self._barge_in_count,
            "avg_turn_time_ms": avg_turn_time * 1000,
            "consecutive_errors": self._consecutive_errors,
            "audio_buffer_duration": self.audio_buffer.get_duration(),
            "memory": self.memory_monitor.check_memory(),
            "active_tasks": self.thread_manager.get_active_count()
        }
        
        return status
    
    def _print_session_summary(self) -> None:
        """Print session summary with barge-in and turn stats."""
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
        logger.info(f"Barge-in interruptions: {self._barge_in_count}")
        logger.info("=" * 50)
        
        # Turn timing summary
        if self._turn_stats:
            avg_total = sum(t.total_time for t in self._turn_stats) / len(self._turn_stats)
            avg_llm = sum(t.llm_time for t in self._turn_stats) / len(self._turn_stats)
            avg_tts = sum(t.tts_time for t in self._turn_stats) / len(self._turn_stats)
            barge_in_pct = (self._barge_in_count / len(self._turn_stats)) * 100 if self._turn_stats else 0
            
            logger.info("Turn Timing:")
            logger.info(f"  Average total: {avg_total*1000:.0f}ms")
            logger.info(f"  Average LLM: {avg_llm*1000:.0f}ms")
            logger.info(f"  Average TTS: {avg_tts*1000:.0f}ms")
            logger.info(f"  Barge-in rate: {barge_in_pct:.1f}%")
        
        # Performance summary
        perf_summary = self.performance_tracker.get_summary()
        if perf_summary:
            logger.info("Performance Metrics:")
            for name, stats in perf_summary.items():
                logger.info(f"  {name}: {stats['mean']*1000:.2f}ms avg")
        
        # Component stats
        if self.stt_handler and hasattr(self.stt_handler, 'get_performance_stats'):
            stt_stats = self.stt_handler.get_performance_stats()
            logger.info(f"STT Stats: {stt_stats}")
        
        if hasattr(self, 'tts_handler') and self.tts_handler:
            if hasattr(self.tts_handler, 'engine') and hasattr(self.tts_handler.engine, 'get_stats'):
                tts_stats = self.tts_handler.engine.get_stats()
                logger.info(f"TTS Stats: {tts_stats}")
        
        logger.info("=" * 50)
