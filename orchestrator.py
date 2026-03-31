"""Coordinates all components for real-time operation with full-duplex support."""
import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime
import time
import threading
import numpy as np

from config import AppConfig
from src.audio import (
    MicrophoneCapture,
    VirtualAudioCapture,
    AudioPreprocessor,
    AudioBuffer
)
from src.diarization import SpeakerDiarizer
from src.transcription import STTHandler
from src.wake_word import WakeWordDetector
from src.expert import ExpertAgent, KnowledgeBase, ContextManager, PromptBuilder, RAGRetriever
from src.llm import LLMHandler
from src.tts import TTSHandler
from src.meeting import ZoomConnector, MeetConnector, TeamsConnector
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
        
        # --- Initialize Speaker Diarization ---
        self.speaker_diarizer = SpeakerDiarizer(
            embedding_model=config.speaker_embedding_model,
            similarity_threshold=config.speaker_similarity_threshold,
            max_speakers=config.max_speakers,
            sample_rate=config.sample_rate,
        )
        
        # --- Initialize STT ---
        self.stt_handler = STTHandler(
            model=config.stt_model,
            language=config.stt_language,
            compute_type=config.stt_compute_type,
            sample_rate=config.sample_rate,
            mode=getattr(config, 'stt_mode', 'accurate'),
            transcription_timeout=getattr(config, 'stt_transcription_timeout', 30.0),
        )
        
        # --- Initialize Wake Word Detector ---
        self.wake_word_detector = WakeWordDetector(
            wake_phrases=config.wake_phrases,
            sensitivity=config.wake_word_sensitivity,
            model_path=config.wake_word_model_path,
            sample_rate=config.sample_rate,
        )
        
        # --- Initialize LLM Handler ---
        self.llm_handler = LLMHandler(
            openai_key=config.openai_api_key,
            openai_model=config.openai_model,
            gemini_key=config.gemini_api_key,
            gemini_model=config.gemini_model,
            ollama_url=config.ollama_base_url,
            ollama_model=config.ollama_model,
            provider_priority=config.llm_provider_chain,
            max_retries=getattr(config, 'max_consecutive_errors', 3),
            retry_base_delay=getattr(config, 'retry_base_delay', 1.0),
            retry_max_delay=getattr(config, 'retry_max_delay', 10.0),
        )
        
        # --- Initialize Expert Agent (Knowledge + RAG + LLM) ---
        self.knowledge_base = KnowledgeBase(
            persist_directory=config.embeddings_path,
        )
        self.context_manager = ContextManager(
            max_context_tokens=config.max_context_tokens,
        )
        self.rag_retriever = RAGRetriever(
            knowledge_base=self.knowledge_base,
            max_context_length=config.max_context_tokens,
        )
        self.prompt_builder = PromptBuilder()
        self.expert_agent = ExpertAgent(
            knowledge_base=self.knowledge_base,
            llm_handler=self.llm_handler,
            rag_retriever=self.rag_retriever,
            context_manager=self.context_manager,
            prompt_builder=self.prompt_builder,
        )
        
        # --- Initialize TTS Handler (with barge-in wired to STT) ---
        self.tts_handler = TTSHandler(
            cartesia_api_key=config.cartesia_api_key,
            cartesia_voice_id=config.cartesia_voice_id,
            cartesia_model=config.cartesia_model,
            sample_rate=config.cartesia_sample_rate,
            stt_handler=self.stt_handler,
            barge_in_enabled=config.tts_barge_in_enabled,
            barge_in_startup_buffer=config.tts_barge_in_startup_buffer,
            barge_in_check_interval=config.tts_barge_in_check_interval,
            barge_in_min_chars=config.tts_barge_in_min_chars,
        )
        
        # Meeting connector (lazily initialized based on mode)
        self.meeting_connector = None
        
        # State flags
        self._wake_word_active = False
        self._processing_query = False
        
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
            source_detail = source_args[0] if source_args else None
            
            # Initialize audio capture based on source
            if source_type == "microphone":
                device_idx = None
                if source_detail and source_detail != "default":
                    try:
                        device_idx = int(source_detail)
                    except ValueError:
                        device_idx = None
                
                self.audio_capture = MicrophoneCapture(
                    sample_rate=self.config.sample_rate,
                    chunk_size=self.config.chunk_size,
                    device_index=device_idx
                )
                logger.info(f"Using microphone device: {source_detail or 'default'}")
            
            elif source_type == "virtual":
                self.audio_capture = VirtualAudioCapture(
                    sample_rate=self.config.sample_rate,
                    chunk_size=self.config.chunk_size,
                    loopback_device=source_detail
                )
                logger.info(f"Using virtual audio capture for: {source_detail or 'auto-detect'}")
            
            elif source_type == "zoom":
                # For Zoom, initialise the meeting connector and fall back to virtual audio
                self.meeting_connector = ZoomConnector(
                    sdk_key=self.config.zoom_sdk_key,
                    sdk_secret=self.config.zoom_sdk_secret,
                    user_name=self.config.meeting_display_name,
                )
                meeting_id = source_detail or ""
                if meeting_id:
                    await self.meeting_connector.connect(meeting_id)
                    await self.meeting_connector.join_meeting(meeting_id)
                
                # Use virtual audio to capture Zoom's output
                self.audio_capture = VirtualAudioCapture(
                    sample_rate=self.config.sample_rate,
                    chunk_size=self.config.chunk_size,
                )
                logger.info(f"Zoom mode — meeting {meeting_id}, using virtual audio capture")
            
            elif source_type == "meet":
                self.meeting_connector = MeetConnector(
                    user_name=self.config.meeting_display_name,
                )
                meeting_url = source_detail or ""
                if meeting_url:
                    await self.meeting_connector.connect(meeting_url)
                    await self.meeting_connector.join_meeting(meeting_url)
                
                self.audio_capture = VirtualAudioCapture(
                    sample_rate=self.config.sample_rate,
                    chunk_size=self.config.chunk_size,
                )
                logger.info(f"Google Meet mode — {meeting_url}, using virtual audio capture")
            
            elif source_type == "teams":
                self.meeting_connector = TeamsConnector(
                    user_name=self.config.meeting_display_name,
                )
                meeting_url = source_detail or ""
                if meeting_url:
                    await self.meeting_connector.connect(meeting_url)
                    await self.meeting_connector.join_meeting(meeting_url)
                
                self.audio_capture = VirtualAudioCapture(
                    sample_rate=self.config.sample_rate,
                    chunk_size=self.config.chunk_size,
                )
                logger.info(f"Teams mode — {meeting_url}, using virtual audio capture")
            
            else:
                raise ValueError(f"Unsupported audio source type: {source_type}")
            
            # Start audio processing
            self._is_running = True
            
            # Start the main audio processing pipeline as a background task
            self.thread_manager.start_background_task(
                self.process_audio_stream,
                "audio_processing"
            )
            
            logger.info("=== System Ready ===")
            logger.info(f"Wake phrases: {', '.join(self.config.wake_phrases)}")
            logger.info(f"Mode: {source_type}")
            logger.info(f"LLM chain: {' -> '.join(self.config.llm_provider_chain)}")
            logger.info(f"TTS: {'Cartesia' if self.config.cartesia_api_key else 'Fallback(pyttsx3)'}")
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
            # 1. Stop TTS playback first (immediate silence)
            try:
                self.tts_handler.stop_playback()
            except Exception as e:
                logger.warning(f"TTS shutdown error: {e}")
            
            # 2. Stop STT streaming
            try:
                self.stt_handler.stop_streaming()
            except Exception as e:
                logger.warning(f"STT shutdown error: {e}")
            
            # 3. Stop wake word detection
            try:
                self.wake_word_detector.stop_detection()
            except Exception as e:
                logger.warning(f"Wake word shutdown error: {e}")
            
            # 4. Shutdown LLM handler (close async clients)
            try:
                await self.llm_handler.shutdown()
            except Exception as e:
                logger.warning(f"LLM shutdown error: {e}")
            
            # 5. Leave meeting if connected
            if self.meeting_connector:
                try:
                    await self.meeting_connector.disconnect()
                except Exception as e:
                    logger.warning(f"Meeting disconnect error: {e}")
            
            # 6. Stop audio capture
            if self.audio_capture and self.audio_capture.is_capturing():
                self.audio_capture.stop_capture()
            
            # 7. Stop all background tasks
            await self.thread_manager.stop_all(timeout=5.0)
            
            # 8. Export transcript if configured
            if self.config.save_transcripts:
                try:
                    self.stt_handler.export_transcript(
                        format="json",
                        output_path=self.config.transcript_output_dir
                    )
                    logger.info(f"Transcript saved to {self.config.transcript_output_dir}")
                except Exception as e:
                    logger.warning(f"Transcript export error: {e}")
            
            # Print final stats
            self._print_session_summary()
            
            logger.info("Orchestrator stopped successfully")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def process_audio_stream(self) -> None:
        """
        Main processing loop for audio.
        
        Pipeline per chunk:
          1. Preprocess audio (highpass, normalise)
          2. Write to ring-buffer
          3. Feed to wake-word detector
          4. Feed to speaker diarizer
          5. Feed to STT (accumulates internally, emits transcripts)
          6. On wake-word → capture query window → expert response → TTS
        """
        logger.info("Starting audio processing loop")
        
        # Accumulation buffer for STT (collect ~1 s before transcribing)
        stt_buffer: list = []
        stt_buffer_start: float = time.time()
        stt_min_duration: float = 1.0  # seconds
        
        try:
            async for audio_chunk in self.audio_capture.get_audio_stream():
                if not self._is_running:
                    break
                
                # --- timing ---
                self.performance_tracker.start_timer("audio_processing")
                now = time.time()
                
                # 1. Preprocess
                processed_audio = self.audio_preprocessor.preprocess(
                    audio_chunk,
                    normalize=True,
                    remove_silence=False,
                    highpass=True,
                )
                
                # 2. Ring-buffer
                self.audio_buffer.write(processed_audio, now)
                self._audio_chunks_processed += 1
                
                # 3. Wake-word detection (unless already handling a query)
                if not self._processing_query:
                    detected, confidence, phrase = self.wake_word_detector.is_wake_word(processed_audio)
                    if detected:
                        self._wake_word_detections += 1
                        logger.info(f"[Wake Word] '{phrase}' detected (conf={confidence:.2f})")
                        
                        # Grab the last ~3 s of audio as the query window
                        query_audio = self.audio_buffer.get_latest(3.0)
                        
                        # Spawn query handling without blocking the audio loop
                        asyncio.create_task(
                            self._handle_wake_word_query(query_audio, now)
                        )
                
                # 4. Speaker diarization (runs on every chunk)
                diar_segments = self.speaker_diarizer.process_audio(processed_audio, now)
                
                # 5. Accumulate audio for STT, flush when we have >= 1 s
                stt_buffer.extend(processed_audio.tolist())
                stt_buffer_duration = len(stt_buffer) / self.config.sample_rate
                
                if stt_buffer_duration >= stt_min_duration:
                    stt_audio = np.array(stt_buffer, dtype=np.float32)
                    
                    # Determine speaker from most recent diarisation segment
                    speaker_id = "Speaker-1"
                    speaker_name = None
                    if diar_segments:
                        seg = diar_segments[-1]
                        speaker_id = seg['speaker_id']
                        speaker_name = self.speaker_diarizer.get_speaker_name(speaker_id) or None
                    
                    # Transcribe
                    result = self.stt_handler.transcribe_segment(
                        stt_audio, speaker_id, speaker_name, stt_buffer_start
                    )
                    
                    if result['text']:
                        # Feed transcript into expert context for future queries
                        self.expert_agent.add_transcript(
                            speaker=result['speaker_name'],
                            content=result['text'],
                        )
                        logger.debug(
                            f"[{result['speaker_name']}] {result['text'][:80]}"
                        )
                    
                    stt_buffer = []
                    stt_buffer_start = time.time()
                
                # --- timing ---
                elapsed = self.performance_tracker.stop_timer("audio_processing")
                
                # Periodic memory check
                if self._audio_chunks_processed % 500 == 0:
                    mem = self.memory_monitor.check_memory()
                    logger.debug(f"Memory: {mem.get('process_mb', 0):.1f}MB | "
                                 f"Chunks: {self._audio_chunks_processed}")
        
        except asyncio.CancelledError:
            logger.info("Audio processing task cancelled")
        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}", exc_info=True)
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
        
        if isinstance(audio_after_wake, (bytes, bytearray)):
            audio_np = np.frombuffer(audio_after_wake, dtype=np.float32)
        else:
            audio_np = audio_after_wake
        
        await self._handle_wake_word_query(audio_np, time.time())
    
    # ------------------------------------------------------------------
    # Internal: full query pipeline  (wake → STT → Expert → TTS)
    # ------------------------------------------------------------------
    async def _handle_wake_word_query(
        self, query_audio: np.ndarray, timestamp: float
    ) -> None:
        """
        End-to-end pipeline triggered after a wake word.
        
        1. Transcribe the captured audio window to get the query text.
        2. Send the query to the ExpertAgent (RAG + LLM).
        3. Speak the response via TTS (with barge-in monitoring).
        """
        if self._processing_query:
            logger.debug("Already processing a query — skipping")
            return
        
        self._processing_query = True
        turn = TurnStats()
        turn_start = time.time()
        
        try:
            # --- Step 1: Transcribe the query audio ---
            stt_start = time.time()
            speaker_id, _confidence = self.speaker_diarizer.get_speaker_for_segment(query_audio)
            speaker_name = self.speaker_diarizer.get_speaker_name(speaker_id) or speaker_id
            result = self.stt_handler.transcribe_segment(
                query_audio, speaker_id, speaker_name, timestamp
            )
            query_text = result.get('text', '').strip()
            turn.stt_time = time.time() - stt_start
            
            if not query_text:
                logger.info("[Wake Word] No speech detected in query window")
                return
            
            logger.info(f"[Query] ({speaker_name}): {query_text}")
            
            # --- Step 2: Expert Agent → LLM response ---
            await self.handle_expert_query(query_text, speaker_id, _turn=turn)
        
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(f"[Wake Word] Pipeline error: {e}", exc_info=True)
        finally:
            self._processing_query = False
        
        logger.info("[Wake Word] Ready for next query")
    
    async def handle_expert_query(
        self, query: str, speaker_id: str, *, _turn: Optional[TurnStats] = None
    ) -> None:
        """
        Processes expert query and speaks response with full-duplex support.
        
        Parameters:
            query: User query text
            speaker_id: Speaker identifier
            _turn: Optional pre-existing TurnStats (reused from wake-word path)
        """
        turn = _turn or TurnStats()
        turn_start = time.time()
        
        speaker_name = self.speaker_diarizer.get_speaker_name(speaker_id) or speaker_id
        logger.info(f"[Expert] Query from {speaker_name}: {query}")
        self._expert_queries += 1
        
        try:
            # Step 1: Get LLM response via Expert Agent (includes RAG)
            llm_start = time.time()
            response = await self.expert_agent.process_query(
                query=query,
                speaker=speaker_name,
                use_rag=True,
                use_context=True,
            )
            turn.llm_time = time.time() - llm_start
            
            if not response:
                logger.warning("[Expert] Empty response from LLM")
                return
            
            logger.info(f"[Expert] Response ({len(response)} chars): {response[:100]}...")
            
            # Step 2: Speak response with barge-in monitoring
            tts_start = time.time()
            
            # Clear STT real-time buffer before speaking (critical for barge-in)
            self.stt_handler.clear_realtime_text()
            
            # Synthesise + play
            await self.tts_handler.speak(response)
            
            # Block until playback finishes or barge-in interrupts
            completed = self.tts_handler.wait_for_completion(
                timeout=self.config.tts_playback_timeout
            )
            turn.tts_time = time.time() - tts_start
            
            # Step 3: Handle barge-in if detected
            if not completed and self.tts_handler.was_barge_in():
                turn.was_barge_in = True
                self._barge_in_count += 1
                logger.info("Barge-in detected — processing interruption")
                
                interruption_text = self.stt_handler.get_realtime_text()
                if interruption_text and len(interruption_text.strip()) > 2:
                    logger.info(f"Interruption: {interruption_text}")
                    # Recursively handle the interruption as a new query
                    await self.handle_expert_query(interruption_text, speaker_id)
            
            turn.total_time = time.time() - turn_start
            self._turn_stats.append(turn)
            
            # Log turn timing
            logger.info(
                f"Turn timing: STT={turn.stt_time*1000:.0f}ms, "
                f"LLM={turn.llm_time*1000:.0f}ms, "
                f"TTS={turn.tts_time*1000:.0f}ms, "
                f"Total={turn.total_time*1000:.0f}ms"
                f"{' [BARGE-IN]' if turn.was_barge_in else ''}"
            )
            
            if turn.total_time > 3.0:
                logger.warning(f"Slow turn: {turn.total_time:.1f}s")
            
            # Reset error counter on success
            self._consecutive_errors = 0
            
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(f"[Expert] Error processing query: {e}", exc_info=True)
            
            max_errors = getattr(self.config, 'max_consecutive_errors', 3)
            if self._consecutive_errors >= max_errors:
                logger.error(
                    f"Circuit breaker: {self._consecutive_errors} consecutive errors"
                )
        
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
