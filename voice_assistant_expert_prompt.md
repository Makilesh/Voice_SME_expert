SUBJECT MATTER EXPERT VOICE ASSISTANT - PROBLEM DESCRIPTION
===========================================================

PROBLEM STATEMENT
-----------------

Build a real-time voice assistant that acts as a Subject Matter Expert (SME) capable of:

1. Joining virtual meetings on Zoom, Google Meet, and Microsoft Teams as a silent participant
2. Capturing audio from physical/real-time meetings using microphone input
3. Performing real-time speaker diarization to identify and separate individual speakers
4. Transcribing each speaker's audio with speaker labels (Speaker 1, Speaker 2, etc. or names if available)
5. Accumulating all transcribed content as meeting context and knowledge base
6. Listening for a configurable wake word (e.g., "Hey Expert", "Okay Assistant")
7. When activated by wake word, responding as a subject matter expert using:
   - Pre-loaded domain knowledge from documents, PDFs, and knowledge bases
   - All meeting context collected during the session
   - Conversation history and speaker contributions
8. Supporting full-duplex audio for natural interruption handling
9. Providing low-latency responses (under 500ms end-to-end)

The system should run locally with optional cloud LLM fallback, support multiple concurrent speakers, and maintain speaker identity across the meeting duration.


CORE REQUIREMENTS
-----------------

Audio Capture Requirements:
- Virtual audio capture from meeting applications (Zoom, Meet, Teams)
- Physical microphone capture for in-person meetings
- Sample rate: 16kHz mono for speech processing
- Buffer size optimized for low latency (20-50ms chunks)

Speaker Diarization Requirements:
- Real-time speaker separation with under 200ms latency
- Support for 2-10 concurrent speakers
- Speaker embedding extraction for identity persistence
- Optional speaker name assignment via voice enrollment

Transcription Requirements:
- Real-time streaming transcription
- Per-speaker transcription with timestamps
- Support for technical vocabulary and domain terms
- Automatic punctuation and formatting

Wake Word Detection Requirements:
- Always-on low-power wake word detection
- Configurable wake phrases
- False positive rate under 1 per hour
- Detection latency under 100ms

Expert Response Requirements:
- RAG (Retrieval Augmented Generation) for document knowledge
- Meeting context integration
- Multi-turn conversation support
- Concise voice-optimized responses


FILE STRUCTURE
--------------

voice_expert_assistant/
    .env.example
    .env
    .gitignore
    requirements.txt
    README.md
    setup.py
    
    config/
        __init__.py
        settings.py
        audio_config.py
        llm_config.py
        wake_word_config.py
    
    src/
        __init__.py
        main.py
        orchestrator.py
        
        audio/
            __init__.py
            audio_capture.py
            virtual_audio_capture.py
            microphone_capture.py
            audio_preprocessor.py
            audio_buffer.py
        
        diarization/
            __init__.py
            speaker_diarizer.py
            speaker_embeddings.py
            speaker_tracker.py
            voice_enrollment.py
        
        transcription/
            __init__.py
            stt_handler.py
            streaming_transcriber.py
            transcript_store.py
        
        wake_word/
            __init__.py
            wake_word_detector.py
            keyword_spotter.py
        
        expert/
            __init__.py
            expert_agent.py
            knowledge_base.py
            rag_retriever.py
            context_manager.py
            prompt_builder.py
        
        llm/
            __init__.py
            llm_handler.py
            openai_provider.py
            gemini_provider.py
            ollama_provider.py
            response_streamer.py
        
        tts/
            __init__.py
            tts_handler.py
            cartesia_engine.py
            audio_player.py
        
        meeting/
            __init__.py
            meeting_connector.py
            zoom_connector.py
            meet_connector.py
            teams_connector.py
        
        utils/
            __init__.py
            logger.py
            memory_monitor.py
            performance_tracker.py
            thread_manager.py
    
    knowledge/
        documents/
        embeddings/
        index/
    
    models/
        wake_word/
        speaker_embedding/
    
    tests/
        __init__.py
        test_audio_capture.py
        test_diarization.py
        test_transcription.py
        test_wake_word.py
        test_expert.py
        test_integration.py
    
    scripts/
        enroll_speaker.py
        train_wake_word.py
        index_knowledge.py
        test_audio_device.py


PURPOSE OF FILES AND IMPORTANT METHODS
--------------------------------------

CONFIG FILES
............

config/settings.py
    Purpose: Central configuration management with validation and environment variable loading
    
    class AppConfig
        Parameters: None (loads from environment)
        Returns: Validated configuration object
        Description: Dataclass containing all application settings
    
    def load_config()
        Parameters: config_path (optional string path to config file)
        Returns: AppConfig instance
        Description: Loads and validates configuration from environment and files

config/audio_config.py
    Purpose: Audio-specific configuration for capture, processing, and output
    
    class AudioCaptureConfig
        Parameters: sample_rate (int), channels (int), chunk_size (int), buffer_duration (float)
        Description: Configuration for audio input devices
    
    class AudioOutputConfig
        Parameters: sample_rate (int), channels (int), output_device (string or None)
        Description: Configuration for audio output/playback

config/llm_config.py
    Purpose: LLM provider configuration with fallback chain
    
    class LLMProviderConfig
        Parameters: provider_name (string), api_key (string), model (string), max_tokens (int), temperature (float)
        Description: Per-provider configuration settings
    
    def get_provider_chain()
        Parameters: None
        Returns: List of LLMProviderConfig in priority order
        Description: Returns ordered list of LLM providers for fallback

config/wake_word_config.py
    Purpose: Wake word detection configuration
    
    class WakeWordConfig
        Parameters: wake_phrases (list of strings), sensitivity (float 0-1), model_path (string)
        Description: Configuration for wake word detection system


AUDIO CAPTURE FILES
...................

src/audio/audio_capture.py
    Purpose: Abstract base class for audio capture with common interface
    
    class AudioCaptureBase
        def start_capture()
            Parameters: callback (callable receiving audio chunks)
            Returns: None
            Description: Starts audio capture and calls callback with each chunk
        
        def stop_capture()
            Parameters: None
            Returns: None
            Description: Stops audio capture and releases resources
        
        def get_audio_stream()
            Parameters: None
            Returns: AsyncGenerator yielding audio chunks
            Description: Async generator for streaming audio data
        
        def set_device()
            Parameters: device_id (int or string)
            Returns: bool success
            Description: Sets the audio input device

src/audio/virtual_audio_capture.py
    Purpose: Captures system audio from virtual meetings (Zoom, Meet, Teams)
    
    class VirtualAudioCapture extends AudioCaptureBase
        def __init__()
            Parameters: config (AudioCaptureConfig), loopback_device (string or None)
            Description: Initializes virtual audio capture using system loopback
        
        def detect_meeting_audio()
            Parameters: None
            Returns: string device name or None
            Description: Auto-detects audio output from meeting applications
        
        def capture_application_audio()
            Parameters: application_name (string)
            Returns: AsyncGenerator yielding audio chunks
            Description: Captures audio specifically from named application

src/audio/microphone_capture.py
    Purpose: Captures audio from physical microphone for in-person meetings
    
    class MicrophoneCapture extends AudioCaptureBase
        def __init__()
            Parameters: config (AudioCaptureConfig), device_index (int or None)
            Description: Initializes microphone capture
        
        def list_devices()
            Parameters: None
            Returns: List of dict with device info
            Description: Returns available microphone devices
        
        def set_noise_reduction()
            Parameters: enabled (bool), aggressiveness (int 0-3)
            Returns: None
            Description: Enables/configures noise reduction

src/audio/audio_preprocessor.py
    Purpose: Preprocesses audio for optimal diarization and transcription
    
    class AudioPreprocessor
        def __init__()
            Parameters: sample_rate (int), target_db (float)
            Description: Initializes preprocessor with target parameters
        
        def normalize()
            Parameters: audio_chunk (numpy array)
            Returns: numpy array normalized audio
            Description: Normalizes audio to target dB level
        
        def remove_silence()
            Parameters: audio_chunk (numpy array), threshold_db (float)
            Returns: numpy array with silence removed, list of speech segments
            Description: Removes silent portions and returns speech segments
        
        def apply_vad()
            Parameters: audio_chunk (numpy array)
            Returns: list of tuples (start_ms, end_ms) speech segments
            Description: Applies Voice Activity Detection

src/audio/audio_buffer.py
    Purpose: Thread-safe circular buffer for audio streaming
    
    class AudioBuffer
        def __init__()
            Parameters: max_duration_seconds (float), sample_rate (int)
            Description: Creates circular buffer with specified capacity
        
        def write()
            Parameters: audio_chunk (numpy array), timestamp (float)
            Returns: bool success
            Description: Writes audio chunk to buffer
        
        def read()
            Parameters: duration_seconds (float)
            Returns: numpy array audio data
            Description: Reads specified duration from buffer
        
        def get_latest()
            Parameters: duration_seconds (float)
            Returns: numpy array most recent audio
            Description: Gets most recent audio without removing from buffer


DIARIZATION FILES
.................

src/diarization/speaker_diarizer.py
    Purpose: Real-time speaker diarization to separate and identify speakers
    
    class SpeakerDiarizer
        def __init__()
            Parameters: config (DiarizationConfig), embedding_model (string or path)
            Description: Initializes diarizer with speaker embedding model
        
        def process_audio()
            Parameters: audio_chunk (numpy array), timestamp (float)
            Returns: list of dict with speaker_id, start_time, end_time, embedding
            Description: Processes audio and returns speaker segments
        
        def get_speaker_for_segment()
            Parameters: audio_segment (numpy array)
            Returns: string speaker_id, float confidence
            Description: Identifies speaker for given audio segment
        
        def merge_short_segments()
            Parameters: segments (list), min_duration (float)
            Returns: list of merged segments
            Description: Merges short speaker segments to reduce fragmentation

src/diarization/speaker_embeddings.py
    Purpose: Extracts and manages speaker voice embeddings
    
    class SpeakerEmbeddingExtractor
        def __init__()
            Parameters: model_path (string), embedding_dim (int)
            Description: Loads speaker embedding model
        
        def extract_embedding()
            Parameters: audio_segment (numpy array)
            Returns: numpy array embedding vector
            Description: Extracts voice embedding from audio
        
        def compare_embeddings()
            Parameters: embedding1 (numpy array), embedding2 (numpy array)
            Returns: float similarity score 0-1
            Description: Computes cosine similarity between embeddings

src/diarization/speaker_tracker.py
    Purpose: Tracks and maintains speaker identities across meeting duration
    
    class SpeakerTracker
        def __init__()
            Parameters: similarity_threshold (float), max_speakers (int)
            Description: Initializes tracker with clustering parameters
        
        def update()
            Parameters: embedding (numpy array), timestamp (float)
            Returns: string speaker_id, bool is_new_speaker
            Description: Updates tracker with new embedding, returns speaker ID
        
        def get_speaker_stats()
            Parameters: speaker_id (string)
            Returns: dict with speaking_time, segment_count, last_active
            Description: Returns statistics for specified speaker
        
        def assign_name()
            Parameters: speaker_id (string), name (string)
            Returns: None
            Description: Assigns human-readable name to speaker ID
        
        def get_all_speakers()
            Parameters: None
            Returns: list of dict with speaker_id, name, embedding, stats
            Description: Returns all tracked speakers

src/diarization/voice_enrollment.py
    Purpose: Enrolls known speakers for identification
    
    class VoiceEnrollment
        def __init__()
            Parameters: embedding_extractor (SpeakerEmbeddingExtractor), storage_path (string)
            Description: Initializes enrollment system
        
        def enroll_speaker()
            Parameters: audio_samples (list of numpy arrays), speaker_name (string)
            Returns: string speaker_id, numpy array averaged_embedding
            Description: Enrolls speaker from multiple audio samples
        
        def load_enrolled_speakers()
            Parameters: None
            Returns: dict mapping speaker_id to embedding and name
            Description: Loads previously enrolled speakers
        
        def match_to_enrolled()
            Parameters: embedding (numpy array)
            Returns: string speaker_name or None, float confidence
            Description: Matches embedding against enrolled speakers


TRANSCRIPTION FILES
...................

src/transcription/stt_handler.py
    Purpose: Main STT handler coordinating transcription with speaker info
    
    class STTHandler
        def __init__()
            Parameters: config (STTConfig), model (string)
            Description: Initializes STT with specified model
        
        def transcribe_segment()
            Parameters: audio_segment (numpy array), speaker_id (string)
            Returns: dict with text, speaker_id, start_time, end_time, confidence
            Description: Transcribes audio segment with speaker attribution
        
        def start_streaming()
            Parameters: audio_stream (AsyncGenerator), diarization_callback (callable)
            Returns: AsyncGenerator yielding transcription results
            Description: Starts streaming transcription with speaker diarization
        
        def get_full_transcript()
            Parameters: None
            Returns: list of transcription dicts ordered by time
            Description: Returns complete meeting transcript

src/transcription/streaming_transcriber.py
    Purpose: Real-time streaming transcription with partial results
    
    class StreamingTranscriber
        def __init__()
            Parameters: model (string), language (string), compute_type (string)
            Description: Initializes streaming transcriber
        
        def feed_audio()
            Parameters: audio_chunk (numpy array)
            Returns: string partial_transcript or None
            Description: Feeds audio and returns partial transcript if available
        
        def get_final_transcript()
            Parameters: None
            Returns: string final transcript for current segment
            Description: Returns finalized transcript when speech ends
        
        def reset()
            Parameters: None
            Returns: None
            Description: Resets transcriber state for new segment

src/transcription/transcript_store.py
    Purpose: Stores and manages meeting transcripts with speaker labels
    
    class TranscriptStore
        def __init__()
            Parameters: max_entries (int), persistence_path (string or None)
            Description: Initializes transcript storage
        
        def add_entry()
            Parameters: text (string), speaker_id (string), timestamp (float), confidence (float)
            Returns: int entry_id
            Description: Adds transcription entry to store
        
        def get_recent()
            Parameters: duration_seconds (float), speaker_id (string or None)
            Returns: list of transcript entries
            Description: Gets recent transcripts, optionally filtered by speaker
        
        def get_by_speaker()
            Parameters: speaker_id (string)
            Returns: list of transcript entries for speaker
            Description: Gets all transcripts from specified speaker
        
        def get_context_window()
            Parameters: num_entries (int)
            Returns: string formatted context
            Description: Gets recent entries formatted for LLM context
        
        def search()
            Parameters: query (string), max_results (int)
            Returns: list of matching entries
            Description: Searches transcripts for relevant content
        
        def export()
            Parameters: format (string: json, txt, srt), output_path (string)
            Returns: bool success
            Description: Exports transcript to file


WAKE WORD FILES
...............

src/wake_word/wake_word_detector.py
    Purpose: Always-on wake word detection with low latency
    
    class WakeWordDetector
        def __init__()
            Parameters: config (WakeWordConfig), model_path (string)
            Description: Initializes wake word detection
        
        def start_detection()
            Parameters: audio_stream (AsyncGenerator), callback (callable)
            Returns: None
            Description: Starts continuous wake word detection
        
        def stop_detection()
            Parameters: None
            Returns: None
            Description: Stops wake word detection
        
        def is_wake_word()
            Parameters: audio_chunk (numpy array)
            Returns: bool detected, float confidence, string matched_phrase
            Description: Checks if audio contains wake word
        
        def set_sensitivity()
            Parameters: sensitivity (float 0-1)
            Returns: None
            Description: Adjusts detection sensitivity

src/wake_word/keyword_spotter.py
    Purpose: Low-level keyword spotting implementation
    
    class KeywordSpotter
        def __init__()
            Parameters: keywords (list of strings), model_type (string)
            Description: Initializes keyword spotter with target phrases
        
        def process_frame()
            Parameters: audio_frame (numpy array)
            Returns: list of dict with keyword, confidence, timestamp
            Description: Processes single audio frame for keywords
        
        def add_keyword()
            Parameters: keyword (string)
            Returns: bool success
            Description: Adds new keyword to detection list
        
        def remove_keyword()
            Parameters: keyword (string)
            Returns: bool success
            Description: Removes keyword from detection list


EXPERT AGENT FILES
..................

src/expert/expert_agent.py
    Purpose: Main expert agent coordinating knowledge and response generation
    
    class ExpertAgent
        def __init__()
            Parameters: knowledge_base (KnowledgeBase), llm_handler (LLMHandler), config (ExpertConfig)
            Description: Initializes expert agent with knowledge and LLM
        
        def handle_query()
            Parameters: query (string), meeting_context (string), speaker_id (string)
            Returns: string response
            Description: Handles user query and generates expert response
        
        async def handle_query_streaming()
            Parameters: query (string), meeting_context (string), speaker_id (string)
            Returns: AsyncGenerator yielding response chunks
            Description: Streaming version of handle_query for low latency
        
        def get_relevant_context()
            Parameters: query (string), max_tokens (int)
            Returns: string relevant context from knowledge and meeting
            Description: Retrieves relevant context for query
        
        def summarize_meeting()
            Parameters: None
            Returns: string meeting summary
            Description: Generates summary of meeting so far

src/expert/knowledge_base.py
    Purpose: Manages pre-loaded domain knowledge documents
    
    class KnowledgeBase
        def __init__()
            Parameters: documents_path (string), embeddings_path (string)
            Description: Initializes knowledge base from documents
        
        def load_documents()
            Parameters: path (string), file_types (list of strings)
            Returns: int document_count
            Description: Loads documents from specified path
        
        def add_document()
            Parameters: content (string), metadata (dict), doc_type (string)
            Returns: string document_id
            Description: Adds single document to knowledge base
        
        def search()
            Parameters: query (string), top_k (int), filters (dict or None)
            Returns: list of dict with content, metadata, score
            Description: Semantic search over knowledge base
        
        def get_by_topic()
            Parameters: topic (string)
            Returns: list of relevant documents
            Description: Gets documents related to topic

src/expert/rag_retriever.py
    Purpose: Retrieval Augmented Generation for combining knowledge with queries
    
    class RAGRetriever
        def __init__()
            Parameters: knowledge_base (KnowledgeBase), embedding_model (string)
            Description: Initializes RAG with knowledge base
        
        def retrieve()
            Parameters: query (string), context (string), top_k (int)
            Returns: list of relevant chunks with scores
            Description: Retrieves relevant knowledge chunks
        
        def rerank()
            Parameters: query (string), chunks (list), top_k (int)
            Returns: list of reranked chunks
            Description: Reranks retrieved chunks for relevance
        
        def build_context()
            Parameters: query (string), retrieved_chunks (list), meeting_context (string)
            Returns: string formatted context for LLM
            Description: Builds final context combining knowledge and meeting

src/expert/context_manager.py
    Purpose: Manages conversation and meeting context for expert responses
    
    class ContextManager
        def __init__()
            Parameters: max_context_tokens (int), transcript_store (TranscriptStore)
            Description: Initializes context manager
        
        def get_meeting_context()
            Parameters: max_tokens (int)
            Returns: string formatted meeting context
            Description: Gets relevant meeting context within token limit
        
        def get_speaker_context()
            Parameters: speaker_id (string), max_entries (int)
            Returns: string context specific to speaker
            Description: Gets context of what specific speaker has said
        
        def get_conversation_history()
            Parameters: max_turns (int)
            Returns: list of dict with role, content
            Description: Gets recent expert conversation history
        
        def add_interaction()
            Parameters: query (string), response (string), speaker_id (string)
            Returns: None
            Description: Adds expert interaction to history

src/expert/prompt_builder.py
    Purpose: Builds optimized prompts for expert LLM queries
    
    class PromptBuilder
        def __init__()
            Parameters: system_prompt_template (string), domain (string)
            Description: Initializes prompt builder with templates
        
        def build_query_prompt()
            Parameters: query (string), context (string), meeting_summary (string)
            Returns: list of messages for LLM
            Description: Builds prompt for answering user query
        
        def build_summary_prompt()
            Parameters: transcript (string), focus_areas (list or None)
            Returns: list of messages for LLM
            Description: Builds prompt for summarization
        
        def build_clarification_prompt()
            Parameters: original_query (string), clarification_request (string)
            Returns: list of messages for LLM
            Description: Builds prompt for clarifying questions


LLM HANDLER FILES
.................

src/llm/llm_handler.py
    Purpose: Main LLM handler with multi-provider fallback
    
    class LLMHandler
        def __init__()
            Parameters: config (LLMConfig)
            Description: Initializes LLM handler with provider chain
        
        async def generate()
            Parameters: messages (list), max_tokens (int), temperature (float)
            Returns: string response
            Description: Generates response using primary or fallback provider
        
        async def generate_streaming()
            Parameters: messages (list), max_tokens (int), temperature (float)
            Returns: AsyncGenerator yielding response chunks
            Description: Streaming generation for low latency
        
        async def generate_with_retry()
            Parameters: messages (list), max_retries (int)
            Returns: string response
            Description: Generates with automatic retry on failure
        
        def set_provider()
            Parameters: provider_name (string)
            Returns: bool success
            Description: Manually sets active provider

src/llm/openai_provider.py
    Purpose: OpenAI API provider implementation
    
    class OpenAIProvider
        def __init__()
            Parameters: api_key (string), model (string), base_url (string or None)
            Description: Initializes OpenAI provider
        
        async def generate()
            Parameters: messages (list), max_tokens (int), temperature (float)
            Returns: string response
            Description: Generates response using OpenAI API
        
        async def generate_streaming()
            Parameters: messages (list), max_tokens (int), temperature (float)
            Returns: AsyncGenerator yielding chunks
            Description: Streaming generation via OpenAI

src/llm/gemini_provider.py
    Purpose: Google Gemini API provider implementation
    
    class GeminiProvider
        def __init__()
            Parameters: api_key (string), model (string)
            Description: Initializes Gemini provider
        
        async def generate()
            Parameters: messages (list), max_tokens (int), temperature (float)
            Returns: string response
            Description: Generates response using Gemini API

src/llm/ollama_provider.py
    Purpose: Local Ollama provider for offline operation
    
    class OllamaProvider
        def __init__()
            Parameters: base_url (string), model (string)
            Description: Initializes Ollama provider
        
        async def generate()
            Parameters: messages (list), max_tokens (int), temperature (float)
            Returns: string response
            Description: Generates response using local Ollama

src/llm/response_streamer.py
    Purpose: Streams and processes LLM responses for TTS
    
    class ResponseStreamer
        def __init__()
            Parameters: sentence_end_tokens (list)
            Description: Initializes response streamer
        
        def process_chunk()
            Parameters: chunk (string)
            Returns: string complete_sentence or None
            Description: Buffers chunks and returns complete sentences
        
        def flush()
            Parameters: None
            Returns: string remaining_text
            Description: Flushes any remaining buffered text


TTS HANDLER FILES
.................

src/tts/tts_handler.py
    Purpose: Main TTS handler with barge-in support
    
    class TTSHandler
        def __init__()
            Parameters: stt_handler (STTHandler), config (TTSConfig)
            Description: Initializes TTS with STT reference for barge-in
        
        def speak()
            Parameters: text (string), enable_barge_in (bool)
            Returns: bool started
            Description: Starts speaking text with optional barge-in
        
        async def speak_streaming()
            Parameters: text_generator (AsyncGenerator), enable_barge_in (bool)
            Returns: bool completed
            Description: Speaks text as it streams from LLM
        
        def stop_speaking()
            Parameters: None
            Returns: None
            Description: Immediately stops TTS output
        
        def is_speaking()
            Parameters: None
            Returns: bool
            Description: Returns True if currently speaking
        
        def was_interrupted()
            Parameters: None
            Returns: bool
            Description: Returns True if last speech was interrupted

src/tts/cartesia_engine.py
    Purpose: Cartesia AI TTS engine with ultra-low latency
    
    class CartesiaTTSEngine
        def __init__()
            Parameters: api_key (string), voice_id (string), model (string)
            Description: Initializes Cartesia TTS
        
        async def synthesize()
            Parameters: text (string)
            Returns: AsyncGenerator yielding audio chunks
            Description: Synthesizes text to audio stream
        
        async def synthesize_and_play()
            Parameters: text (string), enable_barge_in (bool)
            Returns: bool completed
            Description: Synthesizes and plays audio
        
        def stop()
            Parameters: None
            Returns: None
            Description: Stops synthesis and playback
        
        def set_voice()
            Parameters: voice_id (string)
            Returns: None
            Description: Changes voice for synthesis

src/tts/audio_player.py
    Purpose: Low-latency audio playback with interruption support
    
    class AudioPlayer
        def __init__()
            Parameters: sample_rate (int), device (int or None)
            Description: Initializes audio player
        
        def play_chunk()
            Parameters: audio_chunk (numpy array)
            Returns: None
            Description: Plays single audio chunk
        
        async def play_stream()
            Parameters: audio_generator (AsyncGenerator)
            Returns: bool completed
            Description: Plays audio from generator
        
        def stop()
            Parameters: None
            Returns: None
            Description: Stops playback immediately
        
        def is_playing()
            Parameters: None
            Returns: bool
            Description: Returns playback state


MEETING CONNECTOR FILES
.......................

src/meeting/meeting_connector.py
    Purpose: Abstract base for meeting platform connectors
    
    class MeetingConnector
        def join_meeting()
            Parameters: meeting_url (string), display_name (string)
            Returns: bool success
            Description: Joins meeting as participant
        
        def leave_meeting()
            Parameters: None
            Returns: None
            Description: Leaves current meeting
        
        def get_audio_stream()
            Parameters: None
            Returns: AsyncGenerator yielding audio chunks
            Description: Gets audio stream from meeting
        
        def get_participants()
            Parameters: None
            Returns: list of participant info
            Description: Gets current meeting participants
        
        def is_connected()
            Parameters: None
            Returns: bool
            Description: Returns connection state

src/meeting/zoom_connector.py
    Purpose: Zoom meeting integration
    
    class ZoomConnector extends MeetingConnector
        def __init__()
            Parameters: config (ZoomConfig)
            Description: Initializes Zoom connector
        
        def join_meeting()
            Parameters: meeting_id (string), password (string or None), display_name (string)
            Returns: bool success
            Description: Joins Zoom meeting
        
        def capture_audio()
            Parameters: None
            Returns: AsyncGenerator yielding audio chunks
            Description: Captures Zoom audio via virtual audio device

src/meeting/meet_connector.py
    Purpose: Google Meet integration
    
    class MeetConnector extends MeetingConnector
        def __init__()
            Parameters: config (MeetConfig)
            Description: Initializes Meet connector
        
        def join_meeting()
            Parameters: meeting_url (string), display_name (string)
            Returns: bool success
            Description: Joins Google Meet

src/meeting/teams_connector.py
    Purpose: Microsoft Teams integration
    
    class TeamsConnector extends MeetingConnector
        def __init__()
            Parameters: config (TeamsConfig)
            Description: Initializes Teams connector
        
        def join_meeting()
            Parameters: meeting_url (string), display_name (string)
            Returns: bool success
            Description: Joins Teams meeting


ORCHESTRATOR AND MAIN FILES
...........................

src/orchestrator.py
    Purpose: Coordinates all components for real-time operation
    
    class MeetingOrchestrator
        def __init__()
            Parameters: config (AppConfig)
            Description: Initializes all components
        
        async def start()
            Parameters: audio_source (string: microphone, virtual, meeting_url)
            Returns: None
            Description: Starts the orchestrator
        
        async def stop()
            Parameters: None
            Returns: None
            Description: Stops all components gracefully
        
        async def process_audio_stream()
            Parameters: audio_stream (AsyncGenerator)
            Returns: None
            Description: Main processing loop for audio
        
        async def handle_wake_word_detected()
            Parameters: audio_after_wake (numpy array)
            Returns: None
            Description: Handles wake word detection event
        
        async def handle_expert_query()
            Parameters: query (string), speaker_id (string)
            Returns: None
            Description: Processes expert query and speaks response
        
        def get_meeting_status()
            Parameters: None
            Returns: dict with stats and state
            Description: Returns current meeting status

src/main.py
    Purpose: Application entry point and CLI interface
    
    def main()
        Parameters: None (uses CLI args)
        Returns: None
        Description: Main entry point
    
    async def run_assistant()
        Parameters: config (AppConfig), mode (string)
        Returns: None
        Description: Runs the assistant in specified mode
    
    def parse_arguments()
        Parameters: None
        Returns: argparse.Namespace
        Description: Parses command line arguments


UTILITY FILES
.............

src/utils/logger.py
    Purpose: Centralized logging configuration
    
    def setup_logging()
        Parameters: level (string), log_file (string or None)
        Returns: logging.Logger
        Description: Configures application logging
    
    class PerformanceLogger
        def log_latency()
            Parameters: component (string), latency_ms (float)
            Returns: None
            Description: Logs component latency

src/utils/memory_monitor.py
    Purpose: Monitors and manages memory usage
    
    class MemoryMonitor
        def check_memory()
            Parameters: None
            Returns: dict with usage stats
            Description: Checks current memory usage
        
        def trigger_cleanup()
            Parameters: None
            Returns: None
            Description: Triggers garbage collection if needed

src/utils/performance_tracker.py
    Purpose: Tracks performance metrics
    
    class PerformanceTracker
        def record_metric()
            Parameters: name (string), value (float), unit (string)
            Returns: None
            Description: Records performance metric
        
        def get_summary()
            Parameters: None
            Returns: dict with metric summaries
            Description: Gets performance summary

src/utils/thread_manager.py
    Purpose: Manages concurrent threads and tasks
    
    class ThreadManager
        def start_background_task()
            Parameters: coroutine (callable), name (string)
            Returns: asyncio.Task
            Description: Starts named background task
        
        def stop_all()
            Parameters: timeout (float)
            Returns: None
            Description: Stops all managed tasks


RUNNING COMMANDS
----------------

Installation:
    pip install -r requirements.txt
    python setup.py install

Environment Setup:
    cp .env.example .env
    (Edit .env with your API keys)

Index Knowledge Base:
    python scripts/index_knowledge.py --documents ./knowledge/documents --output ./knowledge/index

Enroll Speakers (Optional):
    python scripts/enroll_speaker.py --name "John Smith" --audio ./recordings/john_sample.wav

Test Audio Devices:
    python scripts/test_audio_device.py --list
    python scripts/test_audio_device.py --device 0 --test

Run with Microphone (In-Person Meeting):
    python -m src.main --mode microphone --device default

Run with Virtual Audio (Virtual Meeting):
    python -m src.main --mode virtual --application zoom

Join Zoom Meeting:
    python -m src.main --mode zoom --meeting-id 123456789 --password abc123 --name "AI Expert"

Join Google Meet:
    python -m src.main --mode meet --meeting-url "https://meet.google.com/xxx-xxxx-xxx" --name "AI Expert"

Join Microsoft Teams:
    python -m src.main --mode teams --meeting-url "https://teams.microsoft.com/..." --name "AI Expert"

Run Tests:
    pytest tests/ -v
    pytest tests/test_integration.py -v

Debug Mode:
    python -m src.main --mode microphone --debug --log-level DEBUG


EXPECTED OUTPUT
---------------

Startup Output:
    ==================================================
    Subject Matter Expert Voice Assistant v1.0
    ==================================================
    Loading configuration... Done
    Initializing audio capture... Done
    Loading speaker diarization model... Done
    Initializing STT engine... Done
    Loading knowledge base (247 documents)... Done
    Initializing wake word detector... Done
    Starting TTS engine... Done
    ==================================================
    System Ready
    Wake phrase: "Hey Expert"
    Mode: Microphone capture
    Speakers detected: 0
    ==================================================

During Meeting (Console Log):
    [00:00:15] Speaker-1 (Unknown): "Let's discuss the quarterly results."
    [00:00:23] Speaker-2 (Unknown): "Revenue was up 15 percent from last quarter."
    [00:00:35] Speaker-1: "What about the projections for next quarter?"
    [00:00:42] Speaker-2: "We're expecting similar growth based on current trends."
    [00:01:05] [New Speaker Detected] Speaker-3
    [00:01:08] Speaker-3 (Unknown): "Hey Expert, can you summarize the key financial points discussed?"
    [00:01:10] [Wake Word Detected] Processing query...
    [00:01:10] [Expert] Query: "summarize the key financial points discussed"
    [00:01:11] [Expert] Retrieving context... (2 meeting entries, 3 knowledge chunks)
    [00:01:12] [Expert] Generating response...
    [00:01:13] [TTS] Speaking response...
    [00:01:13] [Expert Response] "Based on the discussion, there are two key financial points. First, revenue increased 15 percent compared to last quarter. Second, projections suggest similar growth will continue next quarter based on current trends. Would you like more details on either point?"
    [00:01:22] [TTS] Response complete
    [00:01:30] Speaker-1: "Hey Expert, what were our Q2 targets from the planning document?"
    [00:01:32] [Wake Word Detected] Processing query...
    [00:01:33] [Expert] Searching knowledge base... Found relevant document: "Q2_Planning_2024.pdf"
    [00:01:34] [Expert Response] "According to the Q2 planning document, your targets were..."

Meeting End Summary:
    ==================================================
    Meeting Session Complete
    Duration: 45 minutes 32 seconds
    ==================================================
    Speakers Identified: 4
      - Speaker-1: 12 min 45 sec speaking time
      - Speaker-2: 8 min 12 sec speaking time  
      - Speaker-3: 5 min 33 sec speaking time
      - Speaker-4: 3 min 15 sec speaking time
    
    Transcription Stats:
      - Total segments: 156
      - Average confidence: 94.2%
    
    Expert Interactions: 7
      - Queries answered: 7
      - Average response time: 1.2 seconds
      - Knowledge base hits: 12
    
    Performance Metrics:
      - STT latency (avg): 145ms
      - Diarization latency (avg): 89ms
      - LLM response (avg): 890ms
      - TTS first byte (avg): 52ms
      - End-to-end response (avg): 1180ms
    
    Transcript saved to: ./outputs/meeting_2024-01-15_14-30-00.txt
    ==================================================


HARDWARE REQUIREMENTS
---------------------

Minimum:
    CPU: 4 cores (Intel i5 / AMD Ryzen 5 or equivalent)
    RAM: 8 GB
    Storage: 10 GB free space
    Microphone: USB or built-in microphone
    Network: Stable internet connection for cloud LLM

Recommended:
    CPU: 8 cores (Intel i7 / AMD Ryzen 7 or equivalent)
    RAM: 16 GB
    GPU: NVIDIA GPU with 4GB VRAM (for local models)
    Storage: 20 GB SSD
    Microphone: Professional USB microphone or array microphone
    Audio Interface: Virtual audio cable software for meeting capture


DEPENDENCIES
------------

Core:
    python >= 3.9
    numpy
    scipy
    asyncio
    httpx
    python-dotenv

Audio Processing:
    pyaudio
    sounddevice
    soundfile
    librosa

Speech Recognition:
    faster-whisper
    RealtimeSTT
    webrtcvad
    silero-vad

Speaker Diarization:
    pyannote.audio
    speechbrain
    resemblyzer

Wake Word:
    openwakeword
    pvporcupine (optional, requires license)

LLM and RAG:
    openai
    google-generativeai
    langchain
    chromadb
    sentence-transformers

TTS:
    cartesia
    pyttsx3 (fallback)

Meeting Integration:
    selenium
    playwright
    pyautogui

Utilities:
    psutil
    colorlog
    tqdm
