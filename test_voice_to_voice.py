"""
Full voice-to-voice test - uses microphone for input and speakers for output.
Tests the complete STT -> LLM -> TTS pipeline.
"""
import os
import sys
import asyncio
import logging
import warnings
from dotenv import load_dotenv
import time
import threading
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import voice components
from src.transcription.stt_handler import STTHandler
from src.llm.llm_handler import LLMHandler
from src.tts.tts_handler import TTSHandler
from config import load_config


class VoiceToVoiceTest:
    """Test complete voice-to-voice conversation flow."""
    
    def __init__(self):
        self.config = load_config()
        self.stt_handler = None
        self.llm_handler = None
        self.tts_handler = None
        self.conversation_history = []
        self.is_running = False
        
    async def initialize(self):
        """Initialize all components."""
        print("\n🔧 Initializing components...")
        
        # Initialize STT
        print("   • STT Handler (Speech-to-Text)...")
        self.stt_handler = STTHandler(
            model=self.config.stt_model,
            language=self.config.stt_language,
            compute_type=self.config.stt_compute_type,
            sample_rate=self.config.sample_rate,
            mode=self.config.stt_mode,
            transcription_timeout=self.config.stt_transcription_timeout
        )
        logger.info("✅ STT: Ready")
        
        # Initialize LLM
        print("   • LLM Handler (Language Model)...")
        self.llm_handler = LLMHandler(
            openai_key=self.config.openai_api_key,
            openai_model=self.config.openai_model,
            gemini_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
            ollama_url=self.config.ollama_base_url,
            ollama_model=self.config.ollama_model,
            provider_priority=self.config.llm_provider_chain
        )
        logger.info("✅ LLM: Ready")
        
        # Initialize TTS
        print("   • TTS Handler (Text-to-Speech)...")
        
        # Only pass Cartesia key if enabled
        tts_api_key = self.config.cartesia_api_key if self.config.use_cartesia_tts else ""
        
        self.tts_handler = TTSHandler(
            cartesia_api_key=tts_api_key,
            cartesia_voice_id=self.config.cartesia_voice_id,
            cartesia_model=self.config.cartesia_model,
            sample_rate=self.config.cartesia_sample_rate,
            stt_handler=self.stt_handler,
            barge_in_enabled=self.config.tts_barge_in_enabled
        )
        
        if tts_api_key:
            logger.info("✅ TTS: Ready (Cartesia)")
        else:
            logger.info("✅ TTS: Ready (System Fallback)")
        
        # Add system prompt
        self.conversation_history.append({
            "role": "system",
            "content": "You are Alex, a friendly AI voice assistant for Shamla Tech. "
                      "Keep responses very concise (1-2 sentences). Be warm and helpful."
        })
        
        print("\n✅ All systems initialized!")
        
    async def listen_for_input(self) -> str:
        """Listen for voice input using microphone."""
        print("\n🎤 Listening... (speak for 5 seconds)")
        
        try:
            import sounddevice as sd
            
            # Record audio
            duration = 5  # seconds
            sample_rate = self.config.sample_rate
            
            print("   🔴 Recording...")
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            print("   ✅ Recording complete")
            
            # Convert to numpy array with correct dtype
            audio_segment = audio_data.flatten().astype(np.float32)
            
            # Check if there's meaningful audio
            max_amplitude = np.max(np.abs(audio_segment))
            if max_amplitude < 0.01:
                print("   ⚠️ Very quiet audio, you might need to speak louder")
                return ""
            
            # Transcribe
            print("   🔄 Transcribing...")
            result = self.stt_handler.transcribe_segment(
                audio_segment=audio_segment,
                speaker_id="User",
                speaker_name="You"
            )
            
            return result.get('text', '').strip()
            
        except Exception as e:
            logger.error(f"Listen error: {e}", exc_info=True)
            return ""
    
    async def process_and_respond(self, user_input: str) -> str:
        """Process input through LLM and respond."""
        if not user_input:
            return ""
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        print(f"\n👤 You said: {user_input}")
        print("   🤖 Thinking...")
        
        # Generate response
        start_time = time.time()
        response = await self.llm_handler.generate(
            messages=self.conversation_history,
            max_tokens=150,
            temperature=0.7
        )
        elapsed = time.time() - start_time
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        print(f"   🤖 Assistant: {response}")
        print(f"   ⏱ Response time: {elapsed*1000:.0f}ms")
        
        return response
    
    async def speak_response(self, text: str):
        """Speak the response."""
        if not text:
            return
        
        print("   🗣 Speaking...")
        
        start_time = time.time()
        await self.tts_handler.speak(text, blocking=False)
        
        # Wait for completion with proper timeout
        completed = self.tts_handler.wait_for_completion(timeout=20.0)
        elapsed = time.time() - start_time
        
        if completed:
            print(f"   ✅ Speech completed ({elapsed*1000:.0f}ms)")
        else:
            print(f"   ⚠️ Speech interrupted or timed out")
    
    async def run_conversation(self, max_turns: int = 5):
        """Run voice conversation loop."""
        print("\n" + "="*60)
        print("✅ Starting Voice Conversation")
        print("="*60)
        
        # Welcome
        welcome = "Hello! I'm ready. Ask me anything about Shamla Tech or AI services."
        print(f"\n🤖 Assistant: {welcome}")
        await self.speak_response(welcome)
        
        print("\n💡 Tips:")
        print("   • Speak clearly after you see '🎤 Listening...'")
        print("   • Wait for the assistant to finish speaking")
        print("   • Say 'goodbye' or 'quit' to exit")
        print()
        
        turn = 0
        while turn < max_turns and self.is_running:
            turn += 1
            print(f"\n{'='*60}")
            print(f"Turn {turn}/{max_turns}")
            print('='*60)
            
            try:
                # Listen
                user_input = await self.listen_for_input()
                
                if not user_input:
                    print("   ⚠️ No speech detected. Try again.")
                    continue
                
                # Check for exit
                if any(word in user_input.lower() for word in ['goodbye', 'bye', 'quit', 'exit', 'stop']):
                    farewell = "Goodbye! Have a great day!"
                    print(f"\n🤖 Assistant: {farewell}")
                    await self.speak_response(farewell)
                    break
                
                # Process and respond
                response = await self.process_and_respond(user_input)
                await self.speak_response(response)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Turn error: {e}", exc_info=True)
                print(f"   ❌ Error: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("📊 CONVERSATION SUMMARY")
        print("="*60)
        print(f"   Turns completed: {turn}")
        print(f"   Messages: {len(self.conversation_history) - 1}")  # Exclude system prompt
        print("="*60)
    
    async def cleanup(self):
        """Cleanup all components."""
        print("\n🧹 Shutting down...")
        
        if self.tts_handler:
            try:
                self.tts_handler.stop()
                logger.info("✅ TTS shutdown complete")
            except Exception as e:
                logger.error(f"TTS cleanup error: {e}")
        
        if self.stt_handler:
            try:
                self.stt_handler.reset()
                logger.info("✅ STT shutdown complete")
            except Exception as e:
                logger.error(f"STT cleanup error: {e}")
        
        if self.llm_handler:
            try:
                await self.llm_handler.shutdown()
                logger.info("✅ LLM shutdown complete")
            except Exception as e:
                logger.error(f"LLM cleanup error: {e}")
        
        print("✅ Shutdown complete")


async def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║     🎙 Voice-to-Voice Test - Complete STT→LLM→TTS        ║
║         Speak into microphone, hear AI responses          ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    print("\n⚠️ IMPORTANT: Make sure you've run 'python test_audio_system.py' first!")
    print("   This test requires working microphone and speakers.\n")
    
    response = input("Ready to start? (y/n): ").lower()
    if response != 'y':
        print("👋 Test cancelled")
        return
    
    test = VoiceToVoiceTest()
    test.is_running = True
    
    try:
        await test.initialize()
        await test.run_conversation(max_turns=10)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await test.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
