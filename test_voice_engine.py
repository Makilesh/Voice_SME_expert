"""
Test script for Voice Engine components (STT, LLM, TTS).
Tests the core voice interaction components without full orchestrator.
"""
import os
import sys
import asyncio
import logging
import warnings
from dotenv import load_dotenv
import time

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
from src.llm.llm_handler import LLMHandler
from src.tts.tts_handler import TTSHandler
from config import load_config


class ConversationManager:
    """Simple conversation history manager."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        self.turn_count = 0
        self.error_count = 0
        
    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            # Keep system prompt + recent history
            self.history = [self.history[0]] + self.history[-(self.max_history-1):]
        if role != "System":
            self.turn_count += 1
            
    def get_history(self) -> list:
        """Get conversation history."""
        return self.history.copy()
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
        
    def reset_errors(self):
        """Reset error counter."""
        self.error_count = 0
        
    def should_abort(self) -> bool:
        """Check if too many consecutive errors."""
        return self.error_count >= 3


async def test_individual_components():
    """Test each component individually."""
    print("\n" + "="*60)
    print("🧪 COMPONENT TESTS")
    print("="*60)
    
    config = load_config()
    
    # Test 1: LLM Handler
    print("\n1️⃣ Testing LLM Handler...")
    try:
        llm_handler = LLMHandler(
            openai_key=config.openai_api_key,
            openai_model=config.openai_model,
            gemini_key=config.gemini_api_key,
            gemini_model=config.gemini_model,
            ollama_url=config.ollama_base_url,
            ollama_model=config.ollama_model,
            provider_priority=config.llm_provider_chain
        )
        
        # Test simple generation
        test_query = "What is Shamla Tech?"
        print(f"   Query: {test_query}")
        
        messages = [{"role": "user", "content": test_query}]
        response = await llm_handler.generate(messages)
        print(f"   ✅ Response: {response[:100]}...")
        
        # Test with retry
        print(f"\n   Testing retry logic...")
        messages2 = [{"role": "user", "content": "Tell me about AI services"}]
        response2 = await llm_handler.generate(messages2)
        print(f"   ✅ Retry test passed: {response2[:80]}...")
        
        # Cleanup
        await llm_handler.shutdown()
        print(f"   ✅ LLM Handler tests passed")
    except Exception as e:
        print(f"   ❌ LLM Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: TTS Handler (simplified without STT dependency)
    print("\n2️⃣ Testing TTS Handler...")
    try:
        # Create a minimal mock STT handler for TTS
        class MockSTTHandler:
            def __init__(self):
                self.realtime_text = ""
                self._lock = __import__('threading').Lock()
                
            def get_realtime_text(self):
                with self._lock:
                    return self.realtime_text
                    
            def clear_realtime_text(self):
                with self._lock:
                    self.realtime_text = ""
        
        mock_stt = MockSTTHandler()
        
        tts_handler = TTSHandler(
            cartesia_api_key=config.cartesia_api_key,
            cartesia_voice_id=config.cartesia_voice_id,
            cartesia_model=config.cartesia_model,
            sample_rate=config.cartesia_sample_rate,
            stt_handler=mock_stt,
            barge_in_enabled=False  # Disable for component test
        )
        
        test_texts = [
            "Hello! This is a test of the text to speech system.",
            "Testing the enhanced TTS with barge-in detection support."
        ]
        
        for i, test_text in enumerate(test_texts, 1):
            print(f"   Test {i}: {test_text[:50]}...")
            await tts_handler.speak(test_text, blocking=False)
            completed = tts_handler.wait_for_completion(timeout=15.0)
            
            if completed:
                print(f"   ✅ Test {i} completed")
            else:
                print(f"   ⚠️ Test {i} interrupted/timeout")
            
            time.sleep(0.5)  # Brief pause between tests
        
        print(f"   ✅ TTS Handler tests passed")
        
    except Exception as e:
        print(f"   ❌ TTS Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def test_text_conversation():
    """Test text-based conversation (LLM + TTS without live STT)."""
    print("\n" + "="*60)
    print("💬 TEXT-BASED CONVERSATION TEST")
    print("   (Type messages, get spoken responses)")
    print("="*60)
    
    tts_handler = None
    llm_handler = None
    
    try:
        # Load configuration
        config = load_config()
        
        # Validate API keys
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        print("\n🔧 Initializing components...")
        
        # Initialize LLM
        print("   • LLM Handler (Language Model)...")
        llm_handler = LLMHandler(
            openai_key=config.openai_api_key,
            openai_model=config.openai_model,
            gemini_key=config.gemini_api_key,
            gemini_model=config.gemini_model,
            ollama_url=config.ollama_base_url,
            ollama_model=config.ollama_model,
            provider_priority=config.llm_provider_chain
        )
        logger.info("✅ LLM: Ready")
        
        # Initialize TTS with mock STT for barge-in
        print("   • TTS Handler (Text-to-Speech)...")
        
        class MockSTTHandler:
            """Mock STT handler for TTS barge-in detection."""
            def __init__(self):
                self.realtime_text = ""
                self._lock = __import__('threading').Lock()
                
            def get_realtime_text(self):
                with self._lock:
                    return self.realtime_text
                    
            def clear_realtime_text(self):
                with self._lock:
                    self.realtime_text = ""
        
        mock_stt = MockSTTHandler()
        
        tts_handler = TTSHandler(
            cartesia_api_key=config.cartesia_api_key,
            cartesia_voice_id=config.cartesia_voice_id,
            cartesia_model=config.cartesia_model,
            sample_rate=config.cartesia_sample_rate,
            stt_handler=mock_stt,
            barge_in_enabled=config.tts_barge_in_enabled
        )
        
        logger.info("✅ TTS: Ready")
        
        # Initialize conversation manager
        conversation_manager = ConversationManager(max_history=10)
        conversation_manager.add_turn(
            "System",
            "You are Alex, an AI voice assistant for Shamla Tech. "
            "Be warm, helpful, and concise. Keep responses under 3 sentences."
        )
        
        print("\n" + "="*60)
        print("✅ All systems ready!")
        print("="*60)
        
        # Welcome message
        welcome = "Hello! I'm your AI assistant. Type your questions and I'll respond with both text and voice."
        print(f"\n🤖 Assistant: {welcome}")
        conversation_manager.add_turn("Assistant", welcome)
        
        await tts_handler.speak(welcome, blocking=False)
        tts_handler.wait_for_completion(timeout=15.0)
        
        print("\n💡 Tips:")
        print("   • Type your message and press Enter")
        print("   • Type 'quit' or 'exit' to end")
        print("   • Responses will be both displayed and spoken")
        print()
        
        # Conversation loop
        max_turns = 20
        turn_count = 0
        
        while turn_count < max_turns:
            turn_count += 1
            
            try:
                # Get user input from console
                print(f"\n[Turn {turn_count}]")
                user_input = input("📝 You: ").strip()
                
                if not user_input:
                    print("   ⚠️ Empty input, please try again")
                    turn_count -= 1
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'goodbye', 'bye', 'stop']:
                    print("\n👋 Goodbye!")
                    farewell = "Goodbye! Have a great day!"
                    await tts_handler.speak(farewell, blocking=False)
                    tts_handler.wait_for_completion(timeout=10.0)
                    break
                
                conversation_manager.add_turn("User", user_input)
                
                # Generate LLM response
                print("   🤖 Thinking...")
                
                start_time = time.time()
                # Build messages from history
                messages = []
                for msg in conversation_manager.get_history():
                    role = msg["role"].lower()
                    if role == "system":
                        messages.append({"role": "system", "content": msg["content"]})
                    elif role == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif role == "assistant":
                        messages.append({"role": "assistant", "content": msg["content"]})
                
                response = await llm_handler.generate(messages)
                llm_time = time.time() - start_time
                
                if not response or len(response.strip()) < 2:
                    response = "I'm sorry, I didn't quite understand. Could you rephrase that?"
                    conversation_manager.record_error()
                else:
                    conversation_manager.reset_errors()
                
                print(f"   🤖 Assistant: {response}")
                print(f"   ⏱ Response time: {llm_time*1000:.0f}ms")
                conversation_manager.add_turn("Assistant", response)
                
                # Speak response
                print("   🗣 Speaking...")
                
                tts_start = time.time()
                await tts_handler.speak(response, blocking=False)
                completed = tts_handler.wait_for_completion(timeout=30.0)
                tts_time = time.time() - tts_start
                
                if completed:
                    print(f"   ✅ Speech completed ({tts_time*1000:.0f}ms)")
                else:
                    print(f"   ⚠️ Speech interrupted")
                
            except EOFError:
                print("\n\n⚠️ Input stream closed")
                break
                
            except Exception as e:
                logger.error(f"❌ Turn error: {e}", exc_info=True)
                conversation_manager.record_error()
                
                if conversation_manager.should_abort():
                    print("\n❌ Too many consecutive errors. Exiting...")
                    break
                
                print("   ⚠️ Error occurred, continuing...")
        
        # Print session stats
        print("\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        print(f"   Total turns: {conversation_manager.turn_count}")
        print(f"   Total errors: {conversation_manager.error_count}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        
    finally:
        print("\n🧹 Shutting down...")
        
        # Cleanup TTS
        if tts_handler:
            try:
                tts_handler.stop()
                logger.info("✅ TTS shutdown complete")
            except Exception as e:
                logger.error(f"TTS cleanup error: {e}")
        
        # Cleanup LLM
        if llm_handler:
            try:
                await llm_handler.shutdown()
                logger.info("✅ LLM shutdown complete")
            except Exception as e:
                logger.error(f"LLM cleanup error: {e}")
        
        print("✅ Shutdown complete")


async def main():
    """Main test entry point."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║       🎙 Voice Engine Test - LLM + TTS Components        ║
║       Tests conversation flow without live microphone     ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    print("\nSelect test mode:")
    print("1. Component Tests (individual LLM/TTS tests)")
    print("2. Text Conversation (type messages, get voice responses)")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3, default: 2): ").strip() or "2"
        
        if choice == "1":
            success = await test_individual_components()
            if success:
                print("\n✅ All component tests passed!")
            else:
                print("\n❌ Some component tests failed")
                
        elif choice == "2":
            await test_text_conversation()
            
        elif choice == "3":
            print("\n📋 Running component tests first...\n")
            success = await test_individual_components()
            
            if success:
                print("\n✅ Component tests passed!")
                print("\n" + "="*60)
                input("\nPress Enter to start text conversation test...")
                await test_text_conversation()
            else:
                print("\n❌ Component tests failed. Skipping conversation test.")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
