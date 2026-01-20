"""Main LLM handler with multi-provider fallback."""
import logging
import time
from typing import List, Dict, Optional, AsyncGenerator
import asyncio

from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .response_streamer import ResponseStreamer

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Main LLM handler with multi-provider fallback and retry logic.
    Enhanced with error tracking and exponential backoff from voice_engine_MVP.
    """
    
    def __init__(
        self,
        openai_key: str = "",
        openai_model: str = "gpt-4-turbo-preview",
        gemini_key: str = "",
        gemini_model: str = "gemini-pro",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama2",
        provider_priority: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 10.0
    ):
        """
        Initializes LLM handler with provider chain.
        
        Parameters:
            openai_key: OpenAI API key
            openai_model: OpenAI model name
            gemini_key: Google Gemini API key
            gemini_model: Gemini model name
            ollama_url: Ollama server URL
            ollama_model: Ollama model name
            provider_priority: List of providers in priority order
            max_retries: Maximum retry attempts per provider
            retry_base_delay: Initial retry delay (exponential backoff)
            retry_max_delay: Maximum retry delay
        """
        self.provider_priority = provider_priority or ["openai", "gemini", "ollama"]
        
        # Retry configuration (from voice_engine_MVP)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        
        # Error tracking (from voice_engine_MVP)
        self._consecutive_errors = 0
        self._last_error_time = 0.0
        self._interaction_count = 0
        
        # Initialize providers
        self._providers = {}
        
        if openai_key:
            self._providers["openai"] = OpenAIProvider(
                api_key=openai_key,
                model=openai_model
            )
        
        if gemini_key:
            self._providers["gemini"] = GeminiProvider(
                api_key=gemini_key,
                model=gemini_model
            )
        
        # Ollama doesn't need API key
        self._providers["ollama"] = OllamaProvider(
            base_url=ollama_url,
            model=ollama_model
        )
        
        # Response streamer
        self.response_streamer = ResponseStreamer()
        
        # Current active provider
        self._active_provider: Optional[str] = None
        
        logger.info(f"🤖 LLMHandler initialized: providers={' → '.join(list(self._providers.keys())).upper()}")
    
    async def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generates response using primary or fallback provider.
        Enhanced with retry logic and exponential backoff from voice_engine_MVP.
        
        Parameters:
            messages: List of message dicts
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            string: Generated response
        """
        self._interaction_count += 1
        
        # Try each provider in order with retry
        for provider_idx, provider_name in enumerate(self.provider_priority):
            if provider_name not in self._providers:
                continue
            
            provider = self._providers[provider_name]
            
            # Retry with exponential backoff for current provider
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Trying {provider_name.upper()} (attempt {attempt + 1}/{self.max_retries})")
                    response = await provider.generate(messages, max_tokens, temperature)
                    
                    # Success - reset error tracking
                    if provider_name != self._active_provider:
                        logger.info(f"✅ Switched to {provider_name.upper()} provider")
                    self._active_provider = provider_name
                    self._consecutive_errors = 0
                    return response
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ {provider_name.upper()} timeout (attempt {attempt + 1})")
                except Exception as e:
                    logger.warning(f"⚠️ {provider_name.upper()} error: {str(e)[:100]}")
                
                # Exponential backoff before retry
                if attempt < self.max_retries - 1:
                    delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
                    logger.debug(f"⏳ Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
            
            # All retries failed for this provider, try next
            logger.warning(f"❌ {provider_name.upper()} failed after {self.max_retries} retries")
        
        # All providers failed
        self._track_error()
        raise RuntimeError("All LLM providers failed")
    
    async def generate_streaming(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation for low latency.
        
        Parameters:
            messages: List of message dicts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Yields:
            Response chunks
        """
        # Try each provider
        for provider_name in self.provider_priority:
            if provider_name not in self._providers:
                continue
            
            provider = self._providers[provider_name]
            
            try:
                logger.debug(f"Streaming with provider: {provider_name}")
                self._active_provider = provider_name
                
                async for chunk in provider.generate_streaming(messages, max_tokens, temperature):
                    yield chunk
                
                return  # Success
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} streaming failed: {e}")
                continue
        
        raise RuntimeError("All LLM providers failed for streaming")
    
    async def generate_with_retry(
        self,
        messages: List[Dict],
        max_retries: int = 3,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generates with automatic retry on failure.
        
        Parameters:
            messages: List of message dicts
            max_retries: Maximum retry attempts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            string: Generated response
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await self.generate(messages, max_tokens, temperature)
            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
        
        raise last_error or RuntimeError("Generation failed after retries")
    
    def set_provider(self, provider_name: str) -> bool:
        """
        Manually sets active provider.
        
        Parameters:
            provider_name: Provider to activate
        
        Returns:
            bool: Success
        """
        if provider_name in self._providers:
            # Move to front of priority
            if provider_name in self.provider_priority:
                self.provider_priority.remove(provider_name)
            self.provider_priority.insert(0, provider_name)
            
            logger.info(f"Set active provider to: {provider_name}")
            return True
        
        logger.warning(f"Provider not found: {provider_name}")
        return False
    
    def get_active_provider(self) -> Optional[str]:
        """Get currently active provider name."""
        return self._active_provider
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self._providers.keys())
    
    async def check_providers(self) -> Dict[str, bool]:
        """Check availability of all providers."""
        status = {}
        
        for name, provider in self._providers.items():
            if name == "ollama":
                status[name] = await provider.is_available()
            else:
                status[name] = provider.is_available()
        
        return status
    
    def _track_error(self):
        """Track consecutive errors for circuit breaker pattern (from voice_engine_MVP)."""
        self._consecutive_errors += 1
        self._last_error_time = time.time()
        
        if self._consecutive_errors >= 3:
            logger.error(
                f"⚠️ {self._consecutive_errors} consecutive errors detected. "
                f"Consider checking API status."
            )
    
    def get_stats(self) -> Dict:
        """Get handler statistics."""
        return {
            "interaction_count": self._interaction_count,
            "consecutive_errors": self._consecutive_errors,
            "active_provider": self._active_provider,
            "available_providers": list(self._providers.keys())
        }
    
    async def shutdown(self):
        """Cleanup handler resources."""
        try:
            # Close any async clients if providers have them
            for name, provider in self._providers.items():
                if hasattr(provider, '_async_client') and provider._async_client:
                    try:
                        await provider._async_client.close()
                    except Exception as e:
                        logger.debug(f"Provider {name} client close error: {e}")
            
            logger.info("✅ LLMHandler shutdown complete")
        except Exception as e:
            logger.error(f"❌ LLMHandler shutdown error: {e}")
