"""Main LLM handler with multi-provider fallback."""
import logging
from typing import List, Dict, Optional, AsyncGenerator
import asyncio

from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .response_streamer import ResponseStreamer

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Main LLM handler with multi-provider fallback.
    """
    
    def __init__(
        self,
        openai_key: str = "",
        openai_model: str = "gpt-4-turbo-preview",
        gemini_key: str = "",
        gemini_model: str = "gemini-pro",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama2",
        provider_priority: Optional[List[str]] = None
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
        """
        self.provider_priority = provider_priority or ["openai", "gemini", "ollama"]
        
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
        
        logger.info(f"LLMHandler initialized: providers={list(self._providers.keys())}")
    
    async def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generates response using primary or fallback provider.
        
        Parameters:
            messages: List of message dicts
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            string: Generated response
        """
        # Try each provider in order
        for provider_name in self.provider_priority:
            if provider_name not in self._providers:
                continue
            
            provider = self._providers[provider_name]
            
            try:
                logger.debug(f"Trying provider: {provider_name}")
                response = await provider.generate(messages, max_tokens, temperature)
                self._active_provider = provider_name
                return response
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
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
