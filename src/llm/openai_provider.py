"""OpenAI API provider implementation."""
import logging
from typing import List, Dict, Optional, AsyncGenerator
import asyncio

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    OpenAI API provider implementation.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initializes OpenAI provider.
        
        Parameters:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Optional custom base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        
        self._client = None
        self._async_client = None
        
        logger.info(f"OpenAIProvider initialized: model={model}")
    
    def _get_client(self):
        """Get or create sync client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client
    
    def _get_async_client(self):
        """Get or create async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._async_client
    
    async def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generates response using OpenAI API.
        
        Parameters:
            messages: List of message dicts
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            string: Generated response
        """
        try:
            client = self._get_async_client()
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def generate_streaming(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation via OpenAI.
        
        Parameters:
            messages: List of message dicts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Yields:
            Response chunks
        """
        try:
            client = self._get_async_client()
            
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self.api_key)
