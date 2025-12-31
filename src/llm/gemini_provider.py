"""Google Gemini API provider implementation."""
import logging
from typing import List, Dict, AsyncGenerator

logger = logging.getLogger(__name__)


class GeminiProvider:
    """
    Google Gemini API provider implementation.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro"
    ):
        """
        Initializes Gemini provider.
        
        Parameters:
            api_key: Google API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self._client = None
        
        logger.info(f"GeminiProvider initialized: model={model}")
    
    def _get_client(self):
        """Get or create client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                logger.error("google-generativeai package not installed")
                raise
        return self._client
    
    async def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generates response using Gemini API.
        
        Parameters:
            messages: List of message dicts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            string: Generated response
        """
        try:
            client = self._get_client()
            
            # Convert messages to Gemini format
            prompt = self._format_messages(messages)
            
            # Generate
            response = client.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': temperature
                }
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    async def generate_streaming(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation via Gemini.
        
        Yields:
            Response chunks
        """
        try:
            client = self._get_client()
            
            prompt = self._format_messages(messages)
            
            response = client.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': temperature
                },
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Convert chat messages to Gemini format."""
        parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                parts.append(f"Instructions: {content}")
            elif role == 'assistant':
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        
        return "\n\n".join(parts)
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self.api_key)
