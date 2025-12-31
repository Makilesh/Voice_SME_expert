"""Local Ollama provider for offline operation."""
import logging
from typing import List, Dict, AsyncGenerator
import httpx

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Local Ollama provider for offline operation.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: int = 60
    ):
        """
        Initializes Ollama provider.
        
        Parameters:
            base_url: Ollama server URL
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        
        logger.info(f"OllamaProvider initialized: model={model}, url={base_url}")
    
    async def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generates response using local Ollama.
        
        Parameters:
            messages: List of message dicts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            string: Generated response
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        },
                        "stream": False
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                return data.get("message", {}).get("content", "")
                
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    async def generate_streaming(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Streaming generation via Ollama.
        
        Yields:
            Response chunks
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        },
                        "stream": True
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
