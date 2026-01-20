"""LLM provider configuration with fallback chain."""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


@dataclass
class LLMProviderConfig:
    """Per-provider configuration settings."""
    
    provider_name: str
    api_key: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    base_url: Optional[str] = None
    timeout: int = 30
    
    def __post_init__(self):
        """Validate LLM provider configuration."""
        if self.provider_name not in [p.value for p in LLMProvider]:
            raise ValueError(f"Unsupported provider: {self.provider_name}")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be between 0 and 2")
        
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive")
        
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive")


def get_provider_chain(
    openai_key: str = "",
    openai_model: str = "gpt-4-turbo-preview",
    openai_base_url: str = "https://api.openai.com/v1",
    gemini_key: str = "",
    gemini_model: str = "gemini-pro",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llama2",
    provider_priority: Optional[List[str]] = None
) -> List[LLMProviderConfig]:
    """
    Returns ordered list of LLM providers for fallback.
    
    Parameters:
        openai_key: OpenAI API key
        openai_model: OpenAI model name
        openai_base_url: OpenAI base URL
        gemini_key: Google Gemini API key
        gemini_model: Gemini model name
        ollama_url: Ollama base URL for local models
        ollama_model: Ollama model name
        provider_priority: List of provider names in priority order
    
    Returns:
        List of LLMProviderConfig in priority order
    """
    if provider_priority is None:
        provider_priority = ["openai", "ollama"]
    
    # Create provider configurations
    providers = {
        "openai": LLMProviderConfig(
            provider_name="openai",
            api_key=openai_key,
            model=openai_model,
            base_url=openai_base_url,
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        ),
        "gemini": LLMProviderConfig(
            provider_name="gemini",
            api_key=gemini_key,
            model=gemini_model,
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        ),
        "ollama": LLMProviderConfig(
            provider_name="ollama",
            api_key="",  # Local doesn't need API key
            model=ollama_model,
            base_url=ollama_url,
            max_tokens=1000,
            temperature=0.7,
            timeout=60  # Local models may be slower
        ),
    }
    
    # Return providers in priority order, skipping those without keys (except ollama)
    provider_chain = []
    for provider_name in provider_priority:
        if provider_name in providers:
            provider = providers[provider_name]
            # Include provider if it has an API key or is ollama (local)
            if provider.api_key or provider_name == "ollama":
                provider_chain.append(provider)
    
    return provider_chain


@dataclass
class LLMConfig:
    """General LLM configuration."""
    
    provider_chain: List[LLMProviderConfig]
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 10.0  # Maximum delay for exponential backoff
    enable_streaming: bool = True
    enable_fallback: bool = True
    request_timeout: float = 12.0  # Request timeout in seconds
    connect_timeout: float = 5.0  # Connection timeout
    
    def __post_init__(self):
        """Validate LLM configuration."""
        if not self.provider_chain:
            raise ValueError("provider_chain cannot be empty")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
