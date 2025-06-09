from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Model Provider Configuration
    model_provider: str = "openrouter"  # Options: "openrouter", "cerebras", "openai"
    
    # OpenRouter API configuration
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "qwen/qwen2.5-vl-72b-instruct:free"
    
    # Cerebras API configuration
    cerebras_api_key: Optional[str] = None
    cerebras_model: str = "llama-4-scout-17b-16e-instruct"
    
    # OpenAI API configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    # Audio configuration
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_size: int = 1024
    audio_device_index: Optional[int] = None  # None = use system default device
    vad_aggressiveness: int = 3  # WebRTC VAD aggressiveness (0-3)
    silence_duration_threshold: float = 2.0  # seconds of silence to stop recording
    
    # Vision configuration
    camera_index: int = 0  # Default to first available camera
    camera_width: int = 640   # Start with VGA resolution (widely supported)
    camera_height: int = 480  # VGA height
    camera_fps: int = 30      # 30fps is widely supported (60fps often fails)
    
    # LLM configuration
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    
    @property
    def current_api_key(self) -> str:
        """Get the API key for the current provider"""
        if self.model_provider == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is required when using OpenRouter provider")
            return self.openrouter_api_key
        elif self.model_provider == "cerebras":
            if not self.cerebras_api_key:
                raise ValueError("Cerebras API key is required when using Cerebras provider")
            return self.cerebras_api_key
        elif self.model_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required when using OpenAI provider")
            return self.openai_api_key
        else:
            raise ValueError(f"Unknown model provider: {self.model_provider}")
    
    @property
    def current_model(self) -> str:
        """Get the model for the current provider"""
        if self.model_provider == "openrouter":
            return self.openrouter_model
        elif self.model_provider == "cerebras":
            return self.cerebras_model
        elif self.model_provider == "openai":
            return self.openai_model
        else:
            raise ValueError(f"Unknown model provider: {self.model_provider}")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
