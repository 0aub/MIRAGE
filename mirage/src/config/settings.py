"""
MIRAGE Configuration Management
Centralized settings using Pydantic for type safety and validation
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Configuration
    port: int = 8000
    debug: bool = True
    environment: str = "development"

    # LLM Provider API Keys (at least ONE required for entity extraction)
    openai_api_key: str = ""
    anthropic_api_key: str = ""  # Claude
    google_api_key: str = ""     # Gemini

    # Local LLM via Text Generation Inference (TGI)
    use_tgi: bool = False
    tgi_endpoint: str = "http://localhost:8765"  # Main inference endpoint (Allam)
    entity_extraction_endpoint: str = "http://localhost:8766"  # Entity extraction endpoint (Qwen)

    # Local LLM via Ollama (lightweight alternative to TGI)
    use_ollama: bool = False
    ollama_endpoint: str = "http://ollama:11434"  # Ollama API endpoint (use 'ollama' service name in Docker)
    ollama_model: str = "gemma3:4b"  # Model to use (gemma3:4b, qwen2.5:7b, etc.)

    # Database Configuration
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    redis_url: str = "redis://redis:6379"

    mongodb_url: str = "mongodb://mongo:27017"
    mongodb_db: str = "mirage"

    # REFRAG Configuration
    refrag_compression_rate: int = 16
    refrag_cache_size: int = 1000

    # Processing Configuration
    max_upload_size: int = 100  # MB
    chunk_size: int = 1000      # tokens
    chunk_overlap: int = 100    # tokens

    # Content Rewriting
    # IMPORTANT: Disabled by default to test entity extraction on raw text
    # Enable this to pre-process content before entity extraction (adds 2-5 min per document)
    enable_content_rewriting: bool = False

    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS - Allow all origins in development for easier testing
    allowed_origins: Union[List[str], str] = ["*"]

    # Logging
    log_level: str = "INFO"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience export
settings = get_settings()
