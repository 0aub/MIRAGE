"""
Configuration Loader Utility
Loads all centralized YAML configurations
Provides type-safe access to settings with validation
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from functools import lru_cache


class ConfigLoader:
    """
    Centralized configuration loading from YAML files
    Provides type-safe access to all configuration settings
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader

        Args:
            config_dir: Path to config directory (defaults to this file's directory)
        """
        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)

        # Configuration caches
        self._models_config: Dict[str, Any] = {}
        self._processing_config: Dict[str, Any] = {}
        self._databases_config: Dict[str, Any] = {}

        # Load all configs on initialization
        self._load_all_configs()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        file_path = self.config_dir / filename

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logger.debug(f"Loaded config from {filename}")
            return config

        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            raise

    def _load_all_configs(self) -> None:
        """Load all configuration files"""
        try:
            self._models_config = self._load_yaml("models.yaml")
            self._processing_config = self._load_yaml("processing.yaml")
            self._databases_config = self._load_yaml("databases.yaml")
            logger.info("Successfully loaded all configurations")

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

    def reload(self) -> None:
        """Reload all configurations from files (useful for hot-reloading)"""
        logger.info("Reloading all configurations")
        self._load_all_configs()

    # ============================================================
    # MODELS CONFIGURATION
    # ============================================================

    def get_llm_config(self) -> Dict[str, Any]:
        """Get complete LLM configuration"""
        return self._models_config.get("llm", {})

    def get_enabled_llm_provider(self) -> Optional[Dict[str, Any]]:
        """
        Get the highest-priority enabled LLM provider

        Returns:
            Dict with provider name and config, or None if no provider enabled
        """
        providers = self._models_config.get("llm", {}).get("providers", {})

        # Sort by priority (lower number = higher priority)
        enabled_providers = [
            (name, config)
            for name, config in providers.items()
            if config.get("enabled", False)
        ]

        if not enabled_providers:
            return None

        # Sort by priority
        enabled_providers.sort(key=lambda x: x[1].get("priority", 999))

        provider_name, provider_config = enabled_providers[0]
        return {
            "name": provider_name,
            **provider_config
        }

    def get_llm_parameters(self) -> Dict[str, Any]:
        """Get LLM generation parameters (temperature, max_tokens)"""
        return self._models_config.get("llm", {}).get("parameters", {})

    def get_model_constraints(self, model_name: str) -> Dict[str, Any]:
        """
        Get token and character constraints for a specific model

        Args:
            model_name: Model identifier (e.g., "tgi_allam_7b", "openai_gpt4")

        Returns:
            Dict with max_input_tokens, reserved_response_tokens, chars_per_token
        """
        constraints = self._models_config.get("llm", {}).get("constraints", {})

        if model_name in constraints:
            return constraints[model_name]

        # Return conservative defaults if model not found
        logger.warning(f"Model constraints not found for {model_name}, using defaults")
        return {
            "max_input_tokens": 4096,
            "reserved_response_tokens": 1000,
            "chars_per_token": 4
        }

    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get complete embeddings configuration"""
        return self._models_config.get("embeddings", {})

    def get_jina_config(self) -> Dict[str, Any]:
        """Get Jina embeddings configuration"""
        return self._models_config.get("embeddings", {}).get("jina", {})

    # ============================================================
    # PROCESSING CONFIGURATION
    # ============================================================

    def get_semantic_chunking_config(self) -> Dict[str, Any]:
        """Get semantic chunking configuration"""
        return self._processing_config.get("semantic_chunking", {})

    def get_content_rewriting_config(self) -> Dict[str, Any]:
        """Get content rewriting configuration"""
        return self._processing_config.get("content_rewriting", {})

    def is_content_rewriting_enabled(self) -> bool:
        """Check if content rewriting is enabled"""
        return self._processing_config.get("content_rewriting", {}).get("enabled", False)

    def get_refrag_config(self) -> Dict[str, Any]:
        """Get REFRAG compression configuration"""
        return self._processing_config.get("refrag", {})

    def get_upload_config(self) -> Dict[str, Any]:
        """Get upload limits configuration"""
        return self._processing_config.get("upload", {})

    def get_entity_extraction_config(self) -> Dict[str, Any]:
        """Get entity extraction configuration"""
        return self._processing_config.get("entity_extraction", {})

    def get_graph_generation_config(self) -> Dict[str, Any]:
        """Get graph generation configuration"""
        return self._processing_config.get("graph_generation", {})

    # ============================================================
    # DATABASES CONFIGURATION
    # ============================================================

    def get_neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j configuration"""
        return self._databases_config.get("neo4j", {})

    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration"""
        return self._databases_config.get("qdrant", {})

    def get_qdrant_vector_config(self) -> Dict[str, Any]:
        """Get Qdrant vector configuration"""
        return self._databases_config.get("qdrant", {}).get("vector_config", {})

    def get_qdrant_search_config(self) -> Dict[str, Any]:
        """Get Qdrant search configuration"""
        return self._databases_config.get("qdrant", {}).get("search", {})

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return self._databases_config.get("redis", {})

    def get_mongodb_config(self) -> Dict[str, Any]:
        """Get MongoDB configuration"""
        return self._databases_config.get("mongodb", {})

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        return {
            "models": self._models_config,
            "processing": self._processing_config,
            "databases": self._databases_config,
        }

    def validate_configs(self) -> bool:
        """
        Validate all configurations

        Returns:
            True if all configs are valid, raises exception otherwise
        """
        # Check that at least one LLM provider is enabled
        enabled_provider = self.get_enabled_llm_provider()
        if not enabled_provider:
            logger.warning("No LLM provider enabled - entity extraction will fail")

        # Check that embedding dimension matches Qdrant config
        jina_dim = self.get_jina_config().get("embedding_dim", 1024)
        qdrant_dim = self.get_qdrant_vector_config().get("dimension", 1024)

        if jina_dim != qdrant_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"Jina={jina_dim}, Qdrant={qdrant_dim}"
            )

        logger.info("Configuration validation passed")
        return True


# Global instance for easy access
_config_loader: Optional[ConfigLoader] = None


@lru_cache()
def get_config_loader() -> ConfigLoader:
    """
    Get or create global ConfigLoader instance

    Returns:
        Global ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
        _config_loader.validate_configs()
    return _config_loader


def reload_configs() -> None:
    """Reload all configurations from files"""
    global _config_loader
    if _config_loader is not None:
        _config_loader.reload()
        _config_loader.validate_configs()
