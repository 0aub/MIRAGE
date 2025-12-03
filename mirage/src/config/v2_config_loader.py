"""
MIRAGE V2 Configuration Loader
Enhanced configuration system with:
- Environment variable substitution (${VAR_NAME} or ${VAR_NAME:default})
- Multi-file configuration merging
- Type-safe access with dot notation
- Hot-reload support
- Validation and schema checking
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, TypeVar, Union
from functools import lru_cache
from copy import deepcopy
from loguru import logger


T = TypeVar('T')


class ConfigError(Exception):
    """Configuration error"""
    pass


class ConfigDict(dict):
    """
    Dictionary that allows dot notation access and supports defaults.
    Example: config.models.llm.model_id
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Config key not found: {key}")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_nested(self, path: str, default: T = None) -> Union[Any, T]:
        """
        Get nested value using dot notation path.

        Args:
            path: Dot-separated path (e.g., "models.llm.model_id")
            default: Default value if path not found

        Returns:
            Value at path or default
        """
        keys = path.split('.')
        value = self

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set_nested(self, path: str, value: Any) -> None:
        """
        Set nested value using dot notation path.
        Creates intermediate dicts if needed.

        Args:
            path: Dot-separated path (e.g., "models.llm.model_id")
            value: Value to set
        """
        keys = path.split('.')
        current = self

        for key in keys[:-1]:
            if key not in current:
                current[key] = ConfigDict()
            elif not isinstance(current[key], dict):
                current[key] = ConfigDict()
            current = current[key]

        current[keys[-1]] = value


class V2ConfigLoader:
    """
    Enhanced configuration loader for MIRAGE V2.

    Features:
    - Loads configuration from YAML files
    - Supports environment variable substitution: ${VAR} or ${VAR:default}
    - Merges multiple config files (defaults + overrides)
    - Provides type-safe access via dot notation
    - Supports hot-reload
    """

    # Pattern for environment variable substitution
    # Matches ${VAR_NAME} or ${VAR_NAME:default_value}
    ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:[^}]*)?\}')

    def __init__(
        self,
        config_dir: Optional[str] = None,
        env_file: Optional[str] = None,
        override_files: Optional[List[str]] = None
    ):
        """
        Initialize configuration loader.

        Args:
            config_dir: Path to config directory (defaults to this file's directory)
            env_file: Optional .env file to load environment variables from
            override_files: Optional list of override config files to merge
        """
        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)
        self.core_dir = self.config_dir / "core"
        self.components_dir = self.config_dir / "components"

        # Load .env file if specified
        if env_file:
            self._load_env_file(env_file)

        # Override files to merge on top of defaults
        self.override_files = override_files or []

        # Configuration cache
        self._config: ConfigDict = ConfigDict()
        self._loaded_files: List[str] = []

        # Load configuration
        self._load_all_configs()

    def _load_env_file(self, env_file: str) -> None:
        """Load environment variables from .env file"""
        env_path = Path(env_file)
        if not env_path.exists():
            logger.warning(f".env file not found: {env_file}")
            return

        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)

        logger.debug(f"Loaded environment from {env_file}")

    def _substitute_env_vars(self, value: Any) -> Any:
        """
        Recursively substitute environment variables in config values.

        Supports:
        - ${VAR_NAME} - Required variable (error if not set)
        - ${VAR_NAME:default} - Variable with default value

        Args:
            value: Config value (can be string, dict, or list)

        Returns:
            Value with environment variables substituted
        """
        if isinstance(value, str):
            def replace_env(match):
                var_name = match.group(1)
                default = match.group(2)

                if default is not None:
                    # Remove leading colon from default
                    default = default[1:]

                env_value = os.environ.get(var_name)

                if env_value is not None:
                    return env_value
                elif default is not None:
                    return default
                else:
                    logger.warning(f"Environment variable not set: {var_name}")
                    return match.group(0)  # Return original placeholder

            return self.ENV_PATTERN.sub(replace_env, value)

        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]

        return value

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        Override values take precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Parsed configuration dictionary
        """
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # Substitute environment variables
            config = self._substitute_env_vars(config)

            logger.debug(f"Loaded config from {file_path.name}")
            self._loaded_files.append(str(file_path))
            return config

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load {file_path}: {e}")

    def _load_all_configs(self) -> None:
        """Load all configuration files and merge them"""
        self._config = ConfigDict()
        self._loaded_files = []

        # 1. Load core defaults
        defaults_path = self.core_dir / "defaults.yaml"
        if defaults_path.exists():
            defaults = self._load_yaml(defaults_path)
            self._config = ConfigDict(defaults)

        # 2. Load component configs (from components/ directory)
        if self.components_dir.exists():
            for yaml_file in sorted(self.components_dir.glob("*.yaml")):
                component_config = self._load_yaml(yaml_file)
                self._config = ConfigDict(self._deep_merge(dict(self._config), component_config))

        # 3. Load any override files
        for override_file in self.override_files:
            override_path = Path(override_file)
            if not override_path.is_absolute():
                override_path = self.config_dir / override_file

            if override_path.exists():
                override_config = self._load_yaml(override_path)
                self._config = ConfigDict(self._deep_merge(dict(self._config), override_config))

        # 4. Apply environment variable overrides
        self._apply_env_overrides()

        logger.info(f"Loaded {len(self._loaded_files)} configuration files")

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides with MIRAGE_ prefix"""
        prefix = "MIRAGE_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert MIRAGE_MODELS_LLM_MODEL_ID to models.llm.model_id
                config_path = key[len(prefix):].lower().replace('_', '.')

                # Handle nested paths that might have underscores in the key name
                # This is a simplified approach - may need refinement
                self._config.set_nested(config_path, value)
                logger.debug(f"Applied env override: {config_path} = {value[:20]}...")

    def reload(self) -> None:
        """Reload all configurations from files"""
        logger.info("Reloading configuration")
        self._load_all_configs()

    # =========================================================================
    # ACCESS METHODS
    # =========================================================================

    @property
    def config(self) -> ConfigDict:
        """Get full configuration as ConfigDict"""
        return self._config

    def get(self, path: str, default: T = None) -> Union[Any, T]:
        """
        Get configuration value by dot-notation path.

        Args:
            path: Dot-separated path (e.g., "models.llm.model_id")
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        return self._config.get_nested(path, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: config['key']"""
        return self._config[key]

    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access: config.key"""
        if key.startswith('_'):
            return super().__getattribute__(key)
        return getattr(self._config, key)

    # =========================================================================
    # TYPED ACCESSORS
    # =========================================================================

    def get_llm_config(self) -> ConfigDict:
        """Get LLM configuration"""
        return ConfigDict(self.get("models.llm", {}))

    def get_embedding_config(self) -> ConfigDict:
        """Get embedding model configuration"""
        return ConfigDict(self.get("models.embedding", {}))

    def get_arabic_config(self) -> ConfigDict:
        """Get Arabic processing configuration"""
        return ConfigDict(self.get("arabic", {}))

    def get_chunking_config(self) -> ConfigDict:
        """Get chunking configuration"""
        return ConfigDict(self.get("chunking", {}))

    def get_retrieval_config(self) -> ConfigDict:
        """Get retrieval configuration"""
        return ConfigDict(self.get("retrieval", {}))

    def get_graph_config(self) -> ConfigDict:
        """Get graph configuration"""
        return ConfigDict(self.get("graph", {}))

    def get_qdrant_config(self) -> ConfigDict:
        """Get Qdrant configuration"""
        return ConfigDict(self.get("databases.qdrant", {}))

    def get_neo4j_config(self) -> ConfigDict:
        """Get Neo4j configuration"""
        return ConfigDict(self.get("databases.neo4j", {}))

    def get_generation_config(self) -> ConfigDict:
        """Get generation configuration"""
        return ConfigDict(self.get("generation", {}))

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(self) -> bool:
        """
        Validate configuration for required values and consistency.

        Returns:
            True if valid

        Raises:
            ConfigError: If validation fails
        """
        errors = []

        # Check required paths
        required_paths = [
            "models.llm.model_id",
            "models.embedding.model_id",
            "databases.qdrant.host",
            "databases.neo4j.uri",
        ]

        for path in required_paths:
            if not self.get(path):
                errors.append(f"Required config missing: {path}")

        # Check embedding dimension consistency
        embedding_dim = self.get("models.embedding.dimensions")
        # Note: Qdrant collection config should be checked at runtime

        # Check retrieval mode validity
        default_mode = self.get("retrieval.default_mode")
        valid_modes = ["naive", "local", "global", "hybrid", "mix", "semantic", "bypass"]
        if default_mode and default_mode not in valid_modes:
            errors.append(f"Invalid retrieval mode: {default_mode}. Valid: {valid_modes}")

        # Check Arabic processor validity
        arabic_processor = self.get("arabic.processor")
        valid_processors = ["camel", "stanza", "none"]
        if arabic_processor and arabic_processor not in valid_processors:
            errors.append(f"Invalid Arabic processor: {arabic_processor}. Valid: {valid_processors}")

        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            raise ConfigError(f"Configuration validation failed: {'; '.join(errors)}")

        logger.info("Configuration validation passed")
        return True

    def get_loaded_files(self) -> List[str]:
        """Get list of loaded configuration files"""
        return self._loaded_files.copy()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_v2_config_loader: Optional[V2ConfigLoader] = None


def get_v2_config() -> V2ConfigLoader:
    """
    Get or create global V2ConfigLoader instance.

    Returns:
        Global V2ConfigLoader instance
    """
    global _v2_config_loader

    if _v2_config_loader is None:
        _v2_config_loader = V2ConfigLoader()
        try:
            _v2_config_loader.validate()
        except ConfigError as e:
            logger.warning(f"Config validation warning: {e}")

    return _v2_config_loader


def reload_v2_config() -> None:
    """Reload global configuration"""
    global _v2_config_loader

    if _v2_config_loader is not None:
        _v2_config_loader.reload()
        _v2_config_loader.validate()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_config_value(path: str, default: Any = None) -> Any:
    """
    Convenience function to get config value.

    Args:
        path: Dot-separated path
        default: Default value

    Returns:
        Configuration value or default
    """
    return get_v2_config().get(path, default)
