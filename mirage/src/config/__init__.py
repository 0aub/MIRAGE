"""Configuration module for MIRAGE"""

from .settings import settings, get_settings
from .config_loader import get_config_loader, reload_configs, ConfigLoader
from .prompt_loader import get_prompt_loader, reload_prompts, PromptLoader

# V2 Configuration System
from .v2_config_loader import (
    V2ConfigLoader,
    get_v2_config,
    reload_v2_config,
    get_config_value,
    ConfigDict,
    ConfigError,
)

__all__ = [
    # Settings (legacy)
    "settings",
    "get_settings",
    # Config Loader (legacy)
    "get_config_loader",
    "reload_configs",
    "ConfigLoader",
    # Prompt Loader
    "get_prompt_loader",
    "reload_prompts",
    "PromptLoader",
    # V2 Configuration System
    "V2ConfigLoader",
    "get_v2_config",
    "reload_v2_config",
    "get_config_value",
    "ConfigDict",
    "ConfigError",
]
