"""
Prompt Loader Utility
Loads prompts from centralized prompts.yaml configuration
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class PromptLoader:
    """
    Centralized prompt loading from YAML configuration
    Provides type-safe access to prompts with language support
    """

    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize prompt loader

        Args:
            prompts_file: Path to prompts YAML file (defaults to config/prompts.yaml)
        """
        if prompts_file is None:
            # Default to prompts.yaml in the same directory
            config_dir = Path(__file__).parent
            prompts_file = config_dir / "prompts.yaml"

        self.prompts_file = Path(prompts_file)
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from YAML file"""
        try:
            if not self.prompts_file.exists():
                raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")

            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f)

            logger.debug(f"Loaded prompts from {self.prompts_file}")

        except Exception as e:
            logger.error(f"Failed to load prompts from {self.prompts_file}: {e}")
            raise

    def reload(self) -> None:
        """Reload prompts from file (useful for hot-reloading during development)"""
        logger.info("Reloading prompts from file")
        self._load_prompts()

    def get_content_enrichment_prompt(self, language: str = "english") -> Dict[str, str]:
        """
        Get content enrichment prompt for entity extraction preparation

        Args:
            language: Language code ("english" or "arabic")

        Returns:
            Dict with 'system' and 'user_template' keys
        """
        try:
            prompts = self._prompts["content_enrichment"][language]
            return {
                "system": prompts["system"],
                "user_template": prompts["user_template"]
            }
        except KeyError:
            logger.error(f"Content enrichment prompt not found for language: {language}")
            raise ValueError(f"No content enrichment prompt for language: {language}")

    def get_youtube_cleanup_prompt(self, language: str = "english") -> Dict[str, str]:
        """
        Get YouTube transcript cleanup prompt

        Args:
            language: Language code ("english" or "arabic")

        Returns:
            Dict with 'system' and 'user_template' keys
        """
        try:
            prompts = self._prompts["youtube_cleanup"][language]
            return {
                "system": prompts["system"],
                "user_template": prompts["user_template"]
            }
        except KeyError:
            logger.error(f"YouTube cleanup prompt not found for language: {language}")
            raise ValueError(f"No YouTube cleanup prompt for language: {language}")

    def get_web_content_cleanup_prompt(self, language: str = "english") -> Dict[str, str]:
        """
        Get web content cleanup prompt

        Args:
            language: Language code ("english" or "arabic")

        Returns:
            Dict with 'system' and 'user_template' keys
        """
        try:
            prompts = self._prompts["web_content_cleanup"][language]
            return {
                "system": prompts["system"],
                "user_template": prompts["user_template"]
            }
        except KeyError:
            logger.error(f"Web content cleanup prompt not found for language: {language}")
            raise ValueError(f"No web content cleanup prompt for language: {language}")

    def get_entity_extraction_prompt(self, language: str = "english") -> Dict[str, str]:
        """
        Get entity extraction prompt

        Args:
            language: Language code ("english" or "arabic")

        Returns:
            Dict with 'system' and 'user_template' keys
        """
        try:
            prompts = self._prompts["entity_extraction"][language]
            return {
                "system": prompts["system"],
                "user_template": prompts["user_template"]
            }
        except KeyError:
            logger.error(f"Entity extraction prompt not found for language: {language}")
            raise ValueError(f"No entity extraction prompt for language: {language}")

    def get_content_type_config(self, content_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific content type

        Args:
            content_type: Content type ("youtube", "webpage", "file")

        Returns:
            Dict with content type configuration
        """
        try:
            return self._prompts["content_type_config"][content_type]
        except KeyError:
            logger.error(f"Content type config not found: {content_type}")
            raise ValueError(f"No configuration for content type: {content_type}")

    def get_model_constraints(self, model_name: str) -> Dict[str, Any]:
        """
        Get token and character constraints for a specific model

        Args:
            model_name: Model identifier ("tgi_allam_7b", "openai_gpt4", etc.)

        Returns:
            Dict with max_input_tokens, reserved_response_tokens, chars_per_token
        """
        try:
            return self._prompts["model_constraints"][model_name]
        except KeyError:
            logger.warning(f"Model constraints not found for: {model_name}, using defaults")
            # Return conservative defaults
            return {
                "max_input_tokens": 4096,
                "reserved_response_tokens": 1000,
                "chars_per_token": 4
            }

    def format_user_prompt(self, template: str, **kwargs) -> str:
        """
        Format a user prompt template with provided variables

        Args:
            template: Template string with {variable} placeholders
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Template variable not provided: {e}")


# Global instance for easy access
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """
    Get or create global PromptLoader instance

    Returns:
        Global PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


def reload_prompts() -> None:
    """Reload prompts from file (useful for hot-reloading)"""
    global _prompt_loader
    if _prompt_loader is not None:
        _prompt_loader.reload()
