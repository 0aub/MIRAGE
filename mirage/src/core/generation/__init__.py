"""
MIRAGE V2 Generation Module
Prompt management and response generation.

Usage:
    from core.generation import get_prompt_manager, get_response_generator

    # Get managers
    prompts = get_prompt_manager()
    generator = get_response_generator()

    # List available prompts
    templates = prompts.list_templates()

    # Generate response
    response = generator.generate(
        question="What is MIRAGE?",
        context=[{"text": "MIRAGE is a RAG system...", "document_id": "doc1"}]
    )
    print(response.answer)
"""

from .prompt_manager import (
    PromptManager,
    PromptType,
    PromptTemplate,
    FormattedPrompt,
    get_prompt_manager,
)

from .response_generator import (
    ResponseGenerator,
    GenerationConfig,
    GenerationMode,
    GeneratedResponse,
    get_response_generator,
)

__all__ = [
    # Prompt management
    "PromptManager",
    "PromptType",
    "PromptTemplate",
    "FormattedPrompt",
    "get_prompt_manager",
    # Response generation
    "ResponseGenerator",
    "GenerationConfig",
    "GenerationMode",
    "GeneratedResponse",
    "get_response_generator",
]
