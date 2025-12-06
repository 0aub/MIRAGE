"""
MIRAGE V2 Prompt Manager
Manages versioned prompt templates with YAML configuration.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import yaml
import re
from loguru import logger
from datetime import datetime


class PromptType(Enum):
    """Types of prompts"""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    SUMMARIZATION = "summarization"
    QA_GENERATION = "qa_generation"
    ANSWER_GENERATION = "answer_generation"


@dataclass
class PromptTemplate:
    """A prompt template with metadata"""
    name: str
    version: str
    type: PromptType
    template: str
    system_message: str = ""
    description: str = ""

    # Chain-of-thought settings
    use_cot: bool = False
    cot_steps: List[str] = field(default_factory=list)

    # Few-shot settings
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    max_examples: int = 3

    # Language settings
    languages: List[str] = field(default_factory=lambda: ["en", "ar"])

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormattedPrompt:
    """A formatted prompt ready for LLM"""
    system_message: str
    user_message: str
    template_name: str
    template_version: str
    variables_used: Dict[str, Any]


class PromptManager:
    """
    Manages prompt templates for different tasks.

    Features:
    - YAML-based template storage
    - Version management
    - Chain-of-thought formatting
    - Few-shot example injection
    - Bilingual support (Arabic/English)
    """

    def __init__(
        self,
        prompts_dir: Optional[Union[str, Path]] = None,
        default_language: str = "auto"
    ):
        """
        Initialize prompt manager.

        Args:
            prompts_dir: Directory containing prompt YAML files
            default_language: Default language for prompts ("en", "ar", "auto")
        """
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self.default_language = default_language

        # Template cache: {type: {version: PromptTemplate}}
        self._templates: Dict[PromptType, Dict[str, PromptTemplate]] = {}

        # Load built-in templates
        self._load_builtin_templates()

        # Load from directory if provided
        if self.prompts_dir and self.prompts_dir.exists():
            self._load_from_directory()

        logger.info(
            f"PromptManager initialized: {len(self._templates)} template types, "
            f"language={default_language}"
        )

    def _load_builtin_templates(self):
        """Load built-in prompt templates"""

        # Entity extraction prompt
        self._register_template(PromptTemplate(
            name="entity_extraction_v1",
            version="1.0",
            type=PromptType.ENTITY_EXTRACTION,
            description="Extract entities from text using structured format",
            system_message="""You are an expert at extracting named entities from text.
Extract all entities and their types. Support Arabic and English text.

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, EVENT, OTHER""",
            template="""Extract all named entities from the following text.

Text:
{text}

Return as JSON array:
[{{"name": "entity name", "type": "ENTITY_TYPE", "confidence": 0.0-1.0}}]

Entities:""",
            use_cot=False
        ))

        # Entity extraction with CoT
        self._register_template(PromptTemplate(
            name="entity_extraction_cot_v1",
            version="1.0",
            type=PromptType.ENTITY_EXTRACTION,
            description="Extract entities with chain-of-thought reasoning",
            system_message="""You are an expert at extracting named entities from text.
Think step by step to identify all entities. Support Arabic and English.""",
            template="""Extract named entities from the following text using step-by-step reasoning.

Text:
{text}

Let's think step by step:
1. First, identify any person names mentioned
2. Then, identify organizations or companies
3. Next, identify locations (countries, cities, places)
4. Finally, identify dates, products, or events

Now extract the entities:""",
            use_cot=True,
            cot_steps=[
                "Identify person names",
                "Identify organizations",
                "Identify locations",
                "Identify other entities"
            ]
        ))

        # Relationship extraction
        self._register_template(PromptTemplate(
            name="relationship_extraction_v1",
            version="1.0",
            type=PromptType.RELATIONSHIP_EXTRACTION,
            description="Extract relationships between entities",
            system_message="""You are an expert at identifying relationships between entities.
Extract semantic relationships from the text. Support Arabic and English.""",
            template="""Given the following text and entities, extract relationships between them.

Text:
{text}

Entities:
{entities}

For each relationship, provide:
- source_entity: The subject entity
- target_entity: The object entity
- relationship_type: The type of relationship (e.g., WORKS_FOR, LOCATED_IN, PART_OF)
- description: Brief description of the relationship

Relationships:""",
            use_cot=False
        ))

        # Answer generation - V3 (Phase 3: Never lead with negatives)
        self._register_template(PromptTemplate(
            name="answer_generation_v1",
            version="3.0",
            type=PromptType.ANSWER_GENERATION,
            description="Generate answer from retrieved context - optimized for Arabic",
            system_message="""أنت مساعد ذكي متخصص في الإجابة على الأسئلة باستخدام السياق المقدم.
يجب أن تجيب دائماً باللغة العربية.

CRITICAL RULES:
1. NEVER start your answer with negative phrases like "لا يوجد" or "لا أجد" or "لم أتمكن"
2. ALWAYS start directly with the relevant information from the context
3. Focus on what IS mentioned, not what isn't
4. If the exact entity isn't mentioned, discuss related entities or topics from the context
5. Keep your answer informative and focused on the context provided""",
            template="""أجب على السؤال التالي. ابدأ مباشرة بالمعلومات المتوفرة.

السياق:
{context}

السؤال: {question}

تعليمات صارمة:
- ابدأ إجابتك مباشرة بالمعلومات (لا تبدأ بـ "لا يوجد" أو "لم أجد")
- إذا لم يُذكر الكيان المطلوب مباشرة، تحدث عن الكيانات المرتبطة
- قدم المعلومات المتوفرة في السياق بشكل مختصر ومفيد
- لا تتجاوز 3 جمل

الإجابة:""",
            use_cot=False
        ))

        # Answer generation with CoT
        self._register_template(PromptTemplate(
            name="answer_generation_cot_v1",
            version="1.0",
            type=PromptType.ANSWER_GENERATION,
            description="Generate answer with chain-of-thought reasoning",
            system_message="""You are a helpful assistant that thinks step by step.
Answer in the same language as the question. Always cite sources.""",
            template="""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Let me analyze this step by step:

Step 1: Identify the key information needed
Step 2: Find relevant information in the context
Step 3: Synthesize the answer
Step 4: Verify with citations

Thinking: {cot_placeholder}

Final Answer:""",
            use_cot=True,
            cot_steps=[
                "Identify key information needed",
                "Find relevant context",
                "Synthesize answer",
                "Verify with citations"
            ]
        ))

        # Summarization
        self._register_template(PromptTemplate(
            name="summarization_v1",
            version="1.0",
            type=PromptType.SUMMARIZATION,
            description="Summarize text content",
            system_message="""You are an expert at creating concise, accurate summaries.
Preserve key information while reducing length. Support Arabic and English.""",
            template="""Summarize the following text in {target_length} sentences.

Text:
{text}

Key points to preserve:
- Main topic and themes
- Important entities mentioned
- Key facts and figures
- Conclusions or outcomes

Summary:""",
            use_cot=False
        ))

        # Community summarization (for GraphRAG)
        self._register_template(PromptTemplate(
            name="community_summary_v1",
            version="1.0",
            type=PromptType.SUMMARIZATION,
            description="Summarize entity community for global search",
            system_message="""You are an expert at summarizing knowledge graph communities.
Create summaries that capture the key themes and relationships.""",
            template="""Summarize this community of related entities.

Community Entities:
{entities}

Relationships:
{relationships}

Sample Text Mentions:
{text_samples}

Create a summary that captures:
1. What this community is about (main theme)
2. Key entities and their roles
3. Important relationships and patterns
4. Overall significance

Community Summary:""",
            use_cot=False
        ))

    def _register_template(self, template: PromptTemplate):
        """Register a template using name as key"""
        if template.type not in self._templates:
            self._templates[template.type] = {}

        # Use template name as key to avoid overwrites
        self._templates[template.type][template.name] = template

    def _load_from_directory(self):
        """Load templates from YAML files in directory"""
        for yaml_file in self.prompts_dir.glob("**/*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if isinstance(data, dict):
                    template = self._parse_template(data)
                    if template:
                        self._register_template(template)
                        logger.debug(f"Loaded template: {template.name}")

            except Exception as e:
                logger.warning(f"Failed to load prompt {yaml_file}: {e}")

    def _parse_template(self, data: Dict[str, Any]) -> Optional[PromptTemplate]:
        """Parse template from YAML data"""
        try:
            prompt_type = PromptType(data.get("type", "answer_generation"))

            return PromptTemplate(
                name=data.get("name", "unnamed"),
                version=data.get("version", "1.0"),
                type=prompt_type,
                template=data.get("template", ""),
                system_message=data.get("system_message", ""),
                description=data.get("description", ""),
                use_cot=data.get("use_cot", False),
                cot_steps=data.get("cot_steps", []),
                few_shot_examples=data.get("examples", []),
                max_examples=data.get("max_examples", 3),
                languages=data.get("languages", ["en", "ar"]),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            logger.warning(f"Failed to parse template: {e}")
            return None

    def get_template(
        self,
        prompt_type: PromptType,
        version: Optional[str] = None,
        use_cot: Optional[bool] = None,
        name: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """
        Get a prompt template.

        Args:
            prompt_type: Type of prompt
            version: Specific version to filter by
            use_cot: Filter by CoT setting
            name: Specific template name

        Returns:
            PromptTemplate or None
        """
        if prompt_type not in self._templates:
            return None

        templates = self._templates[prompt_type]

        # Get by specific name
        if name:
            return templates.get(name)

        # Get all templates for this type
        candidates = list(templates.values())

        # Filter by version if specified
        if version:
            candidates = [t for t in candidates if t.version == version]

        # Filter by CoT if specified
        if use_cot is not None:
            candidates = [t for t in candidates if t.use_cot == use_cot]

        if not candidates:
            return None

        # Return latest version
        return sorted(candidates, key=lambda t: t.version, reverse=True)[0]

    def format_prompt(
        self,
        prompt_type: PromptType,
        variables: Dict[str, Any],
        version: Optional[str] = None,
        use_cot: bool = False,
        include_examples: bool = True,
        language: Optional[str] = None
    ) -> FormattedPrompt:
        """
        Format a prompt with variables.

        Args:
            prompt_type: Type of prompt
            variables: Variables to substitute
            version: Template version
            use_cot: Use chain-of-thought version
            include_examples: Include few-shot examples
            language: Language preference

        Returns:
            FormattedPrompt ready for LLM
        """
        template = self.get_template(prompt_type, version, use_cot)

        if not template:
            raise ValueError(f"No template found for {prompt_type.value}")

        # Build user message
        user_message = template.template

        # Add few-shot examples if requested
        if include_examples and template.few_shot_examples:
            examples_text = self._format_examples(
                template.few_shot_examples[:template.max_examples]
            )
            user_message = examples_text + "\n\n" + user_message

        # Substitute variables
        try:
            user_message = user_message.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in prompt: {e}")
            # Do partial substitution
            for key, value in variables.items():
                user_message = user_message.replace(f"{{{key}}}", str(value))

        return FormattedPrompt(
            system_message=template.system_message,
            user_message=user_message,
            template_name=template.name,
            template_version=template.version,
            variables_used=variables
        )

    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples"""
        if not examples:
            return ""

        formatted = ["Here are some examples:"]

        for i, ex in enumerate(examples, 1):
            formatted.append(f"\nExample {i}:")
            if "input" in ex:
                formatted.append(f"Input: {ex['input']}")
            if "output" in ex:
                formatted.append(f"Output: {ex['output']}")

        return "\n".join(formatted)

    def list_templates(
        self,
        prompt_type: Optional[PromptType] = None
    ) -> List[Dict[str, Any]]:
        """List available templates"""
        results = []

        types_to_check = [prompt_type] if prompt_type else list(self._templates.keys())

        for ptype in types_to_check:
            if ptype not in self._templates:
                continue

            for name, template in self._templates[ptype].items():
                results.append({
                    "name": template.name,
                    "type": ptype.value,
                    "version": template.version,
                    "description": template.description,
                    "use_cot": template.use_cot,
                    "languages": template.languages
                })

        return results

    def create_qa_prompt(
        self,
        question: str,
        context: List[Dict[str, Any]],
        use_cot: bool = False,
        max_context_length: int = 4000
    ) -> FormattedPrompt:
        """
        Create a question-answering prompt.

        Args:
            question: User question
            context: Retrieved context chunks
            use_cot: Use chain-of-thought
            max_context_length: Maximum context length

        Returns:
            Formatted prompt for QA
        """
        # Format context with citations
        context_parts = []
        for i, chunk in enumerate(context, 1):
            text = chunk.get("text", "")
            doc_id = chunk.get("document_id", "unknown")
            context_parts.append(f"[{i}] ({doc_id}): {text}")

        context_str = "\n\n".join(context_parts)

        # Truncate if needed
        if len(context_str) > max_context_length:
            context_str = context_str[:max_context_length] + "..."

        return self.format_prompt(
            PromptType.ANSWER_GENERATION,
            variables={
                "question": question,
                "context": context_str,
                "cot_placeholder": "" if not use_cot else "[thinking...]"
            },
            use_cot=use_cot
        )

    def create_extraction_prompt(
        self,
        text: str,
        extraction_type: str = "entities",
        use_cot: bool = False
    ) -> FormattedPrompt:
        """
        Create an extraction prompt.

        Args:
            text: Text to extract from
            extraction_type: "entities" or "relationships"
            use_cot: Use chain-of-thought

        Returns:
            Formatted prompt for extraction
        """
        if extraction_type == "entities":
            prompt_type = PromptType.ENTITY_EXTRACTION
        else:
            prompt_type = PromptType.RELATIONSHIP_EXTRACTION

        return self.format_prompt(
            prompt_type,
            variables={"text": text, "entities": ""},
            use_cot=use_cot
        )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(prompts_dir: Optional[str] = None) -> PromptManager:
    """Get or create global PromptManager"""
    global _prompt_manager

    if _prompt_manager is None:
        _prompt_manager = PromptManager(prompts_dir=prompts_dir)

    return _prompt_manager
