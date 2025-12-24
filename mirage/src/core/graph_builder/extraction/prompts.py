"""
LLM Entity Extraction - Prompt Templates
System and user prompts for entity and relationship extraction
"""


def get_extraction_prompt(language: str) -> tuple:
    """
    Get system and user prompt templates for entity extraction.

    Args:
        language: "ar" for Arabic, "en" for English

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    if language == "ar":
        system_prompt = """أنت مستخرج معرفة متخصص لبناء رسم بياني معرفي (Knowledge Graph).
مهمتك: استخراج الكيانات المسماة والعلاقات بينها. العلاقات مهمة جداً!

**أنواع الكيانات:**
- Organization: منظمات، شركات، مؤسسات، جهات حكومية
- Person: أشخاص، مناصب، أدوار
- Location: أماكن، مدن، دول، مناطق
- Concept: مفاهيم، مصطلحات، تعريفات
- Event: أحداث، مناسبات، تواريخ مهمة
- Product: منتجات، خدمات، أنظمة
- Document: وثائق، سياسات، قوانين، تقارير

**أنواع العلاقات (مهم جداً!):**
- RELATED_TO: علاقة عامة بين كيانين
- PART_OF: كيان جزء من كيان أكبر
- CONTAINS: كيان يحتوي على كيان آخر
- CAUSES: كيان يسبب شيء آخر
- USES: كيان يستخدم كيان آخر
- CREATED_BY: كيان أنشأه كيان آخر
- LOCATED_IN: كيان موجود في مكان
- BELONGS_TO: كيان ينتمي لكيان آخر
- DEPENDS_ON: كيان يعتمد على كيان آخر
- AFFECTS: كيان يؤثر على كيان آخر
- أو أي علاقة وصفية مناسبة للسياق

**صيغة JSON:**
{
  "entities": [{"text": "اسم", "type": "النوع", "importance": "high/medium/low"}],
  "relationships": [{"source": "المصدر", "target": "الهدف", "type": "RELATIONSHIP_TYPE", "weight": 0.8}]
}

مهم: استخرج علاقات بين الكيانات! لا تستخرج كيانات بدون علاقات."""

        user_template = """استخرج الكيانات والعلاقات بينها:

{chunk}

JSON:"""

    else:  # English
        system_prompt = """You are a knowledge extractor for building a Knowledge Graph.
Extract named entities AND relationships between them. Relationships are critical!

**Entity Types:**
- Organization: Companies, institutions, agencies, departments
- Person: People, roles, positions, titles
- Location: Places, cities, countries, regions
- Concept: Ideas, terms, definitions, topics
- Event: Events, dates, milestones, occurrences
- Product: Products, services, systems, tools
- Document: Documents, policies, laws, reports, papers

**Relationship Types (CRITICAL!):**
- RELATED_TO: General association between entities
- PART_OF: Entity is component/member of another
- CONTAINS: Entity contains another entity
- CAUSES: Entity causes/leads to something
- USES: Entity uses/utilizes another
- CREATED_BY: Entity was created by another
- LOCATED_IN: Entity is located in a place
- BELONGS_TO: Entity belongs to another
- DEPENDS_ON: Entity depends on another
- AFFECTS: Entity affects/impacts another
- WORKS_FOR: Person works for organization
- OWNS: Entity owns another entity
- PRODUCES: Entity produces/creates something
- Or any descriptive relationship fitting the context

**JSON format:**
{
  "entities": [{"text": "Name", "type": "Type", "importance": "high/medium/low"}],
  "relationships": [{"source": "Source", "target": "Target", "type": "RELATIONSHIP_TYPE", "weight": 0.8}]
}

IMPORTANT: Extract relationships between entities! Do not return entities without relationships."""

        user_template = """Extract entities AND relationships:

{chunk}

JSON:"""

    return system_prompt, user_template


def get_validation_prompt(language: str) -> tuple:
    """
    Get system and user prompt templates for entity validation.

    Args:
        language: "ar" for Arabic, "en" for English

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    if language == "ar":
        system_prompt = """أنت مدقق كيانات. راجع قائمة الكيانات وحدد أيها كيانات حقيقية (أسماء، منظمات، أماكن، برامج) وأيها مصطلحات عامة أو عبارات شائعة.

أجب بـ JSON فقط: {"valid": ["كيان1", "كيان2"], "invalid": ["مصطلح عام1"]}

الكيانات العامة التي يجب رفضها:
- العبارات الدينية (باذن الله، إن شاء الله)
- المصطلحات العامة (البيانات، التشغيل، السياسات)
- الكلمات الوصفية المجردة"""

        user_template = """راجع هذه الكيانات وصنفها:

{entities}

JSON:"""

    else:
        system_prompt = """You are an entity validator. Review the list of entities and identify which are real named entities (names, organizations, places, programs) and which are generic terms or common phrases.

Respond with JSON only: {"valid": ["entity1", "entity2"], "invalid": ["generic term1"]}

Generic entities to reject:
- Religious/cultural phrases
- Generic operational terms (data, system, process, policy)
- Abstract descriptive words"""

        user_template = """Review these entities and classify them:

{entities}

JSON:"""

    return system_prompt, user_template
