"""
LLM Entity Extraction - JSON Parser
Parse and repair JSON from LLM responses
"""

import json
import re
from typing import Dict, Any, Optional
from loguru import logger


def parse_json_with_repair(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON with automatic repair for common LLM output errors.

    Tries multiple repair strategies for malformed JSON:
    - Unterminated strings
    - Missing/extra commas
    - Truncated output

    Args:
        content: Raw JSON string from LLM

    Returns:
        Parsed JSON dict or None if all repair attempts fail
    """
    # Strategy 1: Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object boundaries
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_content = content[start:end+1]
            return json.loads(json_content)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Fix common issues
    try:
        repaired = content

        # Remove trailing commas before } or ]
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)

        # Fix unterminated strings
        if repaired.count('"') % 2 != 0:
            last_brace = repaired.rfind('}')
            if last_brace != -1:
                repaired = repaired[:last_brace] + '"' + repaired[last_brace:]

        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Truncate to last complete item
    try:
        last_complete = max(
            content.rfind('  }'),
            content.rfind('}\n'),
        )

        if last_complete != -1:
            truncated = content[:last_complete + 3]

            open_braces = truncated.count('{') - truncated.count('}')
            open_brackets = truncated.count('[') - truncated.count(']')

            for _ in range(open_brackets):
                truncated += '\n  ]'
            for _ in range(open_braces):
                truncated += '\n}'

            return json.loads(truncated)
    except json.JSONDecodeError:
        pass

    logger.warning("JSON repair failed after 4 strategies")
    return None


def clean_json_wrapper(content: str) -> str:
    """
    Clean LLM response by removing markdown code blocks.

    Args:
        content: Raw LLM response

    Returns:
        Cleaned JSON string
    """
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


def normalize_extraction_result(result: Any) -> Dict[str, Any]:
    """
    Normalize LLM extraction result to standard format.

    Handles cases where LLM returns:
    - A list instead of dict
    - Nested entities

    Args:
        result: Parsed JSON result

    Returns:
        Normalized dict with 'entities' and 'relationships' lists
    """
    if result is None:
        return {"entities": [], "relationships": []}

    # Handle list returned instead of dict
    if isinstance(result, list):
        if all(isinstance(item, dict) for item in result):
            return {"entities": result, "relationships": []}
        else:
            logger.warning(f"LLM returned unexpected list format")
            return {"entities": [], "relationships": []}

    if not isinstance(result, dict):
        logger.warning(f"LLM returned unexpected type: {type(result)}")
        return {"entities": [], "relationships": []}

    # Handle nested entities
    entities = result.get("entities", [])
    if isinstance(entities, dict) and "entities" in entities:
        entities = entities.get("entities", [])
    elif not isinstance(entities, list):
        entities = []

    relationships = result.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    return {"entities": entities, "relationships": relationships}
