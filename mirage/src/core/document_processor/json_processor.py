"""
JSON Processor
Intelligently extracts text-bearing fields from JSON files
Skips IDs, timestamps, and other non-content fields
"""

import json
from typing import Dict, Any, List
from loguru import logger


class JSONProcessor:
    """Process JSON files and extract meaningful text content"""

    def __init__(self):
        self.supported_extensions = [".json"]

        # Fields to skip (non-content)
        self.skip_patterns = [
            'id', '_id', 'uuid', 'guid',  # IDs
            'created', 'updated', 'modified', 'timestamp', 'date',  # Timestamps
            'count', 'index', 'position', 'order',  # Counters
            'url', 'uri', 'link', 'href',  # URLs (unless we want them)
            'token', 'key', 'secret', 'password',  # Sensitive data
        ]

        # Minimum text length to consider
        self.min_text_length = 3

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text content from JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            Dict containing:
                - text: Extracted text from all text fields
                - metadata: JSON structure info
                - structure: Nested structure with extracted fields
        """
        logger.info(f"Processing JSON: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract all text fields
            extracted = self._extract_text_fields(data)

            # Build text output
            text_parts = []
            for field_path, value in extracted:
                # Format: field_path: value
                text_parts.append(f"{field_path}: {value}")

            full_text = "\n".join(text_parts)

            # Metadata
            metadata = {
                "structure_type": type(data).__name__,
                "is_array": isinstance(data, list),
                "total_fields_extracted": len(extracted),
            }

            if isinstance(data, list):
                metadata["array_length"] = len(data)
                metadata["item_type"] = type(data[0]).__name__ if data else "empty"

            result = {
                "text": full_text,
                "metadata": metadata,
                "structure": {
                    "extracted_fields": [
                        {"path": path, "value_preview": str(value)[:100]}
                        for path, value in extracted[:50]  # Limit preview
                    ],
                    "total_extracted": len(extracted),
                },
            }

            logger.info(
                f"Successfully processed JSON: {len(extracted)} fields extracted, "
                f"{len(full_text)} chars"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing JSON {file_path}: {e}")
            raise

    def _extract_text_fields(
        self,
        data: Any,
        parent_path: str = "",
        extracted: List = None
    ) -> List[tuple]:
        """
        Recursively extract text-bearing fields from nested structure

        Args:
            data: JSON data (dict, list, or primitive)
            parent_path: Current path in the structure
            extracted: Accumulated extracted fields

        Returns:
            List of (field_path, value) tuples
        """
        if extracted is None:
            extracted = []

        if isinstance(data, dict):
            for key, value in data.items():
                field_path = f"{parent_path}.{key}" if parent_path else key

                # Skip if field name matches skip patterns
                if self._should_skip_field(key):
                    continue

                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    self._extract_text_fields(value, field_path, extracted)
                # Extract text values
                elif isinstance(value, str) and len(value) >= self.min_text_length:
                    extracted.append((field_path, value))
                # Convert other primitives to string if meaningful
                elif isinstance(value, (int, float, bool)) and value is not None:
                    # Only include if it's a descriptive field
                    if any(keyword in key.lower() for keyword in ['count', 'total', 'amount', 'number']):
                        extracted.append((field_path, str(value)))

        elif isinstance(data, list):
            for idx, item in enumerate(data):
                item_path = f"{parent_path}[{idx}]" if parent_path else f"[{idx}]"
                self._extract_text_fields(item, item_path, extracted)

        return extracted

    def _should_skip_field(self, field_name: str) -> bool:
        """Check if field should be skipped based on name"""
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in self.skip_patterns)

    def is_supported(self, filename: str) -> bool:
        """Check if file extension is supported"""
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)
