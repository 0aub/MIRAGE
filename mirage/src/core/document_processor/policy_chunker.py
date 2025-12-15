"""
Policy Document Chunker
=======================
Specialized chunker for policy documents (regulations, laws, guidelines).
Preserves document structure: sections, articles, clauses.
Supports both English and Arabic policy formats.
"""

import re
from typing import List, Dict, Any, Optional
from loguru import logger


class PolicyChunker:
    """
    Chunk policy documents while preserving hierarchical structure.

    Features:
    - Detects section headers (Section 1, Article 1, المادة الأولى)
    - Keeps articles/clauses together
    - Adds section metadata to each chunk
    - Handles numbered lists and definitions
    """

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        overlap_sentences: int = 1
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences

        # English section patterns
        self.en_section_patterns = [
            r'^(?:SECTION|Section)\s+(\d+)[\s:.-]*(.*)$',
            r'^(?:ARTICLE|Article)\s+(\d+)[\s:.-]*(.*)$',
            r'^(?:CHAPTER|Chapter)\s+(\d+)[\s:.-]*(.*)$',
            r'^(?:PART|Part)\s+(\d+)[\s:.-]*(.*)$',
            r'^(\d+)\.\s+([A-Z][A-Za-z\s]+)$',  # "1. Data Classification"
            r'^(\d+\.\d+)\s+(.+)$',  # "1.1 Subsection"
        ]

        # Arabic section patterns
        self.ar_section_patterns = [
            r'^المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|الحادية\s+عشر(?:ة)?|الثانية\s+عشر(?:ة)?|الثالثة\s+عشر(?:ة)?|الرابعة\s+عشر(?:ة)?|الخامسة\s+عشر(?:ة)?|رقم\s*\(?\d+\)?)',
            r'^الباب\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)',
            r'^الفصل\s+(?:الأول|الثاني|الثالث|الرابع|الخامس)',
            r'^(?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)',
            r'^(?:البند|الفقرة)\s*(?:\d+|الأول(?:ى)?|الثاني(?:ة)?)',
        ]

        # Definition patterns
        self.definition_patterns = [
            r'^"?([^":]+)"?\s*[:–-]\s*(.+)$',  # "Term": Definition
            r'^([أ-ي\s]+)\s*[:–-]\s*(.+)$',  # Arabic term: definition
        ]

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        document_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text (compatible interface with other chunkers).
        Alias for chunk_policy_document.
        """
        return self.chunk_policy_document(text, document_id, metadata)

    def chunk_policy_document(
        self,
        text: str,
        document_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a policy document while preserving structure.

        Args:
            text: Full document text
            document_id: Document identifier for chunk IDs
            metadata: Additional metadata for chunks

        Returns:
            List of chunks with structure metadata
        """
        logger.info(f"Policy chunking document: {len(text)} chars")

        # Split into sections first
        sections = self._split_into_sections(text)
        logger.debug(f"Found {len(sections)} sections")

        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(
                section_text=section["text"],
                section_title=section.get("title", ""),
                section_level=section.get("level", 0),
                section_number=section.get("number", ""),
                start_index=chunk_index
            )

            for chunk in section_chunks:
                chunk["section"] = {
                    "title": section.get("title", ""),
                    "number": section.get("number", ""),
                    "level": section.get("level", 0),
                }
                if document_id:
                    chunk["id"] = f"{document_id}_chunk_{chunk_index}"
                if metadata:
                    chunk["document_metadata"] = metadata
                chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Created {len(chunks)} policy chunks from {len(sections)} sections")
        return chunks

    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sections based on headers."""
        lines = text.split('\n')
        sections = []
        current_section = {
            "title": "",
            "number": "",
            "level": 0,
            "lines": []
        }

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_section["lines"]:
                    current_section["lines"].append("")
                continue

            # Check if line is a section header
            header_info = self._detect_header(stripped)

            if header_info:
                # Save previous section if it has content
                if current_section["lines"]:
                    current_section["text"] = '\n'.join(current_section["lines"]).strip()
                    if current_section["text"]:
                        sections.append(current_section)

                # Start new section
                current_section = {
                    "title": header_info["title"],
                    "number": header_info["number"],
                    "level": header_info["level"],
                    "lines": [stripped]  # Include header in section
                }
            else:
                current_section["lines"].append(stripped)

        # Save final section
        if current_section["lines"]:
            current_section["text"] = '\n'.join(current_section["lines"]).strip()
            if current_section["text"]:
                sections.append(current_section)

        # If no sections found, treat whole document as one section
        if not sections:
            sections = [{
                "title": "",
                "number": "",
                "level": 0,
                "text": text.strip()
            }]

        return sections

    def _detect_header(self, line: str) -> Optional[Dict[str, Any]]:
        """Detect if a line is a section header."""
        # Check English patterns
        for i, pattern in enumerate(self.en_section_patterns):
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    "number": groups[0] if len(groups) > 0 else "",
                    "title": groups[1].strip() if len(groups) > 1 else line,
                    "level": i // 2 + 1  # Approximate hierarchy level
                }

        # Check Arabic patterns
        for i, pattern in enumerate(self.ar_section_patterns):
            match = re.match(pattern, line)
            if match:
                return {
                    "number": str(i + 1),
                    "title": line,
                    "level": 1
                }

        # Check for all-caps headers (common in policies)
        if line.isupper() and len(line) > 3 and len(line) < 100:
            return {
                "number": "",
                "title": line,
                "level": 1
            }

        return None

    def _chunk_section(
        self,
        section_text: str,
        section_title: str,
        section_level: int,
        section_number: str,
        start_index: int
    ) -> List[Dict[str, Any]]:
        """Chunk a single section."""
        if len(section_text) <= self.max_chunk_size:
            # Section fits in one chunk
            return [{
                "text": section_text,
                "char_count": len(section_text),
                "word_count": len(section_text.split()),
                "chunk_index": start_index,
            }]

        # Section too large - split by articles/clauses
        chunks = []
        paragraphs = self._split_into_paragraphs(section_text)

        current_chunk_text = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # If adding paragraph exceeds max, save current and start new
            if current_length + para_length > self.max_chunk_size and current_chunk_text:
                chunk_text = '\n\n'.join(current_chunk_text)
                chunks.append({
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "chunk_index": start_index + len(chunks),
                })

                # Keep last paragraph for overlap context
                if self.overlap_sentences > 0 and current_chunk_text:
                    current_chunk_text = [current_chunk_text[-1]]
                    current_length = len(current_chunk_text[0])
                else:
                    current_chunk_text = []
                    current_length = 0

            current_chunk_text.append(para)
            current_length += para_length

        # Save remaining content
        if current_chunk_text:
            chunk_text = '\n\n'.join(current_chunk_text)
            chunks.append({
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "chunk_index": start_index + len(chunks),
            })

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (articles, clauses, or semantic blocks)."""
        # Split by double newlines first
        blocks = re.split(r'\n\s*\n', text)
        paragraphs = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # If block is still too large, split by numbered items
            if len(block) > self.max_chunk_size:
                # Try to split by numbered lists
                numbered_parts = re.split(r'\n(?=\d+[\.\)]\s)', block)
                if len(numbered_parts) > 1:
                    paragraphs.extend([p.strip() for p in numbered_parts if p.strip()])
                else:
                    # Split by sentences as last resort
                    sentences = re.split(r'(?<=[.؟!])\s+', block)
                    current = []
                    current_len = 0
                    for sent in sentences:
                        if current_len + len(sent) > self.max_chunk_size and current:
                            paragraphs.append(' '.join(current))
                            current = []
                            current_len = 0
                        current.append(sent)
                        current_len += len(sent)
                    if current:
                        paragraphs.append(' '.join(current))
            else:
                paragraphs.append(block)

        return paragraphs


# Update chunking strategy config
POLICY_CHUNKING_CONFIG = {
    "strategy": "policy",
    "description": "Section-aware chunking for policy/regulation documents",
    "max_chunk_size": 1500,
    "min_chunk_size": 200,
    "overlap_sentences": 1,
    "detect_headers": True,
    "preserve_articles": True,
    "supported_languages": ["en", "ar"],
}
