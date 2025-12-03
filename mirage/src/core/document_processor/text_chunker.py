"""
Text Chunker
Splits text into semantic chunks with configurable size and overlap
Maintains sentence boundaries for better coherence
"""

import re
from typing import List, Dict, Any
from loguru import logger


class TextChunker:
    """Split text into overlapping chunks while preserving sentence boundaries"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize chunker

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Sentence boundary patterns (supports English and Arabic)
        self.sentence_endings = re.compile(
            r'[.!?؟।।\u0964\u0965]+[\s\n]+'  # Period, !, ?, Arabic question mark, etc.
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None, document_id: str = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            document_id: Optional document ID for generating deterministic chunk IDs

        Returns:
            List of chunk dictionaries with text, position, metadata, and unique IDs
        """
        if not text or not text.strip():
            return []

        logger.debug(f"Chunking text of length {len(text)}")

        # Split into sentences first
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Build chunks from sentences
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "sentence_count": len(current_chunk),
                    "chunk_index": len(chunks),
                })

                # Start new chunk with overlap
                # Calculate how many sentences to keep for overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, self.chunk_overlap)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "sentence_count": len(current_chunk),
                "chunk_index": len(chunks),
            })

        # Add metadata to all chunks
        if metadata:
            for chunk in chunks:
                chunk["metadata"] = metadata

        # Add deterministic chunk IDs for hybrid vector-graph architecture
        for i, chunk in enumerate(chunks):
            if document_id:
                chunk["id"] = f"{document_id}_chunk_{i}"
            else:
                import uuid
                chunk["id"] = str(uuid.uuid4())

        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split by sentence endings
        sentences = self.sentence_endings.split(text)

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no sentence boundaries found, split by newlines
        if len(sentences) == 1 and '\n' in text:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]

        # If still one big block, split by approximate sentence length
        if len(sentences) == 1 and len(sentences[0]) > self.chunk_size:
            sentences = self._split_long_text(sentences[0])

        return sentences

    def _split_long_text(self, text: str, max_length: int = None) -> List[str]:
        """
        Split long text that has no sentence boundaries

        Args:
            text: Text to split
            max_length: Maximum length per segment

        Returns:
            List of text segments
        """
        if max_length is None:
            max_length = self.chunk_size

        # Try to split on other delimiters
        segments = []
        words = text.split()

        current_segment = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > max_length and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = [word]
                current_length = word_length
            else:
                current_segment.append(word)
                current_length += word_length

        if current_segment:
            segments.append(' '.join(current_segment))

        return segments

    def _get_overlap_sentences(self, sentences: List[str], target_overlap: int) -> List[str]:
        """
        Get sentences from the end to create overlap

        Args:
            sentences: List of sentences
            target_overlap: Target overlap size in characters

        Returns:
            List of sentences for overlap
        """
        overlap_sentences = []
        overlap_length = 0

        # Take sentences from the end until we reach target overlap
        for sentence in reversed(sentences):
            sentence_length = len(sentence)

            if overlap_length + sentence_length > target_overlap:
                break

            overlap_sentences.insert(0, sentence)
            overlap_length += sentence_length

        return overlap_sentences

    def chunk_document(
        self,
        document: Dict[str, Any],
        text_field: str = "text"
    ) -> Dict[str, Any]:
        """
        Chunk a processed document

        Args:
            document: Processed document dict
            text_field: Field containing the text to chunk

        Returns:
            Document with added chunks field
        """
        text = document.get(text_field, "")

        if not text:
            logger.warning("No text found in document for chunking")
            document["chunks"] = []
            return document

        # Extract metadata for chunks
        metadata = {
            "source": document.get("metadata", {}),
            "structure": document.get("structure", {}),
        }

        # Extract document_id if available for chunk ID generation
        document_id = document.get("document_id") or document.get("id")

        # Create chunks
        chunks = self.chunk_text(text, metadata, document_id)

        # Add to document
        document["chunks"] = chunks
        document["chunk_count"] = len(chunks)

        return document
