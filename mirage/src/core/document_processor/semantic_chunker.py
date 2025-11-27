"""
Semantic Text Chunker
Uses Jina embeddings to create semantically coherent chunks
Maintains semantic boundaries rather than just sentence boundaries
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

from ..embeddings import JinaEmbedder
from ...config.config_loader import get_config_loader


class SemanticChunker:
    """
    Split text into semantically coherent chunks using embeddings
    Groups sentences based on semantic similarity
    """

    def __init__(
        self,
        embedder: Optional[JinaEmbedder] = None,
        chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        min_sentences_per_chunk: Optional[int] = None,
    ):
        """
        Initialize semantic chunker

        Args:
            embedder: Jina embedder instance (creates one if not provided)
            chunk_size: Target chunk size in characters (loaded from config if not provided)
            max_chunk_size: HARD maximum chunk size (loaded from config if not provided)
            similarity_threshold: Minimum similarity to group sentences (loaded from config if not provided)
            min_sentences_per_chunk: Minimum sentences per chunk (loaded from config if not provided)
        """
        # Load configuration
        config_loader = get_config_loader()
        chunking_config = config_loader.get_semantic_chunking_config()

        self.embedder = embedder or JinaEmbedder()
        # Optimized chunk sizes for Qwen3-4B-Instruct (128K context window)
        # Arabic: ~2-3 chars/token, so 1200 chars ≈ 400-600 tokens
        # With 128K context, we have plenty of room for prompt + chunk + response
        # Target: 800 chars (~300-400 tokens), Max: 1200 chars (~500-600 tokens)
        self.chunk_size = chunk_size or chunking_config.get("chunk_size", 800)
        self.max_chunk_size = max_chunk_size or chunking_config.get("max_chunk_size", 1200)
        self.similarity_threshold = similarity_threshold or chunking_config.get("similarity_threshold", 0.7)
        self.min_sentences_per_chunk = min_sentences_per_chunk or chunking_config.get("min_sentences_per_chunk", 2)

        logger.debug(
            f"Initialized SemanticChunker: chunk_size={self.chunk_size}, "
            f"max_chunk_size={self.max_chunk_size}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"min_sentences={self.min_sentences_per_chunk}"
        )

        # Sentence boundary patterns (supports English and Arabic)
        self.sentence_endings = re.compile(
            r'[.!?؟।।\u0964\u0965]+[\s\n]+'
        )

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text, embeddings, and metadata
        """
        if not text or not text.strip():
            return []

        logger.debug(f"Semantic chunking text of length {len(text)}")

        # Split into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # If very few sentences, just return as single chunk
        if len(sentences) < self.min_sentences_per_chunk:
            return self._create_single_chunk(sentences, metadata)

        # Compute embeddings for all sentences
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        sentence_embeddings = self.embedder.embed(
            sentences,
            task="retrieval.passage",
            normalize=True,
        )

        # Group sentences into semantic chunks
        chunks = self._group_sentences_by_similarity(
            sentences,
            sentence_embeddings,
        )

        # Add metadata and statistics
        chunk_dicts = []
        for i, chunk_sentences in enumerate(chunks):
            chunk_text = ' '.join(chunk_sentences)
            
            # Compute chunk embedding (average of sentence embeddings)
            chunk_indices = self._get_sentence_indices(chunk_sentences, sentences)
            chunk_embedding = np.mean(
                sentence_embeddings[chunk_indices],
                axis=0,
            )

            chunk_dict = {
                "text": chunk_text,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "sentence_count": len(chunk_sentences),
                "chunk_index": i,
                "embedding": chunk_embedding.tolist(),
            }

            if metadata:
                chunk_dict["metadata"] = metadata

            chunk_dicts.append(chunk_dict)

        logger.info(
            f"Created {len(chunk_dicts)} semantic chunks "
            f"from {len(sentences)} sentences"
        )

        return chunk_dicts

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

        # If still one big block, split by words to approximate sentences
        if len(sentences) == 1 and len(sentences[0]) > self.chunk_size:
            sentences = self._split_long_text(sentences[0])

        return sentences

    def _split_long_text(self, text: str) -> List[str]:
        """
        Split long text that has no sentence boundaries

        Args:
            text: Text to split

        Returns:
            List of text segments
        """
        segments = []
        words = text.split()

        current_segment = []
        current_length = 0
        target_length = self.chunk_size // 2  # Smaller segments for better semantics

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > target_length and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = [word]
                current_length = word_length
            else:
                current_segment.append(word)
                current_length += word_length

        if current_segment:
            segments.append(' '.join(current_segment))

        return segments

    def _group_sentences_by_similarity(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
    ) -> List[List[str]]:
        """
        Group sentences into chunks based on semantic similarity

        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings (N x D)

        Returns:
            List of sentence groups (chunks)
        """
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = len(sentence)
            sentence_embedding = embeddings[i]

            # CRITICAL: Enforce HARD maximum chunk size
            # This check MUST come first and cannot be overridden
            if current_length + sentence_length > self.max_chunk_size:
                # Exceeds hard limit - MUST start new chunk
                logger.debug(
                    f"Chunk {len(chunks)} reached max size "
                    f"({current_length} chars, {len(current_chunk)} sentences) - starting new chunk"
                )
                chunks.append(current_chunk)
                current_chunk = [sentence]
                current_embedding = sentence_embedding
                current_length = sentence_length
                continue

            # Compute similarity with current chunk
            # (average embedding of sentences in chunk)
            similarity = float(np.dot(current_embedding, sentence_embedding))

            # Decide whether to add to current chunk or start new one
            should_add = (
                similarity >= self.similarity_threshold and
                current_length + sentence_length <= self.chunk_size
            )

            # Always add if chunk is too small (but only if we haven't exceeded max)
            if len(current_chunk) < self.min_sentences_per_chunk:
                should_add = True

            if should_add:
                # Add to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length

                # Update chunk embedding (running average)
                current_embedding = (
                    current_embedding * (len(current_chunk) - 1) +
                    sentence_embedding
                ) / len(current_chunk)
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk)

                current_chunk = [sentence]
                current_embedding = sentence_embedding
                current_length = sentence_length

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"Grouped {len(sentences)} sentences into {len(chunks)} chunks "
            f"(max_chunk_size={self.max_chunk_size} chars enforced)"
        )

        return chunks

    def _create_single_chunk(
        self,
        sentences: List[str],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Create a single chunk from all sentences
        If total length exceeds max_chunk_size, split into multiple chunks
        """
        chunk_text = ' '.join(sentences)

        # Check if it exceeds max_chunk_size
        if len(chunk_text) > self.max_chunk_size:
            logger.warning(
                f"Single chunk ({len(chunk_text)} chars) exceeds max_chunk_size "
                f"({self.max_chunk_size} chars) - splitting into multiple chunks"
            )
            # Split the text into smaller chunks
            chunks = []
            current_text = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence)

                # If single sentence exceeds max, we have to include it anyway
                # but log a warning
                if sentence_length > self.max_chunk_size:
                    logger.warning(
                        f"Single sentence ({sentence_length} chars) exceeds "
                        f"max_chunk_size ({self.max_chunk_size} chars) - including anyway"
                    )
                    if current_text:
                        # Save current chunk first
                        text = ' '.join(current_text)
                        embedding = self.embedder.embed(text)
                        chunks.append({
                            "text": text,
                            "char_count": len(text),
                            "word_count": len(text.split()),
                            "sentence_count": len(current_text),
                            "chunk_index": len(chunks),
                            "embedding": embedding.tolist(),
                        })
                        current_text = []
                        current_length = 0

                    # Add oversized sentence as its own chunk
                    embedding = self.embedder.embed(sentence)
                    chunks.append({
                        "text": sentence,
                        "char_count": sentence_length,
                        "word_count": len(sentence.split()),
                        "sentence_count": 1,
                        "chunk_index": len(chunks),
                        "embedding": embedding.tolist(),
                    })
                    continue

                # Normal case: add sentence if it fits
                if current_length + sentence_length > self.max_chunk_size and current_text:
                    # Save current chunk and start new one
                    text = ' '.join(current_text)
                    embedding = self.embedder.embed(text)
                    chunks.append({
                        "text": text,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                        "sentence_count": len(current_text),
                        "chunk_index": len(chunks),
                        "embedding": embedding.tolist(),
                    })
                    current_text = [sentence]
                    current_length = sentence_length
                else:
                    current_text.append(sentence)
                    current_length += sentence_length

            # Add final chunk
            if current_text:
                text = ' '.join(current_text)
                embedding = self.embedder.embed(text)
                chunks.append({
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "sentence_count": len(current_text),
                    "chunk_index": len(chunks),
                    "embedding": embedding.tolist(),
                })

            # Add metadata to all chunks
            if metadata:
                for chunk in chunks:
                    chunk["metadata"] = metadata

            return chunks

        # Normal case: fits in single chunk
        embedding = self.embedder.embed(chunk_text)

        chunk = {
            "text": chunk_text,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "sentence_count": len(sentences),
            "chunk_index": 0,
            "embedding": embedding.tolist(),
        }

        if metadata:
            chunk["metadata"] = metadata

        return [chunk]

    def _get_sentence_indices(
        self,
        chunk_sentences: List[str],
        all_sentences: List[str],
    ) -> List[int]:
        """Get indices of chunk sentences in the full sentence list"""
        indices = []
        for chunk_sent in chunk_sentences:
            try:
                idx = all_sentences.index(chunk_sent)
                indices.append(idx)
            except ValueError:
                continue
        return indices

    def chunk_document(
        self,
        document: Dict[str, Any],
        text_field: str = "text",
    ) -> Dict[str, Any]:
        """
        Chunk a processed document using semantic chunking

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

        # Create semantic chunks
        chunks = self.chunk_text(text, metadata)

        # Add to document
        document["chunks"] = chunks
        document["chunk_count"] = len(chunks)

        return document
