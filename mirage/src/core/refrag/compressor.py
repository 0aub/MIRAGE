"""
REFRAG Compressor
Main compression engine with multiple strategies
Achieves significant speedup through selective compression
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

from .policy import CompressionPolicy
from .cache import CompressionCache


class REFRAGCompressor:
    """
    REFRAG compression engine
    Selectively compresses text while maintaining semantic meaning
    """

    def __init__(
        self,
        policy: Optional[CompressionPolicy] = None,
        cache: Optional[CompressionCache] = None,
        strategy: str = "token_pruning",
    ):
        """
        Initialize REFRAG compressor

        Args:
            policy: Compression policy (creates default if None)
            cache: Compression cache (creates default if None)
            strategy: Compression strategy (token_pruning, extractive, hybrid)
        """
        self.policy = policy or CompressionPolicy()
        self.cache = cache or CompressionCache()
        self.strategy = strategy
        
        logger.info(
            f"Initialized REFRAGCompressor with strategy={strategy}"
        )

    def compress(
        self,
        chunks: List[Dict[str, Any]],
        query_context: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Compress text chunks using REFRAG

        Args:
            chunks: List of text chunks
            query_context: Optional query for relevance-aware compression
            force: Force recompression (skip cache)

        Returns:
            Compression result with compressed chunks and stats
        """
        if not chunks:
            return {
                "compressed_chunks": [],
                "original_length": 0,
                "compressed_length": 0,
                "compression_ratio": 1.0,
                "chunks_compressed": 0,
                "processing_time": 0.0,
            }

        start_time = time.time()

        # Get compression decisions from policy
        decisions = self.policy.decide_compression(chunks, query_context)

        # Compress each chunk according to policy
        compressed_chunks = []
        total_original_length = 0
        total_compressed_length = 0
        chunks_compressed = 0

        for decision in decisions:
            original_text = decision["text"]
            original_length = len(original_text)
            total_original_length += original_length

            should_compress = decision["should_compress"]
            compression_ratio = decision["compression_ratio"]

            if should_compress and not force:
                # Check cache first
                cached = self.cache.get(original_text, compression_ratio)
                if cached:
                    compressed_text = cached["compressed_text"]
                    compressed_chunks.append({
                        **decision,
                        "compressed_text": compressed_text,
                        "compressed_length": len(compressed_text),
                        "from_cache": True,
                    })
                    total_compressed_length += len(compressed_text)
                    chunks_compressed += 1
                    continue

            # Perform compression
            if should_compress:
                compressed_text = self._compress_text(
                    original_text,
                    compression_ratio,
                    self.strategy,
                )

                # Cache the result
                self.cache.set(
                    original_text,
                    compression_ratio,
                    {"compressed_text": compressed_text},
                )

                compressed_chunks.append({
                    **decision,
                    "compressed_text": compressed_text,
                    "compressed_length": len(compressed_text),
                    "from_cache": False,
                })
                total_compressed_length += len(compressed_text)
                chunks_compressed += 1
            else:
                # No compression needed
                compressed_chunks.append({
                    **decision,
                    "compressed_text": original_text,
                    "compressed_length": original_length,
                    "from_cache": False,
                })
                total_compressed_length += original_length

        processing_time = time.time() - start_time

        # Calculate overall statistics
        actual_compression_ratio = (
            total_compressed_length / total_original_length
            if total_original_length > 0
            else 1.0
        )

        result = {
            "compressed_chunks": compressed_chunks,
            "original_length": total_original_length,
            "compressed_length": total_compressed_length,
            "compression_ratio": actual_compression_ratio,
            "chunks_compressed": chunks_compressed,
            "total_chunks": len(chunks),
            "processing_time": processing_time,
            "strategy": self.strategy,
        }

        logger.info(
            f"Compressed {chunks_compressed}/{len(chunks)} chunks: "
            f"{total_original_length} → {total_compressed_length} chars "
            f"(ratio={actual_compression_ratio:.2f}, time={processing_time:.3f}s)"
        )

        return result

    def _compress_text(
        self,
        text: str,
        compression_ratio: float,
        strategy: str,
    ) -> str:
        """
        Compress a single text using specified strategy

        Args:
            text: Text to compress
            compression_ratio: Target ratio (0-1)
            strategy: Compression strategy

        Returns:
            Compressed text
        """
        if compression_ratio >= 1.0:
            return text

        if strategy == "token_pruning":
            return self._token_pruning(text, compression_ratio)
        elif strategy == "extractive":
            return self._extractive_compression(text, compression_ratio)
        elif strategy == "hybrid":
            # Use extractive for high compression, token pruning for low
            if compression_ratio < 0.5:
                return self._extractive_compression(text, compression_ratio)
            else:
                return self._token_pruning(text, compression_ratio)
        else:
            logger.warning(f"Unknown strategy {strategy}, using token pruning")
            return self._token_pruning(text, compression_ratio)

    def _token_pruning(self, text: str, compression_ratio: float) -> str:
        """
        Compress by removing low-importance tokens

        Args:
            text: Text to compress
            compression_ratio: Target ratio

        Returns:
            Compressed text
        """
        # Split into tokens (words)
        tokens = text.split()
        
        if not tokens:
            return text

        # Calculate target token count
        target_count = int(len(tokens) * compression_ratio)
        target_count = max(1, target_count)  # Keep at least 1 token

        if target_count >= len(tokens):
            return text

        # Simple token importance scoring
        # Keep: nouns, verbs, named entities (approximated by capitalization)
        # Remove: common words, articles, prepositions
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            # Arabic stop words
            'في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي',
            'أن', 'كان', 'يكون', 'هو', 'هي', 'ما', 'لا', 'أو', 'و',
        }

        # Score each token
        token_scores = []
        for i, token in enumerate(tokens):
            score = 0.5  # Base score
            
            # Capitalized = likely important (named entity)
            if token and token[0].isupper():
                score += 0.3
            
            # Not a stop word
            if token.lower() not in stop_words:
                score += 0.2
            
            # Longer tokens often more meaningful
            if len(token) > 5:
                score += 0.1
            
            # Position: beginning and end important
            if i < 3 or i >= len(tokens) - 3:
                score += 0.15
            
            token_scores.append((i, token, score))

        # Sort by score and keep top tokens
        token_scores.sort(key=lambda x: x[2], reverse=True)
        top_tokens = token_scores[:target_count]
        
        # Sort by original position to maintain order
        top_tokens.sort(key=lambda x: x[0])
        
        # Reconstruct text
        compressed_text = ' '.join([token for _, token, _ in top_tokens])
        
        return compressed_text

    def _extractive_compression(
        self,
        text: str,
        compression_ratio: float,
    ) -> str:
        """
        Compress by extracting most important sentences

        Args:
            text: Text to compress
            compression_ratio: Target ratio

        Returns:
            Compressed text
        """
        # Split into sentences
        import re
        sentence_endings = re.compile(r'[.!?؟।।\u0964\u0965]+[\s\n]+')
        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text

        # Calculate target sentence count
        target_count = int(len(sentences) * compression_ratio)
        target_count = max(1, target_count)

        if target_count >= len(sentences):
            return text

        # Score sentences (first and last are important)
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0.5
            
            # Position importance
            if i == 0:
                score += 0.3  # First sentence very important
            elif i == len(sentences) - 1:
                score += 0.2  # Last sentence important
            
            # Length (not too short, not too long)
            sent_len = len(sentence.split())
            if 5 <= sent_len <= 30:
                score += 0.2
            
            # Contains named entities (capitalized words)
            capitals = sum(1 for word in sentence.split() if word and word[0].isupper())
            score += min(0.2, capitals * 0.05)
            
            sentence_scores.append((i, sentence, score))

        # Keep top sentences
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        top_sentences = sentence_scores[:target_count]
        
        # Sort by original position
        top_sentences.sort(key=lambda x: x[0])
        
        # Reconstruct text
        compressed_text = '. '.join([sent for _, sent, _ in top_sentences])
        if compressed_text and not compressed_text.endswith('.'):
            compressed_text += '.'
        
        return compressed_text

    def get_stats(self) -> Dict[str, Any]:
        """Get compressor statistics"""
        return {
            "strategy": self.strategy,
            "policy_stats": self.policy.get_stats(),
            "cache_stats": self.cache.get_stats(),
        }
