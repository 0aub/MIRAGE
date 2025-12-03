"""
MIRAGE V2 Chunking - Base Classes
Defines the interface for all text chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import hashlib


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    TOKEN = "token"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    LATE = "late"


@dataclass
class Chunk:
    """A text chunk with metadata"""
    text: str
    index: int  # Position in document
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Generate unique chunk ID based on content"""
        content = f"{self.index}:{self.text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class ChunkingResult:
    """Result of chunking a document"""
    chunks: List[Chunk]
    original_length: int
    strategy: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def avg_chunk_size(self) -> float:
        if not self.chunks:
            return 0
        return sum(len(c) for c in self.chunks) / len(self.chunks)


class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.

    All chunking strategies must implement this interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize chunker with configuration.

        Args:
            config: Chunker-specific configuration
        """
        self.config = config or {}
        self._tokenizer = None

    @property
    def name(self) -> str:
        """Chunker name for logging"""
        return self.__class__.__name__

    @property
    @abstractmethod
    def strategy(self) -> ChunkingStrategy:
        """Return the chunking strategy type"""
        pass

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> ChunkingResult:
        """
        Split text into chunks.

        Args:
            text: Input text to chunk
            **kwargs: Additional parameters

        Returns:
            ChunkingResult with list of chunks
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self._tokenizer is None:
            self._init_tokenizer()

        if self._tokenizer:
            return len(self._tokenizer.encode(text))

        # Fallback: estimate ~4 chars per token
        return len(text) // 4

    def _init_tokenizer(self) -> None:
        """Initialize tokenizer for token counting"""
        try:
            import tiktoken
            tokenizer_name = self.config.get("tokenizer", "cl100k_base")
            self._tokenizer = tiktoken.get_encoding(tokenizer_name)
        except ImportError:
            self._tokenizer = None

    def _create_chunk(
        self,
        text: str,
        index: int,
        start_char: int = 0,
        end_char: int = 0,
        **metadata
    ) -> Chunk:
        """
        Create a chunk with metadata.

        Args:
            text: Chunk text
            index: Chunk position
            start_char: Start character offset
            end_char: End character offset
            **metadata: Additional metadata

        Returns:
            Chunk object
        """
        return Chunk(
            text=text,
            index=index,
            start_char=start_char,
            end_char=end_char if end_char else start_char + len(text),
            token_count=self.count_tokens(text),
            metadata=metadata
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get chunker statistics"""
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "config": self.config
        }


class TextSplitter:
    """
    Utility class for splitting text by various delimiters.
    Used by multiple chunking strategies.
    """

    # Common sentence-ending patterns
    SENTENCE_ENDINGS = [
        '. ', '! ', '? ',  # English
        '。', '！', '？',   # Chinese/Japanese
        '؟', '۔',          # Arabic/Persian
        '\n\n',            # Paragraph break
    ]

    # Arabic sentence endings
    ARABIC_ENDINGS = [
        '؟',  # Arabic question mark
        '۔',  # Arabic full stop
        '،',  # Arabic comma (can end clauses)
    ]

    @classmethod
    def split_sentences(cls, text: str, respect_arabic: bool = True) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text
            respect_arabic: Include Arabic sentence endings

        Returns:
            List of sentences
        """
        import re

        # Build pattern
        endings = cls.SENTENCE_ENDINGS.copy()
        if respect_arabic:
            endings.extend(cls.ARABIC_ENDINGS)

        # Escape special regex chars and join
        pattern = '|'.join(re.escape(e) for e in endings)

        # Split but keep delimiters
        parts = re.split(f'({pattern})', text)

        # Recombine sentences with their endings
        sentences = []
        current = ""

        for part in parts:
            current += part
            if part.strip() in [e.strip() for e in endings]:
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        # Don't forget last sentence
        if current.strip():
            sentences.append(current.strip())

        return sentences

    @classmethod
    def split_paragraphs(cls, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split on double newlines
        paragraphs = text.split('\n\n')

        # Clean up and filter empty
        return [p.strip() for p in paragraphs if p.strip()]

    @classmethod
    def split_by_separators(
        cls,
        text: str,
        separators: List[str],
        keep_separator: bool = False
    ) -> List[str]:
        """
        Split text by a list of separators in order.

        Args:
            text: Input text
            separators: List of separators to try
            keep_separator: Keep separator in output

        Returns:
            List of text segments
        """
        import re

        segments = [text]

        for separator in separators:
            new_segments = []
            for segment in segments:
                if separator in segment:
                    parts = segment.split(separator)
                    for i, part in enumerate(parts):
                        if part.strip():
                            if keep_separator and i < len(parts) - 1:
                                new_segments.append(part + separator)
                            else:
                                new_segments.append(part)
                else:
                    if segment.strip():
                        new_segments.append(segment)
            segments = new_segments

        return [s.strip() for s in segments if s.strip()]
