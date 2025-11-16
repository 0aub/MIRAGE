"""
Document Processor Module
Phase 1: Data Ingestion & Preprocessing

Handles:
- PDF processing (PyMuPDF)
- HTML processing (BeautifulSoup + readability)
- JSON processing (intelligent field extraction)
- Text normalization and chunking
"""

from .pdf_processor import PDFProcessor
from .html_processor import HTMLProcessor
from .json_processor import JSONProcessor
from .text_chunker import TextChunker
from .semantic_chunker import SemanticChunker
from .chunker_factory import ChunkerFactory, get_chunker_factory

__all__ = [
    "PDFProcessor",
    "HTMLProcessor",
    "JSONProcessor",
    "TextChunker",
    "SemanticChunker",
    "ChunkerFactory",
    "get_chunker_factory",
]
