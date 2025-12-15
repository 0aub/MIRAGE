"""
PDF Processor
Extracts text from PDF files using PyMuPDF while preserving structure
Handles Arabic RTL text properly
"""

import pymupdf
from typing import Dict, List, Any
from loguru import logger


class PDFProcessor:
    """Process PDF files and extract clean text with metadata"""

    def __init__(self):
        self.supported_extensions = [".pdf"]

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF file

        Args:
            file_path: Path to PDF file

        Returns:
            Dict containing:
                - text: Extracted text
                - metadata: Document metadata (pages, author, etc.)
                - structure: List of pages with text
        """
        logger.info(f"Processing PDF: {file_path}")

        try:
            doc = pymupdf.open(file_path)

            # Extract metadata
            metadata = {
                "pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "created": doc.metadata.get("creationDate", ""),
                "modified": doc.metadata.get("modDate", ""),
            }

            # Extract text from each page
            pages = []
            full_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text with layout preservation
                text = page.get_text("text", sort=True)

                if text.strip():
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                    })
                    full_text.append(text)

            # Get page count before closing
            total_pages = len(doc)
            doc.close()

            result = {
                "text": "\n\n".join(full_text),
                "metadata": metadata,
                "structure": {
                    "pages": pages,
                    "total_pages": total_pages,
                    "total_chars": sum(p["char_count"] for p in pages),
                    "total_words": sum(p["word_count"] for p in pages),
                },
            }

            logger.info(
                f"Successfully processed PDF: {total_pages} pages, "
                f"{result['structure']['total_words']} words"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise

    def extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF (optional, for future use)

        Args:
            file_path: Path to PDF file

        Returns:
            List of image metadata
        """
        doc = pymupdf.open(file_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "xref": xref,
                })

        doc.close()
        return images

    def is_supported(self, filename: str) -> bool:
        """Check if file extension is supported"""
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)
