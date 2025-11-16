"""
HTML Processor
Extracts main content from HTML files, removes ads/navigation
Converts to clean markdown format
"""

from bs4 import BeautifulSoup
import html2text
from typing import Dict, Any
from loguru import logger


class HTMLProcessor:
    """Process HTML files and extract clean content"""

    def __init__(self):
        self.supported_extensions = [".html", ".htm"]

        # Configure html2text
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # Don't wrap lines

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Extract clean content from HTML file

        Args:
            file_path: Path to HTML file

        Returns:
            Dict containing:
                - text: Extracted text in markdown format
                - metadata: Document metadata
                - structure: Content structure info
        """
        logger.info(f"Processing HTML: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'lxml')

            # Extract metadata
            metadata = self._extract_metadata(soup)

            # Remove unwanted elements
            cleaned_soup = self._clean_html(soup)

            # Extract main content
            main_content = self._extract_main_content(cleaned_soup)

            # Convert to markdown
            markdown_text = self.html_converter.handle(str(main_content))

            # Get structure info
            structure = self._analyze_structure(cleaned_soup)

            result = {
                "text": markdown_text.strip(),
                "metadata": metadata,
                "structure": structure,
            }

            logger.info(
                f"Successfully processed HTML: {len(markdown_text)} chars, "
                f"{len(markdown_text.split())} words"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing HTML {file_path}: {e}")
            raise

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from HTML head"""
        metadata = {}

        # Title
        title_tag = soup.find('title')
        metadata['title'] = title_tag.get_text().strip() if title_tag else ""

        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')

            if name and content:
                # Common meta tags
                if 'description' in name.lower():
                    metadata['description'] = content
                elif 'author' in name.lower():
                    metadata['author'] = content
                elif 'keywords' in name.lower():
                    metadata['keywords'] = content
                elif 'date' in name.lower() or 'published' in name.lower():
                    metadata['date'] = content

        return metadata

    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove unwanted elements (nav, ads, scripts, etc.)"""
        # Elements to remove
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer',
            'aside', 'iframe', 'noscript'
        ]

        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove elements by class/id (common ad/navigation patterns)
        unwanted_patterns = [
            'nav', 'navigation', 'menu', 'sidebar', 'ad',
            'advertisement', 'footer', 'header', 'cookie',
            'popup', 'modal', 'banner'
        ]

        for element in soup.find_all(class_=True):
            classes = ' '.join(element.get('class', [])).lower()
            if any(pattern in classes for pattern in unwanted_patterns):
                element.decompose()

        for element in soup.find_all(id=True):
            elem_id = element.get('id', '').lower()
            if any(pattern in elem_id for pattern in unwanted_patterns):
                element.decompose()

        return soup

    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content area"""
        # Try to find main content container
        main_selectors = [
            soup.find('main'),
            soup.find('article'),
            soup.find(class_='content'),
            soup.find(class_='main'),
            soup.find(id='content'),
            soup.find(id='main'),
            soup.find('body'),  # Fallback
        ]

        for selector in main_selectors:
            if selector:
                return selector

        # If nothing found, return the whole soup
        return soup

    def _analyze_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze document structure"""
        return {
            "headings": {
                "h1": len(soup.find_all('h1')),
                "h2": len(soup.find_all('h2')),
                "h3": len(soup.find_all('h3')),
                "h4": len(soup.find_all('h4')),
                "h5": len(soup.find_all('h5')),
                "h6": len(soup.find_all('h6')),
            },
            "paragraphs": len(soup.find_all('p')),
            "lists": len(soup.find_all(['ul', 'ol'])),
            "tables": len(soup.find_all('table')),
            "links": len(soup.find_all('a')),
            "images": len(soup.find_all('img')),
        }

    def is_supported(self, filename: str) -> bool:
        """Check if file extension is supported"""
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)
