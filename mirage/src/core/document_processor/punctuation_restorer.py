"""
Punctuation Restorer for ASR/YouTube Transcripts
Adds punctuation marks to unpunctuated text using BERT-based models
Supports Arabic and English (multilingual)
"""

import re
from typing import Dict, Any, List
from loguru import logger


class PunctuationRestorer:
    """Restore punctuation in unpunctuated text using pre-trained models"""

    def __init__(self, model_name: str = "oliverguhr/fullstop-punctuation-multilingual-sonar-base"):
        """
        Initialize punctuation restorer

        Args:
            model_name: HuggingFace model name for punctuation restoration
                       Default: multilingual BERT model supporting Arabic + English
        """
        self.model = None
        self.model_name = model_name
        self._model_loaded = False

        logger.info(f"Punctuation restorer initialized with model: {model_name}")

    def _load_model(self):
        """Lazy load the punctuation model"""
        if self._model_loaded:
            return

        try:
            # Import here to avoid loading on initialization
            from deepmultilingualpunctuation import PunctuationModel

            logger.info(f"Loading punctuation model: {self.model_name}")
            self.model = PunctuationModel(model=self.model_name)
            self._model_loaded = True
            logger.info("Punctuation model loaded successfully")

        except ImportError:
            logger.error(
                "deepmultilingualpunctuation not installed. "
                "Install with: pip install deepmultilingualpunctuation"
            )
            raise ImportError(
                "deepmultilingualpunctuation package required for punctuation restoration"
            )
        except Exception as e:
            logger.error(f"Error loading punctuation model: {e}")
            raise

    def restore_punctuation(self, text: str, chunk_size: int = 200) -> Dict[str, Any]:
        """
        Restore punctuation in unpunctuated text

        Args:
            text: Unpunctuated text (e.g., YouTube transcript)
            chunk_size: Number of words to process at once (default 200)
                       Smaller chunks = faster but may lose context

        Returns:
            Dict with:
                - punctuated_text: Text with restored punctuation
                - original_length: Character count before
                - punctuated_length: Character count after
                - punctuation_marks_added: Count of added marks
        """
        if not text or not text.strip():
            return {
                "punctuated_text": "",
                "original_length": 0,
                "punctuated_length": 0,
                "punctuation_marks_added": 0
            }

        # Load model if needed
        if not self._model_loaded:
            self._load_model()

        original_length = len(text)

        try:
            # Split into words
            words = text.split()

            # Process in chunks to avoid memory issues
            punctuated_chunks = []
            total_chunks = (len(words) + chunk_size - 1) // chunk_size

            logger.info(f"Restoring punctuation in {len(words)} words ({total_chunks} chunks of {chunk_size} words)")

            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)

                # Restore punctuation for this chunk
                punctuated_chunk = self.model.restore_punctuation(chunk_text)
                punctuated_chunks.append(punctuated_chunk)

            # Combine all chunks
            punctuated_text = " ".join(punctuated_chunks)

            # Clean up any double spaces
            punctuated_text = re.sub(r'\s+', ' ', punctuated_text)
            punctuated_text = punctuated_text.strip()

            punctuated_length = len(punctuated_text)

            # Count punctuation marks added
            punctuation_marks = ['.', ',', '!', '?', ':', ';']
            marks_added = sum(punctuated_text.count(mark) for mark in punctuation_marks)

            logger.info(
                f"Punctuation restored: {original_length} → {punctuated_length} chars, "
                f"{marks_added} punctuation marks added"
            )

            return {
                "punctuated_text": punctuated_text,
                "original_length": original_length,
                "punctuated_length": punctuated_length,
                "punctuation_marks_added": marks_added
            }

        except Exception as e:
            logger.error(f"Error restoring punctuation: {e}")
            # Return original text if restoration fails
            return {
                "punctuated_text": text,
                "original_length": original_length,
                "punctuated_length": original_length,
                "punctuation_marks_added": 0,
                "error": str(e)
            }

    def should_restore(self, text: str, threshold: float = 0.02) -> bool:
        """
        Determine if text needs punctuation restoration

        Args:
            text: Text to check
            threshold: Minimum expected punctuation ratio (default 2%)
                      If below this, text likely needs restoration

        Returns:
            True if punctuation restoration is recommended
        """
        if not text or len(text) < 100:
            return False

        # Count existing punctuation marks
        punctuation_marks = ['.', ',', '!', '?', ':', ';']
        mark_count = sum(text.count(mark) for mark in punctuation_marks)

        # Calculate punctuation ratio
        char_count = len(text)
        punctuation_ratio = mark_count / char_count if char_count > 0 else 0

        if punctuation_ratio < threshold:
            logger.info(
                f"Low punctuation ratio ({punctuation_ratio:.1%}) - restoration recommended"
            )
            return True

        return False
