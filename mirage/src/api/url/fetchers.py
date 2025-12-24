"""
URL Service - Content Fetchers
YouTube and webpage content extraction
"""

import re
from typing import Dict, Any, Optional
from loguru import logger
import trafilatura
import requests
from fastapi import HTTPException
from youtube_transcript_api import YouTubeTranscriptApi

from ...core.document_processor.content_cleaner import ContentCleaner
from ...core.document_processor.punctuation_restorer import PunctuationRestorer
from ...core.media_processor import WhisperTranscriber

# Initialize processors
content_cleaner = ContentCleaner()
punctuation_restorer = PunctuationRestorer()
whisper_transcriber = WhisperTranscriber(model_size="large-v3")


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats
    Supports: youtube.com/watch, youtu.be, youtube.com/embed
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*?v=([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def fetch_webpage_content(url: str) -> Dict[str, Any]:
    """Fetch and extract main content from a web page using trafilatura"""
    logger.info(f"Fetching web page: {url}")

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError("Failed to download the web page")

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        if not text:
            raise ValueError("Failed to extract content from the web page")

        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else "Untitled"
        description = metadata.description if metadata and metadata.description else ""

        return {
            "text": text,
            "title": title,
            "description": description,
            "url": url,
            "metadata": {
                "author": metadata.author if metadata else None,
                "date": metadata.date if metadata else None,
                "sitename": metadata.sitename if metadata else None,
            }
        }

    except Exception as e:
        logger.error(f"Error fetching web page {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch web page: {str(e)}")


def fetch_youtube_transcript(video_id: str) -> Dict[str, Any]:
    """
    Fetch transcript from YouTube video using DUAL-PATH approach
    Priority: YouTube API + BERT (speed) → fallback to Whisper large-v3 (quality)
    """
    logger.info(f"Fetching transcript for video: {video_id}")

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    full_text = None
    transcript_method = None
    detected_language = "unknown"

    # PATH 1: Try YouTube API + BERT first (SPEED PRIORITY)
    try:
        logger.info("PATH 1: Attempting YouTube API + BERT punctuation...")
        api = YouTubeTranscriptApi()

        transcript = None
        last_error = None

        # Try English first, then Arabic, then any available
        try:
            transcript = api.fetch(video_id, languages=['en'])
            detected_language = transcript.language_code
        except Exception as e:
            last_error = e
            try:
                transcript = api.fetch(video_id, languages=['ar'])
                detected_language = transcript.language_code
            except Exception as e:
                last_error = e
                try:
                    transcript_list = api.list(video_id)
                    available = list(transcript_list)
                    if available:
                        transcript = available[0].fetch()
                        detected_language = transcript.language_code
                except Exception as e:
                    last_error = e

        if not transcript:
            raise Exception(f"No transcript found ({type(last_error).__name__})")

        raw_text = " ".join([snippet.text for snippet in transcript.snippets])

        # Clean and restore punctuation
        cleaning_result = content_cleaner.clean_youtube_transcript(raw_text)
        cleaned_text = cleaning_result["cleaned_text"]

        if punctuation_restorer.should_restore(cleaned_text):
            punctuation_result = punctuation_restorer.restore_punctuation(cleaned_text)
            full_text = punctuation_result["punctuated_text"]
            transcript_method = "youtube_api_bert"
        else:
            full_text = cleaned_text
            transcript_method = "youtube_api"

        logger.info(f"YouTube API + BERT successful: {len(full_text)} chars")

    except Exception as youtube_error:
        logger.warning(f"YouTube API failed: {youtube_error}")
        logger.info("PATH 2: Falling back to Whisper large-v3...")

        try:
            whisper_result = whisper_transcriber.transcribe_youtube(youtube_url, language=None)
            full_text = whisper_result["text"]
            detected_language = whisper_result["language"]
            transcript_method = "whisper_large-v3"
            logger.info(f"Whisper fallback successful: {len(full_text)} chars")

        except Exception as whisper_error:
            logger.error(f"Both YouTube API and Whisper failed: {whisper_error}")
            raise HTTPException(
                status_code=400,
                detail="Failed to fetch transcript. No captions and Whisper failed."
            )

    # Get video metadata
    title, author = _fetch_video_metadata(video_id)

    return {
        "text": full_text,
        "title": title,
        "description": f"Transcript from YouTube video by {author}",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "metadata": {
            "video_id": video_id,
            "author": author,
            "language": detected_language,
            "transcript_length": len(full_text),
        }
    }


def fetch_youtube_metadata_only(video_id: str) -> Dict[str, Any]:
    """Fetch ONLY metadata from YouTube video (INSTANT - < 1 second)"""
    logger.info(f"Fetching metadata for video: {video_id}")

    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=5)

        if response.status_code == 200:
            video_info = response.json()
            return {
                "title": video_info.get('title', 'YouTube Video'),
                "description": f"YouTube video by {video_info.get('author_name', 'Unknown')}",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "author": video_info.get('author_name', 'Unknown'),
                "thumbnail": video_info.get('thumbnail_url', ''),
                "video_id": video_id,
                "content_type": "youtube"
            }
        else:
            raise Exception(f"oEmbed API failed with status {response.status_code}")

    except Exception as e:
        logger.error(f"Failed to fetch video metadata: {e}")
        return {
            "title": f"YouTube Video {video_id}",
            "description": "YouTube video (metadata unavailable)",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "author": "Unknown",
            "thumbnail": "",
            "video_id": video_id,
            "content_type": "youtube"
        }


def _fetch_video_metadata(video_id: str) -> tuple:
    """Helper to fetch video title and author"""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=10)
        if response.status_code == 200:
            video_info = response.json()
            return video_info.get('title', 'YouTube Video'), video_info.get('author_name', 'Unknown')
    except Exception as e:
        logger.warning(f"Failed to fetch video metadata: {e}")

    return f"YouTube Video {video_id}", "Unknown"
