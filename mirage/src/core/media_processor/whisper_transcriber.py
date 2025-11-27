"""
Whisper Transcriber for YouTube Videos and Audio Files
Uses Whisper large-v3 for maximum quality Arabic + English transcription
CPU-optimized with faster-whisper (5x faster than official implementation)
"""

import os
import tempfile
from typing import Dict, Any, Optional
from loguru import logger


class WhisperTranscriber:
    """Transcribe audio/video using Whisper large-v3 model"""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize Whisper transcriber

        Args:
            model_size: Whisper model size (large-v3 = best for Arabic)
            device: "cpu" or "cuda" (we use CPU for compatibility)
            compute_type: "int8" for CPU (optimized), "float16" for GPU
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._model_loaded = False

        logger.info(f"Whisper transcriber initialized: model={model_size}, device={device}, compute_type={compute_type}")

    def _load_model(self):
        """Lazy load Whisper model (downloads ~3GB on first use)"""
        if self._model_loaded:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper {self.model_size} model (this may take a few minutes on first use)...")

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,  # Use default cache directory
                local_files_only=False
            )

            self._model_loaded = True
            logger.info(f"Whisper {self.model_size} model loaded successfully")

        except ImportError:
            logger.error(
                "faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )
            raise ImportError("faster-whisper package required for transcription")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def download_audio(self, youtube_url: str) -> str:
        """
        Download audio from YouTube video

        Args:
            youtube_url: YouTube video URL

        Returns:
            Path to downloaded audio file (WAV format)
        """
        try:
            import yt_dlp

            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_path = temp_audio.name
            temp_audio.close()

            # Download audio using yt-dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': temp_audio_path.replace('.wav', ''),
                'quiet': True,
                'no_warnings': True,
            }

            logger.info(f"Downloading audio from YouTube: {youtube_url}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            logger.info(f"Audio downloaded successfully: {temp_audio_path}")
            return temp_audio_path

        except ImportError:
            logger.error("yt-dlp not installed. Install with: pip install yt-dlp")
            raise ImportError("yt-dlp required for YouTube audio download")
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 2.0,
        vad_filter: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper

        Args:
            audio_path: Path to audio file
            language: Language code ("ar" or "en", auto-detect if None)
            task: "transcribe" or "translate" (we use transcribe)
            beam_size: Beam search size (higher = better quality, slower)
            best_of: Number of candidates (higher = better quality)
            patience: Patience for beam search
            vad_filter: Use voice activity detection (removes silence)

        Returns:
            Dict with transcript, language, duration, segments
        """
        # Load model if needed
        if not self._model_loaded:
            self._load_model()

        try:
            logger.info(f"Transcribing audio: {audio_path} (language={language or 'auto-detect'})")

            # Transcribe using Whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                vad_filter=vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence to split segments
                    speech_pad_ms=200  # Padding around speech
                )
            )

            # Collect all segments
            transcript_segments = []
            full_text_parts = []

            for segment in segments:
                transcript_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
                full_text_parts.append(segment.text.strip())

            # Combine all text
            full_text = " ".join(full_text_parts)

            logger.info(
                f"Transcription complete: {len(full_text)} chars, "
                f"{len(transcript_segments)} segments, "
                f"language={info.language}, "
                f"duration={info.duration:.1f}s"
            )

            return {
                "text": full_text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": transcript_segments,
                "segment_count": len(transcript_segments),
                "char_count": len(full_text),
                "has_punctuation": True  # Whisper includes punctuation
            }

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def transcribe_youtube(
        self,
        youtube_url: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe YouTube video using Whisper

        Args:
            youtube_url: YouTube video URL
            language: Language code ("ar" or "en", auto-detect if None)

        Returns:
            Dict with transcript and metadata
        """
        audio_path = None

        try:
            # Download audio from YouTube
            audio_path = self.download_audio(youtube_url)

            # Transcribe the audio
            result = self.transcribe_audio(
                audio_path,
                language=language,
                beam_size=5,  # High quality
                best_of=5,    # High quality
                patience=2.0  # High quality
            )

            result["source"] = "whisper"
            result["model"] = self.model_size

            return result

        finally:
            # Clean up temporary audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.debug(f"Cleaned up temporary audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {audio_path}: {e}")
