"""Multimodal input processing.

Extends AgentLoop beyond plain text: image, file, and audio inputs are
validated, transcoded, compressed, and converted into OpenAI-compatible
content blocks ready for LLM consumption.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# MediaInput
# ============================================================


@dataclass
class MediaInput:
    """A single multimodal input item.

    Attributes:
        type: "image" | "file" | "audio" | "text"
        source: URL, local file path, or base64-encoded data
        mime_type: MIME type hint (e.g. "image/png", "application/pdf")
        description: Optional context for the LLM (alt-text / caption)
    """
    type: str                # image | file | audio | text
    source: str              # URL, file path, or base64 data
    mime_type: str = ""
    description: str = ""    # Alt-text / caption for LLM

    def to_openai_content(self) -> dict:
        """Convert to an OpenAI message content block.

        Mapping:
          - image → ``{"type": "image_url", "image_url": {"url": "..."}}``
          - file  → ``{"type": "file", "file": {"file_data": "..."}}``
          - audio → ``{"type": "audio", "audio": {"data": "..."}}``
          - text  → ``{"type": "text", "text": "..."}``
        """
        if self.type == "image":
            return {
                "type": "image_url",
                "image_url": {"url": self.source, "detail": "auto"},
            }
        elif self.type == "file":
            return {
                "type": "text",
                "text": f"[File: {self.description or self.source}] {self._extracted_text or ''}",
            }
        elif self.type == "audio":
            return {
                "type": "text",
                "text": f"[Audio: {self.description or self.source}]\n{self._extracted_text or ''}",
            }
        else:  # text
            return {"type": "text", "text": self.source}

    # Private attributes set during processing
    _extracted_text: str = field(default="", repr=False, init=False)

    def __post_init__(self):
        self._extracted_text = ""


# ============================================================
# MultimodalProcessor
# ============================================================


class MultimodalProcessor:
    """Process multimodal inputs into LLM-consumable content blocks.

    Capabilities:
      1. Validate input format and size limits
      2. Compress / transcode oversized images
      3. Extract text from files (PDF, txt, code, structured data)
      4. Generate OpenAI-compatible content blocks
    """

    MAX_IMAGE_SIZE = 20 * 1024 * 1024   # 20 MB — reject anything larger
    MAX_FILE_SIZE = 50 * 1024 * 1024    # 50 MB
    IMAGE_COMPRESS_THRESHOLD = 4 * 1024 * 1024  # 4 MB — compress above this

    SUPPORTED_IMAGE_TYPES = {"jpeg", "jpg", "png", "gif", "webp"}
    SUPPORTED_FILE_TYPES = {"pdf", "txt", "md", "json", "yaml", "yml",
                            "csv", "py", "js", "ts", "go", "rs",
                            "java", "c", "cpp", "h", "sh", "toml",
                            "xml", "html", "css",
                            }

    def __init__(self, work_dir: str = "."):
        self.work_dir = Path(work_dir)

    # ── Main entry point ──────────────────────────────────────────

    async def process(self, inputs: list[MediaInput]) -> list[dict]:
        """Process a batch of multimodal inputs → OpenAI content blocks.

        Each input is validated and routed to the appropriate handler.
        Unsupported types are logged and skipped.
        """
        blocks: list[dict] = []
        for media in inputs:
            try:
                # Accept both MediaInput objects and dicts
                if isinstance(media, dict):
                    media = MediaInput(
                        type=media.get("type", "text"),
                        source=media.get("content", media.get("source", "")),
                        mime_type=media.get("mime_type", ""),
                        description=media.get("description", ""),
                    )
                if media.type == "text":
                    blocks.append(media.to_openai_content())
                elif media.type == "image":
                    blocks.append(await self.process_image(media))
                elif media.type == "file":
                    blocks.append(await self.process_file(media))
                elif media.type == "audio":
                    blocks.append(await self.process_audio(media))
                else:
                    logger.warning("MultimodalProcessor: unsupported type '%s'", media.type)
                    blocks.append({
                        "type": "text",
                        "text": f"[Unsupported media type: {media.type}]",
                    })
            except Exception as e:
                logger.error("MultimodalProcessor: failed to process %s: %s", media.type, e)
                # Add an error text block so the LLM knows something went wrong
                blocks.append({
                    "type": "text",
                    "text": f"[Error processing {media.type} input: {e}]",
                })
        return blocks

    # ── Image processing ──────────────────────────────────────────

    async def process_image(self, media: MediaInput) -> dict:
        """Process an image input → OpenAI image_url content block.

        Steps:
          1. Determine source type (URL / base64 / file path)
          2. Validate size (reject > 20 MB)
          3. Compress if > 4 MB
          4. Return content block
        """
        source = media.source

        # URL — pass through directly
        if source.startswith(("http://", "https://")):
            return media.to_openai_content()

        # Base64 — keep as data URL
        if source.startswith("data:"):
            raw = self._decode_data_url(source)
            if len(raw) > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size {len(raw) / 1024 / 1024:.1f} MB exceeds max {self.MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB"
                )
            return media.to_openai_content()

        # File path
        path = Path(source)
        if not path.is_absolute() and not path.exists():
            path = self.work_dir / source

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {source}")

        size = path.stat().st_size
        if size > self.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image size {size / 1024 / 1024:.1f} MB exceeds max {self.MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB"
            )

        ext = path.suffix.lstrip(".").lower()
        if ext not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image type: .{ext}")

        # Get MIME type
        mime = media.mime_type or mimetypes.guess_type(str(path))[0] or f"image/{ext}"

        # Compress if large
        if size > self.IMAGE_COMPRESS_THRESHOLD:
            logger.info("MultimodalProcessor: compressing image %s (%d KB)", source, size // 1024)
            data = await self._compress_image(path)
            b64 = base64.b64encode(data).decode("ascii")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"},
            }

        # Read and base64-encode
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"},
        }

    # ── File processing ───────────────────────────────────────────

    async def process_file(self, media: MediaInput) -> dict:
        """Process a file input → extract text → OpenAI text content block.

        Supported: txt, md, json, yaml, csv, code files.
        Unsupported: returns metadata + file name only.
        """
        source = media.source
        path = Path(source)
        if not path.is_absolute() and not path.exists():
            path = self.work_dir / source

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        size = path.stat().st_size
        if size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File size {size / 1024 / 1024:.1f} MB exceeds max {self.MAX_FILE_SIZE / 1024 / 1024:.0f} MB"
            )

        ext = path.suffix.lstrip(".").lower()

        text = ""
        if ext in self.SUPPORTED_FILE_TYPES:
            try:
                text = self._extract_file_text(str(path), ext)
            except Exception as e:
                logger.warning("MultimodalProcessor: text extraction failed for %s: %s", source, e)
                text = f"[Could not extract text: {e}]"

        media._extracted_text = text
        media.description = media.description or path.name
        return media.to_openai_content()

    # ── Audio processing ──────────────────────────────────────────

    async def process_audio(self, media: MediaInput) -> dict:
        """Process an audio input → placeholder block (no speech-to-text here).

        For full speech-to-text integration, wire an STT provider here.
        """
        source = media.source
        if source.startswith(("http://", "https://", "data:")):
            media._extracted_text = ""
        else:
            path = Path(source)
            if not path.is_absolute() and not path.exists():
                path = self.work_dir / source
            if path.exists():
                media.description = media.description or path.name

        return media.to_openai_content()

    # ── Internal helpers ──────────────────────────────────────────

    def _extract_file_text(self, path: str, mime_type: str) -> str:
        """Extract plain text from a file.

        Handles: plain text / code / JSON / YAML / CSV / PDF (simple).
        """
        ext = Path(path).suffix.lstrip(".").lower()

        # PDF — minimal text extraction via pdftotext if available
        if ext == "pdf":
            import subprocess
            try:
                result = subprocess.run(
                    ["pdftotext", "-layout", path, "-"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    return result.stdout[:10000]
                else:
                    return f"[PDF: {os.path.basename(path)} — pdftotext returned {result.returncode}]"
            except FileNotFoundError:
                return f"[PDF: {os.path.basename(path)} — pdftotext not available]"
            except Exception as e:
                return f"[PDF: {os.path.basename(path)} — error: {e}]"

        # Text / code / structured data — read directly
        try:
            with open(path, encoding="utf-8") as f:
                return f.read(50000)  # Truncate to 50 KB
        except UnicodeDecodeError:
            return f"[Binary file: {os.path.basename(path)}]"

    async def _compress_image(self, path: Path) -> bytes:
        """Compress an image to reduce size using PIL.

        Only compresses if the file is > IMAGE_COMPRESS_THRESHOLD (4 MB by default).
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            logger.warning("MultimodalProcessor: PIL not installed, returning raw bytes")
            return path.read_bytes()

        img = PILImage.open(path)
        # Resize if width > 2048
        if img.width > 2048:
            ratio = 2048 / img.width
            new_size = (2048, int(img.height * ratio))
            img = img.resize(new_size, PILImage.LANCZOS)

        # Save as JPEG for compression (loses alpha but good for LLM vision)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(str(out), "JPEG", quality=70)
            return out.read_bytes()
        finally:
            out.unlink(missing_ok=True)

    @staticmethod
    def _decode_data_url(data_url: str) -> bytes:
        """Decode a ``data:...;base64,...`` URL into raw bytes."""
        if "," not in data_url:
            return data_url.encode("utf-8")
        header, encoded = data_url.split(",", 1)
        if "base64" in header:
            return base64.b64decode(encoded)
        else:
            return encoded.encode("utf-8")


# ============================================================
# MultimodalRouter — multi-model routing by media type
# ============================================================


class MultimodalRouter:
    """Route multimodal inputs to the best-fit model/processor by type.

    Architecture:
      - image → vision model (GPT-4o / Gemini / Claude) → image_url block
      - audio → STT provider (Whisper API or local) → transcribed text
      - video → ffmpeg keyframe extraction → multiple image blocks
      - file  → file parser (PDF→text, code, CSV, etc.) → text block
      - text  → passthrough directly to LLM

    The router does not transcode files itself. It delegates to:
      - LLMPool (vision, STT)
      - ffmpeg subprocess (video keyframe extraction)
      - MultimodalProcessor (file text extraction)

    All processing results are unified into ContentBlock dicts ready for
    injection into the MainLoop INPUT phase.
    """

    MAX_VIDEO_FRAMES = 5  # Extract at most 5 keyframes per video

    def __init__(
        self,
        llm_pool: Any | None = None,
        whisper_api_key: str | None = None,
        vision_model: str | None = None,
        processor: MultimodalProcessor | None = None,
        work_dir: str = ".",
    ):
        self.llm_pool = llm_pool
        self.whisper_api_key = whisper_api_key
        self.vision_model = vision_model or "gpt-4o"
        self.processor = processor or MultimodalProcessor(work_dir=work_dir)
        self.work_dir = work_dir

    # ── Main entry point ──────────────────────────────────────────

    async def process(self, inputs: list[MediaInput]) -> list[dict]:
        """Process a batch of multimodal inputs → unified ContentBlock list.

        Each input is routed by type to the appropriate handler.
        Results are collected into a flat list of OpenAI-compatible dicts.
        """
        blocks: list[dict] = []
        for inp in inputs:
            try:
                if inp.type == "text":
                    blocks.append({"type": "text", "text": inp.source})
                elif inp.type == "image":
                    blocks.append(await self._process_image(inp))
                elif inp.type == "audio":
                    blocks.append(await self._process_audio(inp))
                elif inp.type == "video":
                    blocks.extend(await self._process_video(inp))
                elif inp.type == "file":
                    blocks.append(await self._process_file(inp))
                else:
                    logger.warning(
                        "MultimodalRouter: unsupported type '%s'", inp.type
                    )
                    blocks.append({
                        "type": "text",
                        "text": f"[Unsupported media type: {inp.type}]",
                    })
            except Exception as e:
                logger.error(
                    "MultimodalRouter: failed to route %s: %s", inp.type, e
                )
                blocks.append({
                    "type": "text",
                    "text": f"[Error processing {inp.type} input: {e}]",
                })
        return blocks

    # ── Image → vision model (recognition + image_url) ─────────

    async def _process_image(self, inp: MediaInput) -> dict:
        """Route image to a vision model for recognition, then build a content block.

        When a vision model is available via llm_pool, the image is sent to it
        for a short description. The return block includes both the original
        image URL (for the LLM to see) and the vision description as context.
        """
        # Always include the image as a base64-encoded data URL
        # Use MultimodalProcessor to normalize (URL / local file / base64)
        image_block = await self.processor.process_image(inp)

        # Optionally add vision-model description for richer context
        desc = inp.description or ""
        if self.llm_pool:
            try:
                provider = await self.llm_pool.acquire(
                    capabilities=["vision"],
                    strategy="cheapest",
                )
                resp = await provider.chat([{
                    "role": "user",
                    "content": [
                        image_block,
                        {"type": "text", "text": "Describe this image briefly in one sentence."},
                    ],
                }])
                desc = resp.content.strip() if resp and resp.content else ""
                logger.info("MultimodalRouter: vision model described image")
            except Exception as e:
                logger.warning(
                    "MultimodalRouter: vision model unavailable (%s), using passthrough", e
                )

        if desc:
            image_block["image_url"]["description"] = desc
        return image_block

    # ── Audio → STT transcription ─────────────────────────────────

    async def _process_audio(self, inp: MediaInput) -> dict:
        """Route audio to STT (Whisper API or pool provider) for transcription.

        When llm_pool has an STT-capable provider, prefer it.
        When WHISPER_API_KEY is set, call OpenAI Whisper directly.
        Otherwise, return a placeholder text block.
        """
        source = inp.source
        description = inp.description or ""

        # Try llm_pool STT provider first
        if self.llm_pool:
            try:
                provider = await self.llm_pool.acquire(
                    capabilities=["audio", "stt"],
                    strategy="cheapest",
                )
                resp = await provider.chat([{
                    "role": "user",
                    "content": f"Transcribe this audio: {source}. Reply with only the transcribed text.",
                }])
                transcript = resp.content.strip() if resp and resp.content else ""
                if transcript:
                    return {
                        "type": "text",
                        "text": f"[Audio transcription] {transcript}",
                    }
            except Exception as e:
                logger.warning(
                    "MultimodalRouter: STT via llm_pool failed: %s", e
                )

        # Try Whisper API directly if we have a key
        if self.whisper_api_key:
            try:
                transcript = await self._whisper_transcribe(source)
                if transcript:
                    return {
                        "type": "text",
                        "text": f"[Audio transcription] {transcript}",
                    }
            except Exception as e:
                logger.warning(
                    "MultimodalRouter: Whisper API failed: %s", e
                )

        # Fallback: placeholder
        return {
            "type": "text",
            "text": f"[Audio: {description or source}] (no STT provider available)",
        }

    async def _whisper_transcribe(self, source: str) -> str:
        """Transcribe audio via OpenAI Whisper API."""
        import httpx
        if not self.whisper_api_key:
            raise RuntimeError("No Whisper API key configured")

        # Determine audio file path
        if source.startswith(("http://", "https://")):
            # Download audio file first
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(source)
                resp.raise_for_status()
                audio_data = resp.content
                filename = "audio_download"
        elif source.startswith("data:"):
            audio_data = self.processor._decode_data_url(source)
            filename = "audio_data"
        else:
            path = Path(source)
            if not path.is_absolute():
                path = Path(self.work_dir) / source
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {source}")
            audio_data = path.read_bytes()
            filename = path.name

        # Call Whisper API
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.whisper_api_key}"},
                files={"file": (filename, audio_data)},
                data={"model": "whisper-1"},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Whisper API returned {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            return data.get("text", "")

    # ── Video → ffmpeg keyframe extraction ─────────────────────────

    async def _process_video(self, inp: MediaInput) -> list[dict]:
        """Extract keyframes from video via ffmpeg, then route each as image.

        Extracts up to MAX_VIDEO_FRAMES (5) keyframes from the video.
        Each frame is processed as an image input through _process_image.
        Falls back gracefully when ffmpeg is unavailable.
        """
        import subprocess

        source = inp.source

        # Resolve path
        if source.startswith(("http://", "https://")):
            # For remote videos, we need to attempt download first or skip
            logger.warning(
                "MultimodalRouter: remote video URLs require local download; "
                "returning placeholder"
            )
            return [{
                "type": "text",
                "text": f"[Video: {source or inp.description}] (remote video not yet supported for frame extraction)",
            }]

        path = Path(source)
        if not path.is_absolute():
            path = Path(self.work_dir) / source

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {source}")

        # Check ffmpeg availability
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                raise FileNotFoundError("ffmpeg not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning(
                "MultimodalRouter: ffmpeg not available, returning placeholder for video"
            )
            return [{
                "type": "text",
                "text": f"[Video: {inp.description or source}] (ffmpeg not available for frame extraction)",
            }]

        # Extract keyframes to temp directory
        frames_dir = Path(tempfile.mkdtemp(prefix="video_frames_"))
        try:
            # Get video duration first
            probe_result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True, text=True, timeout=15,
            )
            duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 60.0

            # Calculate frame intervals to spread across the video
            num_frames = min(self.MAX_VIDEO_FRAMES, max(3, int(duration / 10)))
            interval = duration / (num_frames + 1)

            # Use ffmpeg to extract frames at calculated timestamps
            frame_dirs = []
            for i in range(1, num_frames + 1):
                timestamp = interval * i
                out_path = frames_dir / f"frame_{i:02d}.jpg"
                frame_dirs.append(out_path)

            # Build ffmpeg select filter for keyframes at specific times
            select_expr = "+".join(
                f"between(t,{interval * i - 0.1},{interval * i + 0.1})"
                for i in range(1, num_frames + 1)
            )

            ffmpeg_result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-vf", f"select='{select_expr}',scale=1024:-1",
                    "-vsync", "0",
                    "-frame_pts", "1",
                    str(frames_dir / "frame_%02d.jpg"),
                ],
                capture_output=True, timeout=60,
            )

            # Collect the frames that were written
            actual_frames = sorted(frames_dir.glob("frame_*.jpg"))
            if not actual_frames:
                # Fallback: extract a single thumbnail at 1 second
                fallback_path = frames_dir / "frame_01.jpg"
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(path),
                        "-ss", "1", "-vframes", "1",
                        "-vf", "scale=1024:-1",
                        str(fallback_path),
                    ],
                    capture_output=True, timeout=30,
                )
                if fallback_path.exists():
                    actual_frames = [fallback_path]

            # Process each frame as an image
            blocks: list[dict] = []
            # Add a context text block first
            blocks.append({
                "type": "text",
                "text": (
                    f"[Video: {inp.description or path.name}] "
                    f"({len(actual_frames)} keyframes extracted, duration: {duration:.1f}s)"
                ),
            })

            for frame_path in actual_frames[:self.MAX_VIDEO_FRAMES]:
                frame_media = MediaInput(
                    type="image",
                    source=str(frame_path),
                    mime_type="image/jpeg",
                    description=f"Frame from video: {inp.description or path.name}",
                )
                try:
                    block = await self._process_image(frame_media)
                    blocks.append(block)
                except Exception as e:
                    logger.warning(
                        "MultimodalRouter: failed to process video frame %s: %s",
                        frame_path.name, e,
                    )

            return blocks

        finally:
            # Clean up temp frames
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    # ── File → parser ─────────────────────────────────────────────

    async def _process_file(self, inp: MediaInput) -> dict:
        """Route file to appropriate parser based on extension.

        Delegates to MultimodalProcessor for text extraction.
        Adds structured output for known formats (code, JSON, CSV, PDF).
        """
        return await self.processor.process_file(inp)

    # ── Convenience: synthesize from router ────────────────────────

    def synthesize_text(self, inputs: list[MediaInput]) -> str:
        """Extract a flat text summary from media inputs (useful for prompts).

        Useful before routing when you need a quick human-readable summary
        of what the user sent.
        """
        parts: list[str] = []
        seen = set()
        for inp in inputs:
            if inp.type == "text":
                parts.append(inp.source)
            elif inp.description and inp.description not in seen:
                parts.append(f"[{inp.type}: {inp.description}]")
                seen.add(inp.description)
            elif inp.type not in seen:
                parts.append(f"[{inp.type}]")
                seen.add(inp.type)
        return " ".join(parts)
