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
