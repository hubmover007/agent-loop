"""Tests for MultimodalProcessor."""

import base64
import json
import pytest
from pathlib import Path
from src.multimodal import MediaInput, MultimodalProcessor


class TestMediaInput:
    """Tests for MediaInput dataclass."""

    def test_text_to_openai_content(self):
        """Text input should map to text content block."""
        media = MediaInput(type="text", source="Hello world")
        block = media.to_openai_content()
        assert block["type"] == "text"
        assert block["text"] == "Hello world"

    def test_image_to_openai_content(self):
        """Image input should map to image_url content block."""
        media = MediaInput(type="image", source="https://example.com/img.png", mime_type="image/png")
        block = media.to_openai_content()
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/img.png"

    def test_file_to_openai_content(self):
        """File input should map to text content block (text is extracted)."""
        media = MediaInput(type="file", source="/tmp/report.pdf", description="Report PDF")
        block = media.to_openai_content()
        assert block["type"] == "text"
        assert "Report PDF" in block["text"]

    def test_audio_to_openai_content(self):
        """Audio input should map to text content block."""
        media = MediaInput(type="audio", source="/tmp/recording.ogg", description="Voice note")
        block = media.to_openai_content()
        assert block["type"] == "text"
        assert "Voice note" in block["text"]

    def test_default_values(self):
        """MediaInput defaults should work."""
        media = MediaInput(type="text", source="x")
        assert media.mime_type == ""
        assert media.description == ""
        assert media._extracted_text == ""


class TestMultimodalProcessor:
    """Tests for MultimodalProcessor."""

    @pytest.fixture
    def processor(self, tmp_path):
        return MultimodalProcessor(work_dir=str(tmp_path))

    # ── test_process_text ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_process_text(self, processor):
        """Text inputs should pass through unchanged."""
        inputs = [
            MediaInput(type="text", source="Hello"),
            MediaInput(type="text", source="World"),
        ]
        blocks = await processor.process(inputs)
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Hello"}
        assert blocks[1] == {"type": "text", "text": "World"}

    # ── test_process_image ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_process_image_from_file(self, processor, tmp_path):
        """Image from local file should be base64-encoded."""
        # Create a small PNG file
        img_path = tmp_path / "test.png"
        # 1x1 pixel transparent PNG
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        img_path.write_bytes(png_data)

        media = MediaInput(type="image", source=str(img_path), mime_type="image/png")
        block = await processor.process_image(media)

        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        assert "detail" in block["image_url"]

    @pytest.mark.asyncio
    async def test_process_image_url(self, processor):
        """Image URL should pass through directly."""
        media = MediaInput(
            type="image",
            source="https://example.com/photo.jpg",
            mime_type="image/jpeg",
        )
        block = await processor.process_image(media)
        assert block == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg", "detail": "auto"},
        }

    @pytest.mark.asyncio
    async def test_process_image_file_not_found(self, processor):
        """Non-existent image file should raise FileNotFoundError."""
        media = MediaInput(type="image", source="/nonexistent/image.png")
        with pytest.raises(FileNotFoundError):
            await processor.process_image(media)

    @pytest.mark.asyncio
    async def test_process_image_too_large(self, processor, tmp_path):
        """Image exceeding MAX_IMAGE_SIZE should raise ValueError."""
        processor.MAX_IMAGE_SIZE = 100  # Set very low for testing
        img_path = tmp_path / "large.png"
        img_path.write_bytes(b"x" * 200)  # 200 bytes > 100 limit
        media = MediaInput(type="image", source=str(img_path), mime_type="image/png")
        with pytest.raises(ValueError, match="exceeds max"):
            await processor.process_image(media)

    # ── test_process_file ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_process_file_txt(self, processor, tmp_path):
        """Text file should have its content extracted."""
        fpath = tmp_path / "notes.txt"
        fpath.write_text("Important notes here", encoding="utf-8")

        media = MediaInput(type="file", source=str(fpath))
        block = await processor.process_file(media)

        assert block["type"] == "text"
        assert "Important notes here" in block["text"]

    @pytest.mark.asyncio
    async def test_process_file_json(self, processor, tmp_path):
        """JSON file should be readable."""
        fpath = tmp_path / "config.json"
        fpath.write_text(json.dumps({"key": "value"}))

        media = MediaInput(type="file", source=str(fpath))
        block = await processor.process_file(media)

        assert block["type"] == "text"
        assert '"key": "value"' in block["text"]

    @pytest.mark.asyncio
    async def test_process_file_not_found(self, processor):
        """Non-existent file should raise FileNotFoundError."""
        media = MediaInput(type="file", source="/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            await processor.process_file(media)

    @pytest.mark.asyncio
    async def test_process_file_too_large(self, processor, tmp_path):
        """File exceeding MAX_FILE_SIZE should raise ValueError."""
        processor.MAX_FILE_SIZE = 50  # Set very low for testing
        fpath = tmp_path / "big.txt"
        fpath.write_bytes(b"x" * 100)
        media = MediaInput(type="file", source=str(fpath))
        with pytest.raises(ValueError, match="exceeds max"):
            await processor.process_file(media)

    # ── test_unsupported_type_rejected ─────────────────────────

    @pytest.mark.asyncio
    async def test_unsupported_type_rejected(self, processor):
        """Unsupported media type should log a warning and produce an error block."""
        media = MediaInput(type="video", source="movie.mp4")
        blocks = await processor.process([media])
        # Should still produce output, just an error text block
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "video" in blocks[0]["text"]

    # ── Audio processing ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_process_audio_file(self, processor, tmp_path):
        """Audio file should be handled (placeholder)."""
        audio_path = tmp_path / "recording.ogg"
        audio_path.write_bytes(b"fake ogg data")
        media = MediaInput(type="audio", source=str(audio_path), description="Meeting recording")
        block = await processor.process_audio(media)
        assert block["type"] == "text"
        assert "Meeting recording" in block["text"]

    # ── Mixed input batch ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_process_mixed_inputs(self, processor, tmp_path):
        """Mixed text + image + file batch should produce correct blocks."""
        # Create a text file
        fpath = tmp_path / "doc.txt"
        fpath.write_text("file content")

        # Create a small image
        img_path = tmp_path / "icon.png"
        img_path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))

        inputs = [
            MediaInput(type="text", source="Context:"),
            MediaInput(type="image", source=str(img_path), mime_type="image/png"),
            MediaInput(type="file", source=str(fpath), description="Document"),
        ]
        blocks = await processor.process(inputs)
        assert len(blocks) == 3

        # Verify types
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image_url"
        assert blocks[2]["type"] == "text"
        assert "file content" in blocks[2]["text"]

    # ── Error resilience ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_error_creates_text_block(self, processor):
        """If processing fails, an error text block is emitted."""
        media = MediaInput(type="file", source="/nonexistent/file.txt")
        blocks = await processor.process([media])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Error" in blocks[0]["text"]
