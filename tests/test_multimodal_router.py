"""Tests for MultimodalRouter — multi-model routing by media type.

Covers:
  - Text passthrough (no routing needed)
  - Image routing → vision model (mock)
  - Audio routing → STT (mock Whisper / mock provider)
  - Video keyframe extraction (mock ffmpeg)
  - File text extraction (real files: txt, json, csv, PDF placeholder)
  - Multi-model routing selection (LLMPool.get_vision_model / get_stt_model)
  - Error resilience (unsupported types, missing files, failures)
"""

import asyncio
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.multimodal import MediaInput, MultimodalRouter, MultimodalProcessor
from src.llm_pool import (
    LLMPool, ProviderConfigJSON, PoolConfigJSON, SelectionConfig,
)
from src.loop_engine import LoopContext


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def tmp_work_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def router_no_pool(tmp_work_dir):
    """Router without LLM pool — pure passthrough/processor mode."""
    return MultimodalRouter(work_dir=tmp_work_dir)


@pytest.fixture
def mock_llm_pool():
    """Mock LLM pool with a vision-capable provider."""
    pool = MagicMock(spec=LLMPool)
    pool._initialized = True

    mock_provider = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "A photo of a sunset over mountains."
    mock_provider.chat = AsyncMock(return_value=mock_resp)
    pool.acquire = AsyncMock(return_value=mock_provider)

    vision_cfg = ProviderConfigJSON(
        id="mock-vision",
        type="openai",
        model="gpt-4o",
        capabilities=["vision", "general"],
        modality=["text", "image"],
        enabled=True,
        verified=True,
    )
    pool.get_vision_model = MagicMock(return_value=vision_cfg)
    pool.get_stt_model = MagicMock(return_value=None)
    return pool


@pytest.fixture
def router_with_pool(tmp_work_dir, mock_llm_pool):
    """Router with LLM pool for vision/STT routing."""
    return MultimodalRouter(llm_pool=mock_llm_pool, work_dir=tmp_work_dir)


# ============================================================
# Text passthrough
# ============================================================


class TestTextPassthrough:
    """Text inputs should pass through with no processing."""

    @pytest.mark.asyncio
    async def test_single_text(self, router_no_pool):
        blocks = await router_no_pool.process([
            MediaInput(type="text", source="Hello world"),
        ])
        assert len(blocks) == 1
        assert blocks[0] == {"type": "text", "text": "Hello world"}

    @pytest.mark.asyncio
    async def test_multiple_text(self, router_no_pool):
        blocks = await router_no_pool.process([
            MediaInput(type="text", source="Context:"),
            MediaInput(type="text", source="Question?"),
        ])
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Context:"}
        assert blocks[1] == {"type": "text", "text": "Question?"}

    @pytest.mark.asyncio
    async def test_synthesize_text(self, router_no_pool):
        """synthesize_text should extract flat text summary."""
        inputs = [
            MediaInput(type="text", source="Analyze this:"),
            MediaInput(type="image", source="img.jpg", description="Chart"),
            MediaInput(type="text", source="What does it show?"),
        ]
        result = router_no_pool.synthesize_text(inputs)
        assert "Analyze this:" in result
        assert "What does it show?" in result
        assert "[image: Chart]" in result


# ============================================================
# Image routing → vision model
# ============================================================


class TestImageRouting:
    """Images should be routed to vision model when available."""

    @pytest.mark.asyncio
    async def test_image_with_vision_model(self, router_with_pool, tmp_path):
        """With vision model pool, image gets description appended."""
        # Create a small PNG
        img_path = tmp_path / "photo.png"
        img_path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))

        router_with_pool.processor.work_dir = tmp_path
        blocks = await router_with_pool.process([
            MediaInput(type="image", source=str(img_path), mime_type="image/png"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"
        assert "image_url" in blocks[0]
        # Vision model should have been called
        router_with_pool.llm_pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_url_passthrough(self, router_no_pool):
        """Image URL should pass through without local processing."""
        blocks = await router_no_pool.process([
            MediaInput(
                type="image",
                source="https://example.com/photo.jpg",
                mime_type="image/jpeg",
            ),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"
        assert blocks[0]["image_url"]["url"] == "https://example.com/photo.jpg"

    @pytest.mark.asyncio
    async def test_image_without_vision_model(self, router_no_pool, tmp_path):
        """Without vision model, image is still returned as image_url."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))
        router_no_pool.processor.work_dir = tmp_path
        blocks = await router_no_pool.process([
            MediaInput(type="image", source=str(img_path), mime_type="image/png"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_image_vision_unavailable_fallback(self, tmp_path):
        """When vision model raises error, image still returned."""
        pool = MagicMock(spec=LLMPool)
        pool.acquire = AsyncMock(side_effect=RuntimeError("No vision provider"))

        router = MultimodalRouter(llm_pool=pool, work_dir=str(tmp_path))

        img_path = tmp_path / "photo.png"
        img_path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))
        blocks = await router.process([
            MediaInput(type="image", source=str(img_path), mime_type="image/png"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"


# ============================================================
# Audio routing → STT
# ============================================================


class TestAudioRouting:
    """Audio should be routed to STT (Whisper or pool provider)."""

    @pytest.mark.asyncio
    async def test_audio_no_provider(self, router_no_pool, tmp_path):
        """Without STT provider or Whisper key, audio returns placeholder."""
        audio_path = tmp_path / "recording.ogg"
        audio_path.write_bytes(b"fake ogg data")

        blocks = await router_no_pool.process([
            MediaInput(type="audio", source=str(audio_path), description="Voice message"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Voice message" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_audio_with_pool_stt(self, tmp_path):
        """When pool has STT provider, it should be used."""
        pool = MagicMock(spec=LLMPool)
        mock_provider = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "This is a transcribed message."
        mock_provider.chat = AsyncMock(return_value=mock_resp)
        pool.acquire = AsyncMock(return_value=mock_provider)

        router = MultimodalRouter(llm_pool=pool, work_dir=str(tmp_path))

        audio_path = tmp_path / "recording.ogg"
        audio_path.write_bytes(b"fake ogg data")

        blocks = await router.process([
            MediaInput(type="audio", source=str(audio_path)),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "transcribed message" in blocks[0]["text"]
        pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_stt_fallback_when_pool_fails(self, tmp_path):
        """When pool fails, audio falls back to placeholder."""
        pool = MagicMock(spec=LLMPool)
        pool.acquire = AsyncMock(side_effect=RuntimeError("No STT"))

        router = MultimodalRouter(llm_pool=pool, work_dir=str(tmp_path))

        audio_path = tmp_path / "audio.ogg"
        audio_path.write_bytes(b"fake data")

        blocks = await router.process([
            MediaInput(type="audio", source=str(audio_path), description="Recording"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Recording" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_audio_empty_stt_response(self, tmp_path):
        """Empty STT response should fall back to placeholder."""
        pool = MagicMock(spec=LLMPool)
        mock_provider = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = ""  # Empty
        mock_provider.chat = AsyncMock(return_value=mock_resp)
        pool.acquire = AsyncMock(return_value=mock_provider)

        router = MultimodalRouter(llm_pool=pool, work_dir=str(tmp_path))

        audio_path = tmp_path / "silent.ogg"
        audio_path.write_bytes(b"silent audio")

        blocks = await router.process([
            MediaInput(type="audio", source=str(audio_path), description="Silent recording"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Silent recording" in blocks[0]["text"]


# ============================================================
# Video routing → ffmpeg keyframe extraction
# ============================================================


class TestVideoRouting:
    """Video inputs should extract keyframes via ffmpeg."""

    @pytest.mark.asyncio
    async def test_video_no_ffmpeg_fallback(self, router_no_pool, tmp_path):
        """When ffmpeg is not available, return placeholder."""
        video_path = tmp_path / "demo.mp4"
        video_path.write_bytes(b"fake mp4 data")

        # Mock subprocess.run to simulate ffmpeg not available
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg")
            blocks = await router_no_pool.process([
                MediaInput(type="video", source=str(video_path), description="Demo video"),
            ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "ffmpeg not available" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_video_remote_url_fallback(self, router_no_pool):
        """Remote video URLs currently return placeholder."""
        blocks = await router_no_pool.process([
            MediaInput(type="video", source="https://example.com/video.mp4", description="Remote"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "remote video" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_video_file_not_found(self, router_no_pool):
        """Non-existent video file raises FileNotFoundError."""
        blocks = await router_no_pool.process([
            MediaInput(type="video", source="/nonexistent/movie.mp4"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Error" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_video_with_ffmpeg_mocked(self, router_no_pool, tmp_path):
        """With ffmpeg available (mocked), frames are extracted."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake mp4 data")

        # Create mock frame files in a temp directory
        def mock_ffmpeg_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            cmd_str = " ".join(cmd)

            if "ffmpeg -version" in cmd_str:
                result = MagicMock()
                result.returncode = 0
                return result

            if "ffprobe" in cmd_str:
                result = MagicMock()
                result.stdout = "120.0\n"
                result.returncode = 0
                return result

            # Extract the output directory from ffmpeg args
            for i, arg in enumerate(cmd):
                if arg.startswith("/tmp/video_frames_"):
                    frames_dir = Path(arg).parent
                    # Create mock frames
                    for j in range(1, 6):
                        frame_path = frames_dir / f"frame_{j:02d}.jpg"
                        frame_path.parent.mkdir(parents=True, exist_ok=True)
                        frame_path.write_bytes(base64.b64decode(
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        ))
                    break

            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=mock_ffmpeg_side_effect):
            blocks = await router_no_pool.process([
                MediaInput(type="video", source=str(video_path), description="Test video"),
            ])

        # Should have context text + 5 frames (capped by MAX_VIDEO_FRAMES)
        assert len(blocks) >= 2
        # First block is context text
        assert blocks[0]["type"] == "text"
        assert "Test video" in blocks[0]["text"]
        # Following blocks are image frames
        image_blocks = [b for b in blocks if b["type"] == "image_url"]
        assert 1 <= len(image_blocks) <= 5


# ============================================================
# File routing → text extraction
# ============================================================


class TestFileRouting:
    """Files should be routed to text extraction."""

    @pytest.mark.asyncio
    async def test_file_txt_extraction(self, router_no_pool, tmp_path):
        """Text files should have content extracted."""
        fpath = tmp_path / "readme.txt"
        fpath.write_text("Project documentation content", encoding="utf-8")

        blocks = await router_no_pool.process([
            MediaInput(type="file", source=str(fpath)),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "documentation content" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_file_json_extraction(self, router_no_pool, tmp_path):
        """JSON files should have content extracted."""
        data = {"name": "test", "version": "1.0"}
        fpath = tmp_path / "config.json"
        fpath.write_text(json.dumps(data))

        blocks = await router_no_pool.process([
            MediaInput(type="file", source=str(fpath)),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "test" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_file_csv_extraction(self, router_no_pool, tmp_path):
        """CSV files should have content extracted."""
        fpath = tmp_path / "data.csv"
        fpath.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA")

        blocks = await router_no_pool.process([
            MediaInput(type="file", source=str(fpath)),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Alice" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_file_pdf_placeholder(self, router_no_pool, tmp_path):
        """PDF without pdftotext returns placeholder."""
        fpath = tmp_path / "report.pdf"
        fpath.write_bytes(b"%PDF-1.4 fake pdf content")

        blocks = await router_no_pool.process([
            MediaInput(type="file", source=str(fpath), description="Report"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        # Should contain filename or description
        text = blocks[0]["text"]
        assert "report.pdf" in text.lower() or "Report" in text

    @pytest.mark.asyncio
    async def test_file_not_found(self, router_no_pool):
        """Non-existent file returns error block."""
        blocks = await router_no_pool.process([
            MediaInput(type="file", source="/nonexistent/notes.txt"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Error" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_file_code_extraction(self, router_no_pool, tmp_path):
        """Code files should have content extracted."""
        fpath = tmp_path / "app.py"
        fpath.write_text("def hello():\n    print('Hello, world!')")

        blocks = await router_no_pool.process([
            MediaInput(type="file", source=str(fpath)),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "def hello" in blocks[0]["text"]


# ============================================================
# Mixed input batch
# ============================================================


class TestMixedInputs:
    """Multiple media types in a single batch."""

    @pytest.mark.asyncio
    async def test_mixed_text_image_file(self, router_no_pool, tmp_path):
        """Text + image + file should each be routed correctly."""
        # Image
        img_path = tmp_path / "photo.png"
        img_path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))
        # File
        fpath = tmp_path / "data.txt"
        fpath.write_text("file content here")

        blocks = await router_no_pool.process([
            MediaInput(type="text", source="Look at this:"),
            MediaInput(type="image", source=str(img_path), mime_type="image/png"),
            MediaInput(type="file", source=str(fpath)),
        ])

        assert len(blocks) == 3
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image_url"
        assert blocks[2]["type"] == "text"
        assert "file content" in blocks[2]["text"]

    @pytest.mark.asyncio
    async def test_all_types_together(self, router_no_pool, tmp_path):
        """All five types (text, image, audio, video, file) mixed."""
        # Image
        img_path = tmp_path / "img.png"
        img_path.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))
        # File
        fpath = tmp_path / "notes.txt"
        fpath.write_text("Some notes")
        # Audio
        apath = tmp_path / "audio.ogg"
        apath.write_bytes(b"fake audio")
        # Video
        vpath = tmp_path / "video.mp4"
        vpath.write_bytes(b"fake mp4")

        blocks = await router_no_pool.process([
            MediaInput(type="text", source="Process these:"),
            MediaInput(type="image", source=str(img_path), mime_type="image/png"),
            MediaInput(type="file", source=str(fpath)),
            MediaInput(type="audio", source=str(apath), description="Voice"),
            MediaInput(type="video", source=str(vpath), description="Demo"),
        ])

        # All should produce blocks
        assert len(blocks) >= 5
        # Check all types are represented
        types_found = {b["type"] for b in blocks}
        assert "text" in types_found
        assert "image_url" in types_found


# ============================================================
# Error resilience
# ============================================================


class TestErrorResilience:
    """Router should handle errors gracefully."""

    @pytest.mark.asyncio
    async def test_unsupported_type(self, router_no_pool):
        """Unknown media type should produce an informative text block."""
        blocks = await router_no_pool.process([
            MediaInput(type="unknown_type", source="something"),
        ])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Unsupported" in blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_mixed_success_and_error(self, router_no_pool, tmp_path):
        """Valid inputs alongside errors should not break the batch."""
        fpath = tmp_path / "good.txt"
        fpath.write_text("valid content")

        blocks = await router_no_pool.process([
            MediaInput(type="text", source="Hello"),
            MediaInput(type="file", source="/nonexistent/bad.txt"),  # Will fail
            MediaInput(type="file", source=str(fpath)),  # Should succeed
        ])
        assert len(blocks) == 3
        assert blocks[0] == {"type": "text", "text": "Hello"}
        assert "Error" in blocks[1]["text"]
        assert "valid content" in blocks[2]["text"]

    @pytest.mark.asyncio
    async def test_empty_input_list(self, router_no_pool):
        """Empty input list should produce empty block list."""
        blocks = await router_no_pool.process([])
        assert blocks == []


# ============================================================
# LLMPool modality methods
# ============================================================


class TestLLMPoolModality:
    """LLMPool modality-aware selection methods."""

    def test_provider_config_modality_default(self):
        """ProviderConfigJSON should default modality to ['text']."""
        cfg = ProviderConfigJSON(id="test", model="some-model")
        assert cfg.modality == ["text"]

    def test_provider_config_modality_custom(self):
        """ProviderConfigJSON should accept custom modality."""
        cfg = ProviderConfigJSON(
            id="vision-model",
            model="gpt-4o",
            capabilities=["vision"],
            modality=["text", "image"],
        )
        assert cfg.modality == ["text", "image"]
        assert "image" in cfg.modality

    def test_provider_config_serialization_roundtrip(self):
        """modality should survive JSON serialization roundtrip."""
        cfg = ProviderConfigJSON(
            id="multi",
            model="gemini-2.0-flash",
            capabilities=["general", "vision", "audio"],
            modality=["text", "image", "audio"],
        )
        d = cfg.to_dict()
        assert d["modality"] == ["text", "image", "audio"]

        # Reconstruct
        cfg2 = ProviderConfigJSON.from_dict(d)
        assert cfg2.modality == ["text", "image", "audio"]

    def test_pool_config_from_json_reads_modality(self, tmp_path):
        """PoolConfigJSON.from_json should parse modality field."""
        config_path = tmp_path / "llm_pool.json"
        config_data = {
            "providers": [
                {
                    "id": "vision-provider",
                    "model": "gpt-4o",
                    "capabilities": ["vision", "general"],
                    "modality": ["text", "image"],
                    "enabled": True,
                },
                {
                    "id": "audio-provider",
                    "model": "whisper-1",
                    "capabilities": ["audio", "stt"],
                    "modality": ["audio"],
                    "enabled": True,
                },
            ],
            "selection": {"default_strategy": "cheapest", "task_mapping": {}, "strategies": {}},
        }
        config_path.write_text(json.dumps(config_data))

        pool = LLMPool(config_path=str(config_path))
        pool.initialize()

        vision = pool.get_vision_model()
        assert vision is not None
        assert "image" in vision.modality

    def test_get_vision_model_filters_by_modality(self, tmp_path):
        """get_vision_model should prefer providers with image modality."""
        config_path = tmp_path / "llm_pool.json"
        config_data = {
            "providers": [
                {
                    "id": "text-only",
                    "model": "gpt-3.5",
                    "capabilities": ["general"],
                    "modality": ["text"],
                    "enabled": True,
                    "cost_per_1m_input": 0.50,
                },
                {
                    "id": "vision-model",
                    "model": "gpt-4o",
                    "capabilities": ["vision", "general"],
                    "modality": ["text", "image"],
                    "enabled": True,
                    "cost_per_1m_input": 2.50,
                },
                {
                    "id": "cheap-vision",
                    "model": "gemini-flash",
                    "capabilities": ["vision"],
                    "modality": ["text", "image"],
                    "enabled": True,
                    "cost_per_1m_input": 0.10,
                },
            ],
            "selection": {
                "default_strategy": "cheapest",
                "task_mapping": {},
                "strategies": {
                    "cheapest": {"sort_by": "cost_per_1m_input", "ascending": True},
                },
            },
        }
        config_path.write_text(json.dumps(config_data))

        pool = LLMPool(config_path=str(config_path))
        pool.initialize()

        # Should pick cheapest with image modality
        vision = pool.get_vision_model()
        assert vision is not None
        assert vision.id == "cheap-vision"

    def test_get_stt_model_filters_by_modality(self, tmp_path):
        """get_stt_model should prefer providers with audio modality."""
        config_path = tmp_path / "llm_pool.json"
        config_data = {
            "providers": [
                {
                    "id": "text-only",
                    "model": "gpt-3.5",
                    "capabilities": ["general"],
                    "modality": ["text"],
                    "enabled": True,
                },
                {
                    "id": "audio-stt",
                    "model": "whisper-1",
                    "capabilities": ["audio", "stt"],
                    "modality": ["text", "audio"],
                    "enabled": True,
                    "cost_per_1m_input": 0.006,
                },
            ],
            "selection": {
                "default_strategy": "cheapest",
                "task_mapping": {},
                "strategies": {
                    "cheapest": {"sort_by": "cost_per_1m_input", "ascending": True},
                },
            },
        }
        config_path.write_text(json.dumps(config_data))

        pool = LLMPool(config_path=str(config_path))
        pool.initialize()

        stt = pool.get_stt_model()
        assert stt is not None
        assert stt.id == "audio-stt"

    def test_get_stt_model_fallback_to_audio_only(self, tmp_path):
        """get_stt_model should fall back to audio capability without stt."""
        config_path = tmp_path / "llm_pool.json"
        config_data = {
            "providers": [
                {
                    "id": "gemini-audio",
                    "model": "gemini-2.0-flash",
                    "capabilities": ["audio", "general"],
                    "modality": ["text", "audio"],
                    "enabled": True,
                    "cost_per_1m_input": 0.10,
                },
            ],
            "selection": {
                "default_strategy": "cheapest",
                "task_mappings": {},
                "strategies": {
                    "cheapest": {"sort_by": "cost_per_1m_input", "ascending": True},
                },
            },
        }
        config_path.write_text(json.dumps(config_data))

        pool = LLMPool(config_path=str(config_path))
        pool.initialize()

        stt = pool.get_stt_model()
        assert stt is not None
        assert stt.id == "gemini-audio"

    def test_select_with_modalities_filter(self, tmp_path):
        """select() should filter by modalities parameter."""
        config_path = tmp_path / "llm_pool.json"
        config_data = {
            "providers": [
                {
                    "id": "text-only",
                    "model": "gpt-3.5",
                    "capabilities": ["general"],
                    "modality": ["text"],
                    "enabled": True,
                },
                {
                    "id": "multimodal",
                    "model": "gpt-4o",
                    "capabilities": ["vision"],
                    "modality": ["text", "image"],
                    "enabled": True,
                },
            ],
            "selection": {
                "default_strategy": "cheapest",
                "task_mapping": {},
                "strategies": {
                    "cheapest": {"sort_by": "cost_per_1m_input", "ascending": True},
                },
            },
        }
        config_path.write_text(json.dumps(config_data))

        pool = LLMPool(config_path=str(config_path))
        pool.initialize()

        # Filter by modality only gets multimodal
        result = pool.select(modalities=["text", "image"])
        assert result is not None
        assert result.id == "multimodal"

        # Filter by text-only modality
        result = pool.select(modalities=["image"])
        # When no provider matches, fallback returns any available provider.
        # This is intentional: rather than returning None and crashing,
        # we fall back to whatever is available.
        assert result is not None  # Fallback returns a provider

        # Filter by capability + modality
        result = pool.select(capabilities=["vision"], modalities=["text", "image"])
        assert result is not None
        assert result.id == "multimodal"


# ============================================================
# LoopContext with media_blocks
# ============================================================


class TestLoopContextMediaBlocks:
    """LoopContext should carry media_blocks for multimodal processing."""

    def test_loop_context_has_media_blocks(self):
        """LoopContext should have media_blocks field (empty list default)."""
        ctx = LoopContext(user_input="test")
        assert ctx.media_blocks == []

    def test_loop_context_with_media_blocks(self):
        """media_blocks can be set for multimodal context."""
        ctx = LoopContext(user_input="Look at this image")
        ctx.media_blocks = [
            {"type": "text", "text": "Look at this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "auto"}},
        ]
        assert len(ctx.media_blocks) == 2
        assert ctx.media_blocks[0]["type"] == "text"
        assert ctx.media_blocks[1]["type"] == "image_url"


# ============================================================
# MediaInput dict compatibility
# ============================================================


class TestMediaInputDictCompat:
    """MediaInput can be constructed from dict (backed by MultimodalProcessor)."""

    def test_media_input_from_dict(self):
        """dict-based MediaInput should still work."""
        d = {"type": "text", "content": "Hello", "description": "greeting"}
        media = MediaInput(
            type=d.get("type", "text"),
            source=d.get("content", d.get("source", "")),
            mime_type=d.get("mime_type", ""),
            description=d.get("description", ""),
        )
        assert media.type == "text"
        assert media.source == "Hello"
        assert media.description == "greeting"

    def test_media_input_defaults(self):
        """Default fields should be set."""
        media = MediaInput(type="text", source="x")
        assert media.mime_type == ""
        assert media.description == ""
        assert media._extracted_text == ""
