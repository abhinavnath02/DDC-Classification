"""Shared test fixtures and helpers for the OCR pipeline test suite."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_files"


@pytest.fixture
def repo_root():
    """Return the repository root path."""
    return REPO_ROOT


@pytest.fixture
def config():
    """Load pipeline config from the repo."""
    from src.ocr_pipeline.config import PipelineConfig
    return PipelineConfig(REPO_ROOT / "pipeline_config.yaml")


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory."""
    return tmp_path


@pytest.fixture
def sample_image(tmp_path) -> Path:
    """Create a synthetic clear-text image for testing.

    Renders 'The Great Gatsby' title-page style text onto a white background.
    This is synthetic, not a real book scan — documented in OCR_PIPELINE.md.
    """
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use default font (Pillow built-in, always available)
    try:
        font_large = ImageFont.truetype("arial.ttf", 48)
        font_medium = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except (IOError, OSError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Title (large, centered, top area)
    draw.text((150, 80), "The Great Gatsby", fill=(0, 0, 0), font=font_large)

    # Subtitle (medium, below title)
    draw.text((200, 180), "A Novel", fill=(50, 50, 50), font=font_medium)

    # Author (with "by" prefix, below subtitle)
    draw.text((250, 280), "by F. Scott Fitzgerald", fill=(80, 80, 80), font=font_small)

    path = tmp_path / "clear_title_page.jpg"
    img.save(str(path), format="JPEG", quality=95)
    return path


@pytest.fixture
def blurry_image(tmp_path) -> Path:
    """Create a blurry/degraded synthetic image."""
    from PIL import ImageFilter

    img = Image.new("RGB", (800, 600), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.text((100, 200), "Blurry Text Here", fill=(180, 180, 180), font=font)

    # Apply heavy blur
    img = img.filter(ImageFilter.GaussianBlur(radius=8))

    path = tmp_path / "blurry_cover.jpg"
    img.save(str(path), format="JPEG", quality=30)
    return path


@pytest.fixture
def blank_image(tmp_path) -> Path:
    """Create a blank white image with no text."""
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    path = tmp_path / "blank.png"
    img.save(str(path), format="PNG")
    return path


@pytest.fixture
def corrupt_file(tmp_path) -> Path:
    """Create a file with invalid image data."""
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not an image file at all")
    return path


@pytest.fixture
def unsupported_file(tmp_path) -> Path:
    """Create a file with an unsupported extension."""
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.4 fake pdf content")
    return path


@pytest.fixture
def title_only_image(tmp_path) -> Path:
    """Create an image with only a title (no subtitle, no author)."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.text((100, 150), "Moby Dick", fill=(0, 0, 0), font=font)

    path = tmp_path / "title_only.jpg"
    img.save(str(path), format="JPEG", quality=95)
    return path


@pytest.fixture
def sample_pipeline_config(tmp_path, config):
    """Return a config with paths redirected to tmp directories."""
    import yaml

    config_data = {
        "ocr": {
            "engine": "tesseract",
            "min_word_confidence": 40,
            "min_mean_confidence": 60,
            "tesseract_cmd": None,
        },
        "preprocessing": {
            "deskew_enabled": True,
            "contrast_normalize": True,
            "crop_to_text": True,
        },
        "routing": {
            "classification_auto_accept_threshold": 0.85,
            "classification_needs_review_threshold": 0.50,
            "ocr_reject_threshold": 40,
        },
        "paths": {
            "model_artifact": str(REPO_ROOT / "models" / "baseline_classifier.joblib"),
            "image_inbox": str(tmp_path / "inbox"),
            "processed_output": str(tmp_path / "processed"),
            "review_queue": str(tmp_path / "review"),
            "archive": str(tmp_path / "archive"),
        },
        "supported_formats": [".jpg", ".jpeg", ".png", ".heic", ".tiff", ".tif"],
        "forbidden_model_features": [
            "ddc_raw", "label_name", "label_override", "label_level_1",
            "review_status", "document_id", "source_work_ids",
            "split_group_id", "previous_split_memberships",
            "prior_holdout_status", "source_urls", "retrieved_at",
            "edition_ids", "isbns", "first_sentence_sources",
            "description_source_url", "text_enrichment_sources",
            "deduplication_status",
        ],
        "valid_labels": [f"{i}00" for i in range(10)],
    }

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(config_data), encoding="utf-8")

    from src.ocr_pipeline.config import PipelineConfig
    return PipelineConfig(config_path)
