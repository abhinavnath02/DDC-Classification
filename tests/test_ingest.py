"""Tests for Stage 1: Ingest (§2.1)."""
import hashlib
from pathlib import Path

import pytest

from src.ocr_pipeline.exceptions import UnsupportedFormatError
from src.ocr_pipeline.ingest import ingest_image


class TestIngest:
    """Ingest stage tests — valid and invalid inputs."""

    def test_valid_jpeg_ingested(self, sample_image, sample_pipeline_config):
        """A valid JPEG is accepted and hashed."""
        result = ingest_image(sample_image, sample_pipeline_config)
        assert result.content_hash
        assert len(result.content_hash) == 64  # SHA-256 hex
        assert result.file_format == ".jpg"
        assert result.archive_path.exists()
        assert result.original_filename == sample_image.name

    def test_content_hash_matches(self, sample_image, sample_pipeline_config):
        """Content hash matches independent SHA-256 of the raw file."""
        expected = hashlib.sha256(sample_image.read_bytes()).hexdigest()
        result = ingest_image(sample_image, sample_pipeline_config)
        assert result.content_hash == expected

    def test_valid_png_ingested(self, blank_image, sample_pipeline_config):
        """A valid PNG is accepted."""
        result = ingest_image(blank_image, sample_pipeline_config)
        assert result.file_format == ".png"

    def test_unsupported_format_rejected(self, unsupported_file, sample_pipeline_config):
        """An unsupported format raises UnsupportedFormatError, not a crash."""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            ingest_image(unsupported_file, sample_pipeline_config)
        assert ".pdf" in str(exc_info.value)

    def test_missing_file_raises(self, sample_pipeline_config):
        """A non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ingest_image("/nonexistent/path.jpg", sample_pipeline_config)

    def test_original_not_mutated(self, sample_image, sample_pipeline_config):
        """The original image file is not modified during ingest."""
        original_bytes = sample_image.read_bytes()
        ingest_image(sample_image, sample_pipeline_config)
        assert sample_image.read_bytes() == original_bytes

    def test_provenance_recorded(self, sample_image, sample_pipeline_config):
        """Upload timestamp, batch_id, and session_id are recorded."""
        result = ingest_image(
            sample_image, sample_pipeline_config,
            batch_id="test_batch", session_id="test_session"
        )
        assert result.batch_id == "test_batch"
        assert result.session_id == "test_session"
        assert result.upload_timestamp  # ISO-8601 string

    def test_archive_separate_from_original(self, sample_image, sample_pipeline_config):
        """Archive path is different from original path."""
        result = ingest_image(sample_image, sample_pipeline_config)
        assert result.archive_path != result.original_path
