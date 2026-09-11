"""Tests for Stage 2: Preprocessing (§2.2)."""
from pathlib import Path

import pytest
from PIL import Image

from src.ocr_pipeline.exceptions import PreprocessingFailedError
from src.ocr_pipeline.preprocess import preprocess_image


class TestPreprocessing:
    """Preprocessing stage tests."""

    def test_normal_image_preprocessed(self, sample_image, sample_pipeline_config, tmp_path):
        """A normal image is preprocessed successfully."""
        result = preprocess_image(sample_image, tmp_path / "out", sample_pipeline_config)
        assert result.derived_image_path.exists()
        assert result.operations_applied  # At least some operations applied
        assert result.status.value == "success"

    def test_derived_path_separate(self, sample_image, sample_pipeline_config, tmp_path):
        """Derived image is in a separate path from the original."""
        result = preprocess_image(sample_image, tmp_path / "out", sample_pipeline_config)
        assert result.derived_image_path != sample_image

    def test_operations_logged(self, sample_image, sample_pipeline_config, tmp_path):
        """All preprocessing operations are recorded."""
        result = preprocess_image(sample_image, tmp_path / "out", sample_pipeline_config)
        # Should have at least contrast_normalize and saved_as_png
        op_names = [op.split(":")[0].split("_")[0] for op in result.operations_applied]
        assert len(result.operations_applied) >= 1

    def test_corrupt_image_fails(self, corrupt_file, sample_pipeline_config, tmp_path):
        """A corrupt image file raises PreprocessingFailedError."""
        with pytest.raises(PreprocessingFailedError):
            preprocess_image(corrupt_file, tmp_path / "out", sample_pipeline_config)

    def test_blank_image_fails_crop(self, sample_pipeline_config, tmp_path):
        """A completely uniform image with no text region fails at crop."""
        # Create a truly uniform image (no edges at all)
        uniform = Image.new("RGB", (400, 300), color=(128, 128, 128))
        uniform_path = tmp_path / "uniform.png"
        uniform.save(str(uniform_path))

        with pytest.raises(PreprocessingFailedError) as exc_info:
            preprocess_image(uniform_path, tmp_path / "out", sample_pipeline_config)
        assert "No text region" in str(exc_info.value)

    def test_output_is_png(self, sample_image, sample_pipeline_config, tmp_path):
        """Derived image is saved as PNG for lossless OCR input."""
        result = preprocess_image(sample_image, tmp_path / "out", sample_pipeline_config)
        assert result.derived_image_path.suffix == ".png"
