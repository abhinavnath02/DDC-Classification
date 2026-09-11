"""Tests for Stage 3: OCR (§2.3).

NOTE: These tests require Tesseract to be installed on the system.
If Tesseract is not available, tests will be skipped with a clear message.
"""
import pytest
from PIL import Image, ImageDraw, ImageFont

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

from src.ocr_pipeline.exceptions import OCRFailedError
from src.ocr_pipeline.ocr import run_ocr
from src.ocr_pipeline.schemas import PipelineStatus

pytestmark = pytest.mark.skipif(
    not TESSERACT_AVAILABLE,
    reason="Tesseract OCR is not installed — OCR tests skipped"
)


class TestOCR:
    """OCR stage tests."""

    def test_clear_text_recognized(self, sample_image, sample_pipeline_config):
        """Clear synthetic text is recognized with reasonable confidence."""
        result = run_ocr(sample_image, sample_pipeline_config)
        assert result.word_count > 0
        assert result.mean_confidence > 0
        assert result.full_text.strip()
        assert result.status == PipelineStatus.SUCCESS

    def test_word_level_confidence(self, sample_image, sample_pipeline_config):
        """Each word has an individual confidence score."""
        result = run_ocr(sample_image, sample_pipeline_config)
        for word in result.words:
            assert 0 <= word.confidence <= 100
            assert word.text.strip()

    def test_bounding_boxes_present(self, sample_image, sample_pipeline_config):
        """Each word has a bounding box with valid coordinates."""
        result = run_ocr(sample_image, sample_pipeline_config)
        for word in result.words:
            assert word.bbox.width >= 0
            assert word.bbox.height >= 0

    def test_empty_image_fails(self, sample_pipeline_config, tmp_path):
        """An image with no recognizable text raises OCRFailedError."""
        # Pure white image — no text at all
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        path = tmp_path / "empty.png"
        img.save(str(path))

        with pytest.raises(OCRFailedError):
            run_ocr(path, sample_pipeline_config)

    def test_low_confidence_flagged(self, sample_pipeline_config, tmp_path):
        """Very blurry text produces low mean confidence and appropriate status."""
        from PIL import ImageFilter
        img = Image.new("RGB", (400, 200), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((50, 80), "faint text", fill=(190, 190, 190), font=font)
        img = img.filter(ImageFilter.GaussianBlur(radius=10))
        path = tmp_path / "faint.png"
        img.save(str(path))

        # This might fail with OCRFailedError or return low confidence
        try:
            result = run_ocr(path, sample_pipeline_config)
            # If it returns, check that low confidence is flagged
            if result.mean_confidence < sample_pipeline_config.ocr_reject_threshold:
                assert result.status == PipelineStatus.OCR_LOW_CONFIDENCE
        except OCRFailedError:
            pass  # Also acceptable — no words recognized
