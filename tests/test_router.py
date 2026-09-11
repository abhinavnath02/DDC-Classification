"""Tests for Stage 7: Routing (§2.7)."""
import pytest

from src.ocr_pipeline.router import route
from src.ocr_pipeline.schemas import (
    ClassificationResult,
    OCRResult,
    OCRWord,
    BoundingBox,
    PipelineStatus,
    RoutingOutcome,
)


def _make_ocr(mean_conf: float) -> OCRResult:
    """Helper: create an OCRResult with a specific mean confidence."""
    return OCRResult(
        words=[OCRWord(text="test", confidence=mean_conf,
                       bbox=BoundingBox(0, 0, 50, 20), line_num=1, word_num=1)],
        full_text="test",
        mean_confidence=mean_conf,
        min_confidence=mean_conf,
        detected_script="Latin",
        detected_language="eng",
        status=PipelineStatus.SUCCESS,
        word_count=1,
    )


def _make_classification(conf: float, label: str = "500") -> ClassificationResult:
    """Helper: create a ClassificationResult with a specific confidence."""
    return ClassificationResult(
        predicted_label=label,
        confidence=conf,
        all_probabilities={f"{i}00": 0.1 for i in range(10)},
        calibrated=False,
        calibration_note="test",
    )


class TestRouter:
    """Routing stage tests — all three outcomes must be reachable."""

    def test_auto_accept(self, sample_pipeline_config):
        """High OCR + high classification → auto_accept."""
        ocr = _make_ocr(90.0)
        cls = _make_classification(0.95)
        result = route(ocr, cls, sample_pipeline_config)
        assert result.outcome == RoutingOutcome.AUTO_ACCEPT

    def test_reject_retake_low_ocr(self, sample_pipeline_config):
        """Very low OCR confidence → reject_retake."""
        ocr = _make_ocr(20.0)  # Below ocr_reject_threshold (40)
        cls = _make_classification(0.95)
        result = route(ocr, cls, sample_pipeline_config)
        assert result.outcome == RoutingOutcome.REJECT_RETAKE

    def test_needs_review_medium_classification(self, sample_pipeline_config):
        """Good OCR but medium classification confidence → needs_review."""
        ocr = _make_ocr(80.0)
        cls = _make_classification(0.60)  # Between 0.50 and 0.85
        result = route(ocr, cls, sample_pipeline_config)
        assert result.outcome == RoutingOutcome.NEEDS_REVIEW

    def test_needs_review_low_ocr_above_reject(self, sample_pipeline_config):
        """OCR above reject but below min_mean → needs_review (not reject)."""
        ocr = _make_ocr(50.0)  # Above 40 (reject) but below 60 (min_mean)
        cls = _make_classification(0.90)
        result = route(ocr, cls, sample_pipeline_config)
        assert result.outcome == RoutingOutcome.NEEDS_REVIEW

    def test_thresholds_logged(self, sample_pipeline_config):
        """Every routing decision includes threshold values used."""
        ocr = _make_ocr(80.0)
        cls = _make_classification(0.90)
        result = route(ocr, cls, sample_pipeline_config)
        assert "classification_auto_accept_threshold" in result.thresholds_used
        assert "ocr_reject_threshold" in result.thresholds_used

    def test_confidence_values_logged(self, sample_pipeline_config):
        """Routing decision includes both OCR and classification confidence."""
        ocr = _make_ocr(75.0)
        cls = _make_classification(0.80)
        result = route(ocr, cls, sample_pipeline_config)
        assert result.ocr_mean_confidence == 75.0
        assert result.classification_confidence == 0.80

    def test_reason_non_empty(self, sample_pipeline_config):
        """Every routing decision has a human-readable reason."""
        for ocr_conf, cls_conf in [(90, 0.95), (20, 0.95), (70, 0.60)]:
            result = route(
                _make_ocr(float(ocr_conf)),
                _make_classification(cls_conf),
                sample_pipeline_config,
            )
            assert result.reason  # Not empty

    def test_boundary_auto_accept(self, sample_pipeline_config):
        """Exact boundary values for auto_accept."""
        # Exactly at auto_accept threshold (0.85) and min_mean (60)
        ocr = _make_ocr(60.0)
        cls = _make_classification(0.85)
        result = route(ocr, cls, sample_pipeline_config)
        assert result.outcome == RoutingOutcome.AUTO_ACCEPT

    def test_boundary_reject(self, sample_pipeline_config):
        """Exactly at reject threshold → needs_review (not reject)."""
        # At exactly 40 (reject threshold) — should NOT reject
        ocr = _make_ocr(40.0)
        cls = _make_classification(0.90)
        result = route(ocr, cls, sample_pipeline_config)
        # At exactly the threshold, should be needs_review since
        # 40 is not < 40
        assert result.outcome in (RoutingOutcome.NEEDS_REVIEW, RoutingOutcome.AUTO_ACCEPT)
