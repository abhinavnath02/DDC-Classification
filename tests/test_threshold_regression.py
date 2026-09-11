"""Threshold regression tests (§5).

Changing a threshold constant should have a visible effect on a fixed
test batch's routing outcome, proving the threshold is actually wired in.
"""
import pytest
import yaml

from src.ocr_pipeline.config import PipelineConfig
from src.ocr_pipeline.router import route
from src.ocr_pipeline.schemas import (
    BoundingBox,
    ClassificationResult,
    OCRResult,
    OCRWord,
    PipelineStatus,
    RoutingOutcome,
)


def _make_ocr(conf: float) -> OCRResult:
    return OCRResult(
        words=[OCRWord("word", conf, BoundingBox(0,0,50,20), 1, 1)],
        full_text="word", mean_confidence=conf, min_confidence=conf,
        detected_script="Latin", detected_language="eng",
        status=PipelineStatus.SUCCESS, word_count=1,
    )


def _make_cls(conf: float) -> ClassificationResult:
    return ClassificationResult(
        predicted_label="500", confidence=conf,
        all_probabilities={f"{i}00": 0.1 for i in range(10)},
        calibrated=False, calibration_note="test",
    )


def _make_config(tmp_path, auto_thresh=0.85, review_thresh=0.50, ocr_reject=40, ocr_min_mean=60):
    """Create a config with custom thresholds."""
    config_data = {
        "ocr": {"engine": "tesseract", "min_word_confidence": 40,
                "min_mean_confidence": ocr_min_mean, "tesseract_cmd": None},
        "preprocessing": {"deskew_enabled": True, "contrast_normalize": True,
                         "crop_to_text": True},
        "routing": {"classification_auto_accept_threshold": auto_thresh,
                    "classification_needs_review_threshold": review_thresh,
                    "ocr_reject_threshold": ocr_reject},
        "paths": {"model_artifact": "models/m.joblib", "image_inbox": str(tmp_path / "in"),
                  "processed_output": str(tmp_path / "out"), "review_queue": str(tmp_path / "q"),
                  "archive": str(tmp_path / "a")},
        "supported_formats": [".jpg", ".png"],
        "forbidden_model_features": ["ddc_raw", "label_name", "label_override",
                                     "label_level_1", "review_status", "document_id",
                                     "source_work_ids", "split_group_id",
                                     "previous_split_memberships", "prior_holdout_status",
                                     "source_urls", "retrieved_at", "edition_ids", "isbns",
                                     "first_sentence_sources", "description_source_url",
                                     "text_enrichment_sources", "deduplication_status"],
        "valid_labels": [f"{i}00" for i in range(10)],
    }
    p = tmp_path / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(config_data))
    return PipelineConfig(p)


class TestThresholdRegression:
    """Changing thresholds must visibly change routing outcomes."""

    def test_raising_auto_accept_threshold_changes_outcome(self, tmp_path):
        """A sample that was auto_accept becomes needs_review when threshold rises."""
        ocr = _make_ocr(80.0)
        cls = _make_cls(0.88)

        # With default threshold (0.85): should auto_accept
        config_low = _make_config(tmp_path / "low", auto_thresh=0.85)
        result_low = route(ocr, cls, config_low)
        assert result_low.outcome == RoutingOutcome.AUTO_ACCEPT

        # With raised threshold (0.95): same sample now needs_review
        config_high = _make_config(tmp_path / "high", auto_thresh=0.95)
        result_high = route(ocr, cls, config_high)
        assert result_high.outcome == RoutingOutcome.NEEDS_REVIEW

    def test_lowering_ocr_reject_threshold_changes_outcome(self, tmp_path):
        """A sample that was reject becomes needs_review when reject threshold drops."""
        ocr = _make_ocr(35.0)
        cls = _make_cls(0.90)

        # With default reject threshold (40): should reject
        config_strict = _make_config(tmp_path / "strict", ocr_reject=40)
        result_strict = route(ocr, cls, config_strict)
        assert result_strict.outcome == RoutingOutcome.REJECT_RETAKE

        # With lower reject threshold (30): same sample now not rejected
        config_lenient = _make_config(tmp_path / "lenient", ocr_reject=30)
        result_lenient = route(ocr, cls, config_lenient)
        assert result_lenient.outcome != RoutingOutcome.REJECT_RETAKE

    def test_raising_ocr_min_mean_changes_auto_accept(self, tmp_path):
        """Raising ocr_min_mean can prevent auto_accept."""
        ocr = _make_ocr(65.0)
        cls = _make_cls(0.90)

        # With default min_mean (60): auto_accept
        config_low = _make_config(tmp_path / "low", ocr_min_mean=60)
        result_low = route(ocr, cls, config_low)
        assert result_low.outcome == RoutingOutcome.AUTO_ACCEPT

        # With raised min_mean (75): needs_review
        config_high = _make_config(tmp_path / "high", ocr_min_mean=75)
        result_high = route(ocr, cls, config_high)
        assert result_high.outcome == RoutingOutcome.NEEDS_REVIEW

    def test_fixed_batch_changes_distribution(self, tmp_path):
        """A fixed batch of samples produces different outcome distributions
        when thresholds change."""
        fixed_batch = [
            (_make_ocr(90.0), _make_cls(0.95)),  # Strong
            (_make_ocr(70.0), _make_cls(0.80)),  # Medium
            (_make_ocr(50.0), _make_cls(0.60)),  # Weak
            (_make_ocr(30.0), _make_cls(0.40)),  # Very weak
        ]

        # Lenient config
        lenient = _make_config(tmp_path / "len", auto_thresh=0.70, ocr_reject=25, ocr_min_mean=50)
        lenient_outcomes = [route(o, c, lenient).outcome for o, c in fixed_batch]

        # Strict config
        strict = _make_config(tmp_path / "str", auto_thresh=0.95, ocr_reject=55, ocr_min_mean=75)
        strict_outcomes = [route(o, c, strict).outcome for o, c in fixed_batch]

        # Outcomes must be different — proves thresholds are wired in
        assert lenient_outcomes != strict_outcomes
        # Lenient should have more auto_accepts
        len_accepts = sum(1 for o in lenient_outcomes if o == RoutingOutcome.AUTO_ACCEPT)
        str_accepts = sum(1 for o in strict_outcomes if o == RoutingOutcome.AUTO_ACCEPT)
        assert len_accepts >= str_accepts
