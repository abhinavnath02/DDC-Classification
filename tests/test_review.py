"""Tests for Stage 8: Human review integration (§2.8)."""
import json

import pytest

from src.ocr_pipeline.review import persist_for_review, record_correction
from src.ocr_pipeline.schemas import (
    AssembledText,
    ClassificationResult,
    IngestResult,
    OCRResult,
    OCRWord,
    BoundingBox,
    PipelineResult,
    PipelineStatus,
    RoutingDecision,
    RoutingOutcome,
    ParsedField,
    ParsedFields,
)
from pathlib import Path


def _make_pipeline_result() -> PipelineResult:
    """Helper: create a complete PipelineResult for review tests."""
    return PipelineResult(
        pipeline_id="test-review-001",
        ingest=IngestResult(
            original_path=Path("/fake/path.jpg"),
            archive_path=Path("/fake/archive/path.jpg"),
            content_hash="abc123def456",
            original_filename="test_book.jpg",
            upload_timestamp="2024-01-01T00:00:00Z",
            batch_id="batch_test",
            session_id="session_test",
            file_format=".jpg",
        ),
        ocr=OCRResult(
            words=[OCRWord(text="test", confidence=90.0,
                          bbox=BoundingBox(0,0,50,20), line_num=1, word_num=1)],
            full_text="The Great Gatsby",
            mean_confidence=90.0,
            min_confidence=90.0,
            detected_script="Latin",
            detected_language="eng",
            status=PipelineStatus.SUCCESS,
            word_count=3,
        ),
        parsed_fields=ParsedFields(
            title=ParsedField(value="The Great Gatsby", confidence=0.9, source="test"),
            subtitle=ParsedField(value="", confidence=0.0, source="not_found"),
            author=ParsedField(value="F. Scott Fitzgerald", confidence=0.8, source="test"),
            fields_present={"title": True, "subtitle": False, "author": True},
        ),
        assembled=AssembledText(
            raw_text="The Great Gatsby",
            text_availability="title_only",
            fields_present={"title": True, "subtitle": False,
                          "first_sentence": False, "subject_headings": False},
            fields_absent=["subtitle", "first_sentence", "subject_headings"],
        ),
        classification=ClassificationResult(
            predicted_label="800",
            confidence=0.87,
            all_probabilities={f"{i}00": 0.1 for i in range(10)},
            calibrated=False,
            calibration_note="test",
        ),
        routing=RoutingDecision(
            outcome=RoutingOutcome.NEEDS_REVIEW,
            ocr_mean_confidence=90.0,
            classification_confidence=0.87,
            thresholds_used={"auto_accept": 0.85},
            reason="Test needs review",
        ),
    )


class TestReview:
    """Human review integration tests."""

    def test_persist_creates_file(self, sample_pipeline_config):
        """Persisting for review creates a JSON file."""
        result = _make_pipeline_result()
        path = persist_for_review(result, sample_pipeline_config)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["pipeline_id"] == "test-review-001"

    def test_persist_includes_original_prediction(self, sample_pipeline_config):
        """Persisted review file includes the original prediction."""
        result = _make_pipeline_result()
        path = persist_for_review(result, sample_pipeline_config)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["predicted_label"] == "800"
        assert data["prediction_confidence"] == 0.87

    def test_persist_includes_ocr_text(self, sample_pipeline_config):
        """Persisted review file includes the original OCR text."""
        result = _make_pipeline_result()
        path = persist_for_review(result, sample_pipeline_config)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["ocr_text"] == "The Great Gatsby"

    def test_correction_appends(self, sample_pipeline_config):
        """A correction appends — never overwrites the original."""
        result = _make_pipeline_result()
        persist_for_review(result, sample_pipeline_config)

        record = record_correction(
            pipeline_id="test-review-001",
            reviewer_id="reviewer_A",
            corrected_label="700",
            correction_reason="Book is about art, not literature",
            config=sample_pipeline_config,
            action="corrected",
        )

        # Read the file back
        path = sample_pipeline_config.review_queue_path / "test-review-001.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        # Original prediction preserved
        assert data["predicted_label"] == "800"
        # Correction appended
        assert len(data["corrections"]) == 1
        assert data["corrections"][0]["corrected_label"] == "700"
        assert data["corrections"][0]["original_label"] == "800"
        assert data["corrections"][0]["reviewer_id"] == "reviewer_A"

    def test_multiple_corrections_append(self, sample_pipeline_config):
        """Multiple corrections are all appended."""
        result = _make_pipeline_result()
        persist_for_review(result, sample_pipeline_config)

        record_correction("test-review-001", "reviewer_A", "700",
                         "First review", sample_pipeline_config)
        record_correction("test-review-001", "reviewer_B", "800",
                         "Second review disagrees", sample_pipeline_config)

        path = sample_pipeline_config.review_queue_path / "test-review-001.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["corrections"]) == 2
        assert data["corrections"][0]["reviewer_id"] == "reviewer_A"
        assert data["corrections"][1]["reviewer_id"] == "reviewer_B"

    def test_confirmation_recorded(self, sample_pipeline_config):
        """A confirmation (no correction) is also recorded."""
        result = _make_pipeline_result()
        persist_for_review(result, sample_pipeline_config)

        record = record_correction(
            pipeline_id="test-review-001",
            reviewer_id="reviewer_C",
            corrected_label=None,
            correction_reason="Prediction looks correct",
            config=sample_pipeline_config,
            action="confirmed",
        )
        assert record.action == "confirmed"
        assert record.corrected_label is None

    def test_missing_review_file_raises(self, sample_pipeline_config):
        """Correcting a non-existent review file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            record_correction("nonexistent-id", "reviewer", "500",
                            "reason", sample_pipeline_config)
