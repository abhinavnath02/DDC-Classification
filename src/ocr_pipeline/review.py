"""Stage 8: Human review integration (§2.8).

Persists enough state that a reviewer's confirm/correct action can be
replayed and audited later. A correction NEVER overwrites the original
prediction in place — it appends a new record, matching the repo's
existing 'preserve ddc_raw alongside corrected label' pattern.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .schemas import PipelineResult, ReviewRecord


def _review_queue_path(config: PipelineConfig) -> Path:
    """Return the review queue directory, creating it if needed."""
    path = config.review_queue_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def persist_for_review(
    result: PipelineResult,
    config: PipelineConfig,
) -> Path:
    """Save a pipeline result to the review queue.

    Writes the full pipeline state so a reviewer can see:
    - Original OCR text
    - Original prediction and confidence
    - Routing decision and reason
    - All threshold values at decision time

    Returns the path to the persisted review file.
    """
    queue = _review_queue_path(config)
    filename = f"{result.pipeline_id}.json"
    filepath = queue / filename

    review_data = {
        "pipeline_id": result.pipeline_id,
        "ocr_text": result.ocr.full_text if result.ocr else "",
        "ocr_mean_confidence": result.ocr.mean_confidence if result.ocr else 0,
        "predicted_label": result.classification.predicted_label if result.classification else "",
        "prediction_confidence": result.classification.confidence if result.classification else 0,
        "all_probabilities": result.classification.all_probabilities if result.classification else {},
        "routing_outcome": result.routing.outcome.value if result.routing else "",
        "routing_reason": result.routing.reason if result.routing else "",
        "thresholds_used": result.routing.thresholds_used if result.routing else {},
        "parsed_title": result.parsed_fields.title.value if result.parsed_fields else "",
        "parsed_subtitle": result.parsed_fields.subtitle.value if result.parsed_fields else "",
        "parsed_author": result.parsed_fields.author.value if result.parsed_fields else "",
        "raw_text": result.assembled.raw_text if result.assembled else "",
        "text_availability": result.assembled.text_availability if result.assembled else "",
        "content_hash": result.ingest.content_hash if result.ingest else "",
        "original_filename": result.ingest.original_filename if result.ingest else "",
        "persisted_at": datetime.now(timezone.utc).isoformat(),
        "corrections": [],  # Will be appended by record_correction()
    }

    filepath.write_text(json.dumps(review_data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return filepath


def record_correction(
    pipeline_id: str,
    reviewer_id: str,
    corrected_label: Optional[str],
    correction_reason: str,
    config: PipelineConfig,
    action: str = "corrected",
) -> ReviewRecord:
    """Record a reviewer's correction or confirmation.

    Appends a new record to the existing review file — never overwrites
    the original prediction in place.

    Parameters
    ----------
    pipeline_id : links to the PipelineResult
    reviewer_id : who is making the correction
    corrected_label : the corrected label, or None if confirming
    correction_reason : explanation for the correction
    config : pipeline configuration
    action : 'confirmed' or 'corrected'

    Returns
    -------
    ReviewRecord for audit
    """
    queue = _review_queue_path(config)
    filepath = queue / f"{pipeline_id}.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Review file not found for pipeline_id '{pipeline_id}'. "
            f"Expected at: {filepath}"
        )

    review_data = json.loads(filepath.read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone.utc).isoformat()

    record = ReviewRecord(
        pipeline_id=pipeline_id,
        original_ocr_text=review_data.get("ocr_text", ""),
        original_predicted_label=review_data.get("predicted_label", ""),
        original_confidence=review_data.get("prediction_confidence", 0),
        reviewer_id=reviewer_id,
        review_timestamp=timestamp,
        corrected_label=corrected_label,
        correction_reason=correction_reason,
        action=action,
    )

    # Append — never overwrite
    correction_entry = {
        "reviewer_id": reviewer_id,
        "timestamp": timestamp,
        "action": action,
        "original_label": review_data.get("predicted_label", ""),
        "corrected_label": corrected_label,
        "reason": correction_reason,
    }
    review_data.setdefault("corrections", []).append(correction_entry)

    filepath.write_text(json.dumps(review_data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return record
