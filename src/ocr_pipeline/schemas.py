"""Typed dataclasses for inter-stage pipeline data.

Every stage consumes and produces one of these types. No stage silently
swallows errors into default values — missing data is represented
explicitly (empty string, None, confidence=0), never fabricated.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ── Enums ───────────────────────────────────────────────────────────────

class PipelineStatus(enum.Enum):
    """Status at any pipeline stage."""
    SUCCESS = "success"
    PREPROCESSING_FAILED = "preprocessing_failed"
    OCR_FAILED = "ocr_failed"
    OCR_LOW_CONFIDENCE = "ocr_low_confidence"
    NON_LATIN_SCRIPT = "non_latin_script"
    CLASSIFICATION_FAILED = "classification_failed"


class RoutingOutcome(enum.Enum):
    """Three explicit routing outcomes — no bare pass-through."""
    AUTO_ACCEPT = "auto_accept"
    NEEDS_REVIEW = "needs_review"
    REJECT_RETAKE = "reject_retake"


# ── Stage 1: Ingest ─────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Output of the ingest stage (§2.1)."""
    original_path: Path
    archive_path: Path
    content_hash: str            # SHA-256 of raw image bytes, computed before any processing
    original_filename: str
    upload_timestamp: str        # ISO-8601
    batch_id: str
    session_id: str
    file_format: str             # e.g. ".jpg"


# ── Stage 2: Preprocessing ──────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """Output of the preprocessing stage (§2.2)."""
    derived_image_path: Path     # Separate from original — never mutate the source
    operations_applied: List[str]  # e.g. ["deskew_2.3deg", "contrast_normalize", "crop_to_text"]
    status: PipelineStatus


# ── Stage 3: OCR ────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Pixel-coordinate bounding box for a word or line."""
    left: int
    top: int
    width: int
    height: int


@dataclass
class OCRWord:
    """Single word with confidence and position."""
    text: str
    confidence: float            # 0–100 scale from Tesseract
    bbox: BoundingBox
    line_num: int
    word_num: int


@dataclass
class OCRResult:
    """Output of the OCR stage (§2.3)."""
    words: List[OCRWord]
    full_text: str               # Reconstructed from words
    mean_confidence: float
    min_confidence: float
    detected_script: Optional[str]   # e.g. "Latin", "Arabic", "Han"
    detected_language: Optional[str] # e.g. "eng", "ara"
    status: PipelineStatus
    word_count: int


# ── Stage 4: Field Parsing ──────────────────────────────────────────────

@dataclass
class ParsedField:
    """A single parsed field with its confidence."""
    value: str
    confidence: float            # 0.0–1.0
    source: str                  # e.g. "positional_heuristic", "by_prefix"


@dataclass
class ParsedFields:
    """Output of the field parsing stage (§2.4)."""
    title: ParsedField
    subtitle: ParsedField
    author: ParsedField
    fields_present: Dict[str, bool]  # {"title": True, "subtitle": False, "author": True}


# ── Stage 5: raw_text Assembly ──────────────────────────────────────────

@dataclass
class AssembledText:
    """Output of the assembly stage (§2.5)."""
    raw_text: str                # Same shape as model_text() in etl.py
    text_availability: str       # "title_only" or "title_and_context"
    fields_present: Dict[str, bool]  # Which fields contributed to raw_text
    fields_absent: List[str]     # Fields not available from OCR (e.g. "subject_headings")


# ── Stage 6: Classification ─────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Output of the classification stage (§2.6)."""
    predicted_label: str         # Three-character string: "000"–"900"
    confidence: float            # Probability for the predicted class
    all_probabilities: Dict[str, float]  # All ten class probabilities
    calibrated: bool             # Whether confidence is calibrated (documented if not)
    calibration_note: str        # Explicit note on calibration status


# ── Stage 7: Routing ────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Output of the routing stage (§2.7)."""
    outcome: RoutingOutcome
    ocr_mean_confidence: float
    classification_confidence: float
    thresholds_used: Dict[str, float]  # Named thresholds and their values at decision time
    reason: str                  # Human-readable explanation


# ── Stage 8: Human Review ───────────────────────────────────────────────

@dataclass
class ReviewRecord:
    """Audit record for a human review action (§2.8).

    A correction appends a new record — never overwrites the original
    prediction in place. Matches the repo's 'preserve ddc_raw alongside
    corrected label' pattern.
    """
    pipeline_id: str             # Links to the PipelineResult
    original_ocr_text: str
    original_predicted_label: str
    original_confidence: float
    reviewer_id: str
    review_timestamp: str        # ISO-8601
    corrected_label: Optional[str]  # None if reviewer confirmed the prediction
    correction_reason: str
    action: str                  # "confirmed" or "corrected"


# ── Composite Result ────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full pipeline output — composite of all stages."""
    pipeline_id: str             # Unique identifier for this run
    ingest: Optional[IngestResult] = None
    preprocess: Optional[PreprocessResult] = None
    ocr: Optional[OCRResult] = None
    parsed_fields: Optional[ParsedFields] = None
    assembled: Optional[AssembledText] = None
    classification: Optional[ClassificationResult] = None
    routing: Optional[RoutingDecision] = None
    final_status: PipelineStatus = PipelineStatus.SUCCESS
    error_message: str = ""
    processing_log: List[str] = field(default_factory=list)
