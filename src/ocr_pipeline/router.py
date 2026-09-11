"""Stage 7: Routing / thresholds (§2.7).

Three explicit outcomes only: auto_accept, needs_review, reject_retake.
Thresholds are named constants loaded from pipeline_config.yaml — never
scattered magic numbers.

Every routing decision is logged with the OCR confidence, classification
confidence, and threshold values used, so a threshold change later can
be audited against historical decisions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .config import PipelineConfig
from .schemas import (
    ClassificationResult,
    OCRResult,
    PipelineStatus,
    RoutingDecision,
    RoutingOutcome,
)

logger = logging.getLogger(__name__)


def route(
    ocr_result: OCRResult,
    classification: ClassificationResult,
    config: PipelineConfig,
) -> RoutingDecision:
    """Determine routing for a pipeline result.

    Decision logic (evaluated in order):
    1. If OCR mean confidence < ocr_reject_threshold → reject_retake
    2. If classification confidence >= auto_accept_threshold
       AND OCR mean confidence >= min_mean_confidence → auto_accept
    3. Everything else → needs_review

    Parameters
    ----------
    ocr_result : output from the OCR stage
    classification : output from the classification stage
    config : pipeline configuration

    Returns
    -------
    RoutingDecision with outcome, all confidence values, and thresholds used
    """
    thresholds = config.routing_thresholds_dict()
    ocr_conf = ocr_result.mean_confidence
    cls_conf = classification.confidence

    # Rule 1: OCR too unreliable → reject
    if ocr_conf < config.ocr_reject_threshold:
        outcome = RoutingOutcome.REJECT_RETAKE
        reason = (
            f"OCR mean confidence ({ocr_conf:.1f}) is below "
            f"reject threshold ({config.ocr_reject_threshold}). "
            "Image quality is too low for reliable classification."
        )

    # Rule 2: Both OCR and classification confident → auto-accept
    elif (cls_conf >= config.classification_auto_accept_threshold
          and ocr_conf >= config.ocr_min_mean_confidence):
        outcome = RoutingOutcome.AUTO_ACCEPT
        reason = (
            f"Classification confidence ({cls_conf:.3f}) >= "
            f"auto_accept threshold ({config.classification_auto_accept_threshold}) "
            f"and OCR mean confidence ({ocr_conf:.1f}) >= "
            f"min_mean_confidence ({config.ocr_min_mean_confidence})."
        )

    # Rule 3: Everything else → needs review
    else:
        outcome = RoutingOutcome.NEEDS_REVIEW
        parts = []
        if cls_conf < config.classification_auto_accept_threshold:
            parts.append(
                f"Classification confidence ({cls_conf:.3f}) < "
                f"auto_accept threshold ({config.classification_auto_accept_threshold})"
            )
        if ocr_conf < config.ocr_min_mean_confidence:
            parts.append(
                f"OCR mean confidence ({ocr_conf:.1f}) < "
                f"min_mean_confidence ({config.ocr_min_mean_confidence})"
            )
        reason = "Needs human review: " + "; ".join(parts) if parts else "Needs human review."

    decision = RoutingDecision(
        outcome=outcome,
        ocr_mean_confidence=round(ocr_conf, 2),
        classification_confidence=round(cls_conf, 6),
        thresholds_used=thresholds,
        reason=reason,
    )

    # Log every routing decision for audit trail
    logger.info(
        "Routing decision: %s | OCR_conf=%.1f | cls_conf=%.4f | "
        "predicted=%s | thresholds=%s | reason=%s",
        outcome.value,
        ocr_conf,
        cls_conf,
        classification.predicted_label,
        json.dumps(thresholds),
        reason,
    )

    return decision
