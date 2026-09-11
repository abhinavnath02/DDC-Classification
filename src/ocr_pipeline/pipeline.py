"""Pipeline orchestrator — chains all stages.

Ingest → Preprocess → OCR → Field Parse → Assemble → Classify → Route

Each stage can fail independently with typed errors. The orchestrator
catches stage-specific exceptions and records them in the PipelineResult
rather than crashing the entire pipeline.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from .assemble import assemble_raw_text
from .classify import classify_text, load_model
from .config import PipelineConfig
from .exceptions import (
    NonLatinScriptError,
    OCRFailedError,
    OCRPipelineError,
    PreprocessingFailedError,
)
from .field_parser import parse_fields
from .ingest import ingest_image
from .ocr import run_ocr
from .preprocess import preprocess_image
from .review import persist_for_review
from .router import route
from .schemas import PipelineResult, PipelineStatus, RoutingOutcome

logger = logging.getLogger(__name__)


def run_pipeline(
    image_path: str | Path,
    config: PipelineConfig | None = None,
    batch_id: str | None = None,
    session_id: str | None = None,
    model: Optional[Any] = None,
    supplementary_texts: str | None = None,
) -> PipelineResult:
    """Run the full OCR → classification → routing pipeline.

    Parameters
    ----------
    image_path : path to the source image
    config : pipeline configuration (loaded from default if None)
    batch_id : optional batch identifier
    session_id : optional session identifier
    model : pre-loaded model (optional — loaded once, reused across calls)
    supplementary_texts : optional string of text from summary/index pages to append to raw_text

    Returns
    -------
    PipelineResult with all stage outputs and final routing decision
    """
    if config is None:
        config = PipelineConfig()

    pipeline_id = str(uuid.uuid4())
    result = PipelineResult(pipeline_id=pipeline_id)

    try:
        # ── Stage 1: Ingest ─────────────────────────────────────────────
        result.processing_log.append("Starting ingest")
        result.ingest = ingest_image(image_path, config, batch_id, session_id)
        result.processing_log.append(
            f"Ingested: {result.ingest.original_filename} "
            f"(hash={result.ingest.content_hash[:12]})"
        )

        # ── Stage 2: Preprocess ────────────────────────────────────────
        result.processing_log.append("Starting preprocessing")
        output_dir = config.processed_output / result.ingest.batch_id
        result.preprocess = preprocess_image(
            result.ingest.archive_path,
            output_dir,
            config,
            image_id=result.ingest.content_hash[:12],
        )
        result.processing_log.append(
            f"Preprocessed: {result.preprocess.operations_applied}"
        )

        # ── Stage 3: OCR ───────────────────────────────────────────────
        result.processing_log.append("Starting OCR")
        result.ocr = run_ocr(result.preprocess.derived_image_path, config)
        result.processing_log.append(
            f"OCR: {result.ocr.word_count} words, "
            f"mean_conf={result.ocr.mean_confidence:.1f}"
        )

        # Check OCR status — if low confidence, route to reject
        if result.ocr.status == PipelineStatus.OCR_LOW_CONFIDENCE:
            result.final_status = PipelineStatus.OCR_LOW_CONFIDENCE
            result.processing_log.append(
                "OCR confidence too low — will route to reject_retake"
            )
            # Still continue to get a prediction for the review record

        # ── Stage 4: Field parsing ─────────────────────────────────────
        result.processing_log.append("Starting field parsing")
        result.parsed_fields = parse_fields(result.ocr)
        result.processing_log.append(
            f"Parsed: title='{result.parsed_fields.title.value[:50]}' "
            f"(conf={result.parsed_fields.title.confidence:.2f}), "
            f"subtitle='{result.parsed_fields.subtitle.value[:30]}', "
            f"author='{result.parsed_fields.author.value[:30]}'"
        )

        # ── Stage 5: raw_text assembly ─────────────────────────────────
        result.processing_log.append("Starting raw_text assembly")
        result.assembled = assemble_raw_text(result.parsed_fields)
        if supplementary_texts:
            result.assembled.raw_text += f"\n{supplementary_texts}"
        result.processing_log.append(
            f"Assembled: availability={result.assembled.text_availability}, "
            f"len={len(result.assembled.raw_text)}"
        )

        # ── Stage 6: Classification ────────────────────────────────────
        result.processing_log.append("Starting classification")
        if model is None:
            model = load_model(config)
        result.classification = classify_text(
            result.assembled.raw_text, config, model=model
        )
        result.processing_log.append(
            f"Classified: {result.classification.predicted_label} "
            f"(conf={result.classification.confidence:.4f})"
        )

        # ── Stage 7: Routing ───────────────────────────────────────────
        result.processing_log.append("Starting routing")
        result.routing = route(result.ocr, result.classification, config)
        result.processing_log.append(
            f"Routed: {result.routing.outcome.value} — {result.routing.reason}"
        )

        # Set final status
        if result.final_status == PipelineStatus.SUCCESS:
            result.final_status = PipelineStatus.SUCCESS

        # ── Persist for review if needed ───────────────────────────────
        if result.routing.outcome in (RoutingOutcome.NEEDS_REVIEW, RoutingOutcome.REJECT_RETAKE):
            review_path = persist_for_review(result, config)
            result.processing_log.append(
                f"Persisted for review: {review_path}"
            )

    except PreprocessingFailedError as exc:
        result.final_status = PipelineStatus.PREPROCESSING_FAILED
        result.error_message = str(exc)
        result.processing_log.append(f"FAILED at preprocessing: {exc}")
        logger.warning("Pipeline %s: preprocessing failed — %s", pipeline_id, exc)

    except NonLatinScriptError as exc:
        result.final_status = PipelineStatus.NON_LATIN_SCRIPT
        result.error_message = str(exc)
        result.processing_log.append(f"FAILED at OCR: {exc}")
        logger.warning("Pipeline %s: non-Latin script — %s", pipeline_id, exc)

    except OCRFailedError as exc:
        result.final_status = PipelineStatus.OCR_FAILED
        result.error_message = str(exc)
        result.processing_log.append(f"FAILED at OCR: {exc}")
        logger.warning("Pipeline %s: OCR failed — %s", pipeline_id, exc)

    except OCRPipelineError as exc:
        result.final_status = PipelineStatus.CLASSIFICATION_FAILED
        result.error_message = str(exc)
        result.processing_log.append(f"FAILED: {exc}")
        logger.error("Pipeline %s: error — %s", pipeline_id, exc)

    except Exception as exc:
        result.final_status = PipelineStatus.CLASSIFICATION_FAILED
        result.error_message = f"Unexpected error: {exc}"
        result.processing_log.append(f"UNEXPECTED FAILURE: {exc}")
        logger.exception("Pipeline %s: unexpected error", pipeline_id)

    return result
