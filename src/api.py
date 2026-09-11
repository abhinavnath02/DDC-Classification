"""FastAPI backend for the DDC Classification OCR pipeline.

Uses Gemini Vision API as the primary OCR/understanding engine for book page images.
Implements ensemble voting between Gemini's DDC prediction and the TF-IDF model
for more accurate classifications.
Falls back to the Tesseract-based pipeline if GEMINI_API_KEY is not set.
"""
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from repo root or parent directories
_REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in [_REPO_ROOT / ".env", _REPO_ROOT.parent / ".env"]:
    if candidate.exists():
        load_dotenv(candidate)
        break

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.ocr_pipeline.config import PipelineConfig
from src.ocr_pipeline.classify import classify_text, load_model
from src.ocr_pipeline.gemini_ocr import extract_with_gemini

app = FastAPI(title="DDC Classification OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config and model at startup
CONFIG = PipelineConfig(_REPO_ROOT / "pipeline_config.yaml")
MODEL = load_model(CONFIG)

# DDC label names for display
LABEL_NAMES = {
    "000": "Computing, information and general works",
    "100": "Philosophy and psychology",
    "200": "Religion",
    "300": "Social sciences",
    "400": "Language",
    "500": "Science",
    "600": "Technology and applied subjects",
    "700": "Arts and recreation",
    "800": "Literature",
    "900": "History and geography",
}


def _save_upload(upload: UploadFile, upload_dir: Path) -> Path:
    """Save an UploadFile to disk and return the path."""
    dest = upload_dir / upload.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _ensemble_classify(
    raw_text: str,
    gemini_class: str,
    gemini_confidence: float,
    model,
    config: PipelineConfig,
) -> dict:
    """Ensemble voting between Gemini DDC prediction and TF-IDF model.

    Strategy:
    - Always run the TF-IDF model on the extracted text.
    - If both Gemini and TF-IDF agree on the class, boost confidence
      by averaging both scores — high agreement = high reliability.
    - If they disagree, pick the one with higher confidence but
      flag as needs_review for human verification.

    Returns dict with predicted_class, confidence, all_probs, routing,
    and ensemble metadata.
    """
    valid_classes = {"000", "100", "200", "300", "400", "500", "600", "700", "800", "900"}

    # Always run the TF-IDF model
    tfidf_result = classify_text(raw_text, config, model=model)
    tfidf_class = tfidf_result.predicted_label
    tfidf_confidence = tfidf_result.confidence
    tfidf_probs = tfidf_result.all_probabilities

    has_gemini = gemini_class in valid_classes and gemini_confidence > 0

    if has_gemini:
        if gemini_class == tfidf_class:
            # Both agree — average confidence for a boosted score
            predicted_class = gemini_class
            confidence = round((gemini_confidence + tfidf_confidence) / 2, 4)
            ensemble_strategy = "agreement"
        else:
            # Disagreement — use higher confidence source, flag for review
            if gemini_confidence >= tfidf_confidence:
                predicted_class = gemini_class
                confidence = round(gemini_confidence * 0.85, 4)  # Discount for disagreement
            else:
                predicted_class = tfidf_class
                confidence = round(tfidf_confidence * 0.85, 4)
            ensemble_strategy = "disagreement"
    else:
        # No valid Gemini prediction — TF-IDF only
        predicted_class = tfidf_class
        confidence = tfidf_confidence
        ensemble_strategy = "tfidf_only"

    # Routing decision
    if ensemble_strategy == "disagreement":
        # Force review when models disagree
        routing = "needs_review"
    elif confidence >= config.classification_auto_accept_threshold:
        routing = "auto_accept"
    elif confidence >= config.classification_needs_review_threshold:
        routing = "needs_review"
    else:
        routing = "reject_retake"

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": tfidf_probs,
        "routing": routing,
        "ensemble_strategy": ensemble_strategy,
        "gemini_class": gemini_class if has_gemini else None,
        "gemini_confidence": gemini_confidence if has_gemini else None,
        "tfidf_class": tfidf_class,
        "tfidf_confidence": round(tfidf_confidence, 4),
    }


@app.post("/api/classify")
async def classify_book(
    front_page: UploadFile = File(...),
    summary_page: Optional[UploadFile] = File(None),
    index_page: Optional[UploadFile] = File(None),
):
    upload_dir = _REPO_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    front_path = _save_upload(front_page, upload_dir)

    # ── Step 1: Extract text from all pages using Gemini Vision ──────
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return JSONResponse(
            status_code=500,
            content={
                "error": "GEMINI_API_KEY environment variable is not set. "
                         "Please set it before running the server.",
            },
        )

    try:
        # Front page — full structured extraction
        front_result = extract_with_gemini(front_path, api_key=gemini_key)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Gemini Vision failed on front page: {exc}"},
        )

    # Supplementary pages — extract only raw text
    supplementary_texts = []

    if summary_page and summary_page.filename:
        sum_path = _save_upload(summary_page, upload_dir)
        try:
            sum_result = extract_with_gemini(sum_path, api_key=gemini_key)
            text = sum_result.get("full_extracted_text", "")
            if text:
                supplementary_texts.append(text)
        except Exception as exc:
            print(f"Gemini failed on summary page (non-fatal): {exc}")

    if index_page and index_page.filename:
        idx_path = _save_upload(index_page, upload_dir)
        try:
            idx_result = extract_with_gemini(idx_path, api_key=gemini_key)
            text = idx_result.get("full_extracted_text", "")
            if text:
                supplementary_texts.append(text)
        except Exception as exc:
            print(f"Gemini failed on index page (non-fatal): {exc}")

    # ── Step 2: Build raw_text from Gemini output ────────────────────
    title = front_result.get("title", "")
    subtitle = front_result.get("subtitle", "")
    author = front_result.get("author", "")
    description = front_result.get("description", "")
    full_text = front_result.get("full_extracted_text", "")

    # Assemble raw_text in the same shape as model_text() in etl.py
    parts = [title]
    if subtitle and subtitle.lower() not in title.lower():
        parts.append(subtitle)
    if description:
        parts.append(description)
    if supplementary_texts:
        parts.extend(supplementary_texts)

    raw_text = "\n".join(p for p in parts if p)
    if not raw_text.strip() and full_text:
        raw_text = full_text

    # ── Step 3: Ensemble Classify ────────────────────────────────────
    gemini_class = front_result.get("ddc_class", "").strip()
    gemini_confidence = float(front_result.get("ddc_confidence", 0.0))

    try:
        ensemble = _ensemble_classify(
            raw_text=raw_text,
            gemini_class=gemini_class,
            gemini_confidence=gemini_confidence,
            model=MODEL,
            config=CONFIG,
        )

        predicted_class = ensemble["predicted_class"]
        confidence = ensemble["confidence"]
        all_probs = ensemble["all_probabilities"]
        routing = ensemble["routing"]

    except Exception as exc:
        traceback.print_exc()
        predicted_class = None
        confidence = None
        all_probs = {}
        routing = "classification_failed"
        ensemble = {}

    # ── Step 4: Build response ───────────────────────────────────────
    response = {
        "status": "success" if predicted_class else "classification_failed",
        "error": None if predicted_class else "Classification failed",
        "predicted_class": predicted_class,
        "predicted_class_name": LABEL_NAMES.get(predicted_class, "Unknown") if predicted_class else None,
        "confidence": confidence,
        "routing_outcome": routing,
        "raw_text": raw_text,
        "fields": {
            "title": title,
            "subtitle": subtitle,
            "author": author,
            "description": description,
        },
        "all_probabilities": all_probs,
        "ocr_engine": "gemini_vision",
        "gemini_confidence": front_result.get("confidence", None),
        # Ensemble metadata for transparency
        "ensemble": {
            "strategy": ensemble.get("ensemble_strategy", "unknown"),
            "gemini_class": ensemble.get("gemini_class"),
            "gemini_ddc_confidence": ensemble.get("gemini_confidence"),
            "tfidf_class": ensemble.get("tfidf_class"),
            "tfidf_confidence": ensemble.get("tfidf_confidence"),
        } if ensemble else None,
    }

    return JSONResponse(content=response)


# Mount the frontend
frontend_dir = _REPO_ROOT / "src" / "frontend"
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
