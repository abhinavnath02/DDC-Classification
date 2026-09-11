"""Demo script for running the DDC Capstone OCR Classification Pipeline.

Usage:
    python demo.py                       # Runs with a synthetic sample image
    python demo.py path/to/book_cover.jpg # Runs with a specific image
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ocr_pipeline.config import PipelineConfig
from src.ocr_pipeline.pipeline import run_pipeline
from src.ocr_pipeline.schemas import (
    BoundingBox,
    OCRResult,
    OCRWord,
    ParsedField,
    ParsedFields,
    PipelineResult,
    PipelineStatus,
    RoutingOutcome,
)
from src.ocr_pipeline.assemble import assemble_raw_text
from src.ocr_pipeline.classify import classify_text, load_model
from src.ocr_pipeline.router import route


def create_sample_image(output_path: Path) -> Path:
    """Create a sample book title page image."""
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("arial.ttf", 44)
        font_medium = ImageFont.truetype("arial.ttf", 26)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except (IOError, OSError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    draw.text((120, 80), "The Intelligent Investor", fill=(0, 0, 0), font=font_large)
    draw.text((140, 180), "The Definitive Book on Value Investing", fill=(50, 50, 50), font=font_medium)
    draw.text((220, 280), "by Benjamin Graham", fill=(80, 80, 80), font=font_small)

    img.save(str(output_path), format="JPEG", quality=95)
    print(f"[+] Created synthetic sample book page: {output_path}")
    return output_path


def check_tesseract_available(config: PipelineConfig) -> bool:
    """Check if Tesseract binary is available on PATH or configured path."""
    import pytesseract

    if config.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    tesseract_executable = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(tesseract_executable) or os.path.exists(tesseract_executable):
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    if shutil.which("tesseract"):
        return True

    return False


def print_result_summary(result: PipelineResult) -> None:
    """Print formatted output of pipeline execution."""
    print("\n" + "=" * 65)
    print("           OCR CLASSIFICATION PIPELINE RESULT")
    print("=" * 65)
    print(f"Pipeline ID   : {result.pipeline_id}")
    print(f"Final Status  : {result.final_status.value}")

    if result.ingest:
        print(f"Image File    : {result.ingest.original_filename}")
        print(f"Content Hash  : {result.ingest.content_hash[:16]}...")

    if result.ocr:
        print(f"OCR Words     : {result.ocr.word_count}")
        print(f"Mean Conf.    : {result.ocr.mean_confidence:.2f}%")
        print(f"Detected Script: {result.ocr.detected_script}")

    if result.parsed_fields:
        pf = result.parsed_fields
        print("\n--- Parsed Candidate Fields ---")
        print(f"Title         : '{pf.title.value}' (conf: {pf.title.confidence:.2f})")
        print(f"Subtitle      : '{pf.subtitle.value}' (conf: {pf.subtitle.confidence:.2f})")
        print(f"Author        : '{pf.author.value}' (conf: {pf.author.confidence:.2f})")

    if result.assembled:
        print("\n--- Assembled raw_text (ETL contract shape) ---")
        print(f"Availability  : {result.assembled.text_availability}")
        print(f"Length        : {len(result.assembled.raw_text)} chars")
        print(f"Preview       : {repr(result.assembled.raw_text[:120])}")

    if result.classification:
        cl = result.classification
        print("\n--- Classification Output ---")
        print(f"Predicted DDC : Class {cl.predicted_label}")
        print(f"Confidence    : {cl.confidence:.4f}")
        if cl.all_probabilities:
            sorted_probs = sorted(cl.all_probabilities.items(), key=lambda x: x[1], reverse=True)
            top_str = ", ".join(f"{k}: {v:.3f}" for k, v in sorted_probs[:3])
            print(f"Top Probs     : {top_str}")

    if result.routing:
        rt = result.routing
        print("\n--- Routing Decision ---")
        print(f"Outcome       : {rt.outcome.value.upper()}")
        print(f"Reason        : {rt.reason}")

    if result.error_message:
        print(f"\n[!] Error Message: {result.error_message}")

    print("\n--- Execution Log ---")
    for log in result.processing_log:
        print(f"  • {log}")
    print("=" * 65 + "\n")


def run_mock_demo(image_path: Path, config: PipelineConfig) -> None:
    """Run a mock pipeline demonstration when Tesseract binary is missing."""
    print("\n" + "!" * 65)
    print(" NOTICE: Tesseract OCR binary not found on Windows PATH.")
    print(" To run real optical character recognition, install Tesseract:")
    print("   • Download installer: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   • Or run: winget install UB-Mannheim.TesseractOCR")
    print(" Running downstream simulation (Parsing -> Classification -> Routing)...")
    print("!" * 65 + "\n")

    # Create mock OCR output for 'The Intelligent Investor'
    words = [
        OCRWord(text="The", confidence=95.0, bbox=BoundingBox(120, 80, 50, 30), line_num=1, word_num=1),
        OCRWord(text="Intelligent", confidence=94.0, bbox=BoundingBox(180, 80, 150, 30), line_num=1, word_num=2),
        OCRWord(text="Investor", confidence=96.0, bbox=BoundingBox(340, 80, 120, 30), line_num=1, word_num=3),
        OCRWord(text="The", confidence=90.0, bbox=BoundingBox(140, 180, 40, 20), line_num=2, word_num=1),
        OCRWord(text="Definitive", confidence=91.0, bbox=BoundingBox(190, 180, 110, 20), line_num=2, word_num=2),
        OCRWord(text="Book", confidence=93.0, bbox=BoundingBox(310, 180, 60, 20), line_num=2, word_num=3),
        OCRWord(text="on", confidence=92.0, bbox=BoundingBox(380, 180, 30, 20), line_num=2, word_num=4),
        OCRWord(text="Value", confidence=94.0, bbox=BoundingBox(420, 180, 60, 20), line_num=2, word_num=5),
        OCRWord(text="Investing", confidence=95.0, bbox=BoundingBox(490, 180, 90, 20), line_num=2, word_num=6),
        OCRWord(text="by", confidence=88.0, bbox=BoundingBox(220, 280, 30, 18), line_num=3, word_num=1),
        OCRWord(text="Benjamin", confidence=92.0, bbox=BoundingBox(260, 280, 90, 18), line_num=3, word_num=2),
        OCRWord(text="Graham", confidence=93.0, bbox=BoundingBox(360, 280, 80, 18), line_num=3, word_num=3),
    ]

    ocr_res = OCRResult(
        words=words,
        full_text="The Intelligent Investor\nThe Definitive Book on Value Investing\nby Benjamin Graham",
        mean_confidence=93.2,
        min_confidence=88.0,
        detected_script="Latin",
        detected_language="eng",
        status=PipelineStatus.SUCCESS,
        word_count=len(words),
    )

    from src.ocr_pipeline.ingest import ingest_image
    from src.ocr_pipeline.field_parser import parse_fields

    result = PipelineResult(pipeline_id="demo-simulation-id")
    result.ingest = ingest_image(image_path, config)
    result.ocr = ocr_res
    result.parsed_fields = parse_fields(ocr_res)
    result.assembled = assemble_raw_text(result.parsed_fields)

    model = load_model(config)
    result.classification = classify_text(result.assembled.raw_text, config, model=model)
    result.routing = route(result.ocr, result.classification, config)

    result.processing_log = [
        f"Ingested: {result.ingest.original_filename}",
        "[SIMULATED] OCR: 12 words extracted (mean_conf=93.2%)",
        f"Parsed: title='{result.parsed_fields.title.value}'",
        f"Assembled raw_text length={len(result.assembled.raw_text)}",
        f"Classified: DDC Class {result.classification.predicted_label}",
        f"Routed: {result.routing.outcome.value} — {result.routing.reason}",
    ]

    print_result_summary(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DDC OCR Pipeline on a book page image.")
    parser.add_argument("image_path", nargs="?", help="Path to image file (optional)")
    args = parser.parse_args()

    config = PipelineConfig(_REPO_ROOT / "pipeline_config.yaml")

    if args.image_path:
        img_path = Path(args.image_path)
        if not img_path.exists():
            print(f"Error: File not found: {img_path}")
            sys.exit(1)
    else:
        sample_dir = _REPO_ROOT / "data" / "demo"
        sample_dir.mkdir(parents=True, exist_ok=True)
        img_path = create_sample_image(sample_dir / "sample_book_page.jpg")

    if check_tesseract_available(config):
        print("[+] Tesseract OCR detected. Running full pipeline end-to-end...")
        result = run_pipeline(img_path, config=config)
        print_result_summary(result)
    else:
        run_mock_demo(img_path, config)


if __name__ == "__main__":
    main()
