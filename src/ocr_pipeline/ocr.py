"""Stage 3: OCR (§2.3).

Runs Tesseract OCR and captures word-level confidence. Detects
script/language where possible; fails explicitly for unsupported
non-Latin scripts rather than emitting garbage.

OCR text is UNTRUSTED INPUT — never execute it, interpolate it
unsanitized into shell commands, SQL queries, or templates.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

from .config import PipelineConfig
from .exceptions import NonLatinScriptError, OCRFailedError
from .schemas import BoundingBox, OCRResult, OCRWord, PipelineStatus

# Scripts that Tesseract's Latin-only config will mangle.
# If OSD detects one of these and we don't have the matching lang pack,
# we must fail explicitly.
_LATIN_SCRIPTS = {"Latin", "Common", "Inherited", ""}


def _detect_script(image: Image.Image, tesseract_cmd: str | None) -> dict:
    """Use Tesseract OSD to detect script and orientation.

    Returns dict with keys: script, orientation, confidence.
    Returns empty dict if OSD fails (e.g. too little text).
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return {
            "script": osd.get("script", ""),
            "script_confidence": float(osd.get("script_conf", 0)),
            "orientation": osd.get("orientation", 0),
        }
    except pytesseract.TesseractError:
        # OSD fails on very small or low-contrast images — not fatal
        return {}


def run_ocr(
    image_path: str | Path,
    config: PipelineConfig,
) -> OCRResult:
    """Run OCR on a preprocessed image.

    Parameters
    ----------
    image_path : path to the preprocessed image
    config : pipeline configuration

    Returns
    -------
    OCRResult with per-word confidences and detected script

    Raises
    ------
    OCRFailedError
        If Tesseract fails entirely or produces zero words.
    NonLatinScriptError
        If a non-Latin script is detected and cannot be handled.
    """
    path = Path(image_path)
    if config.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    try:
        image = Image.open(path)
    except Exception as exc:
        raise OCRFailedError(str(path), f"Cannot open image for OCR: {exc}")

    # Detect script/language before running full OCR
    osd_info = _detect_script(image, config.tesseract_cmd)
    detected_script = osd_info.get("script", None)
    detected_language = None

    # Check for non-Latin script
    if detected_script and detected_script not in _LATIN_SCRIPTS:
        script_conf = osd_info.get("script_confidence", 0)
        if script_conf > config.osd_min_script_confidence:
            raise NonLatinScriptError(detected_script)

    # Run OCR with word-level confidence (image_to_data gives TSV)
    try:
        data = pytesseract.image_to_data(
            image,
            config=config.tesseract_config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as exc:
        raise OCRFailedError(str(path), f"Tesseract engine error: {exc}")

    # Parse word-level results
    words: List[OCRWord] = []
    n_items = len(data.get("text", []))

    for i in range(n_items):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            # Tesseract uses -1 for non-text blocks
            continue
        words.append(OCRWord(
            text=text,
            confidence=conf,
            bbox=BoundingBox(
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            ),
            line_num=int(data["line_num"][i]),
            word_num=int(data["word_num"][i]),
        ))

    if not words:
        raise OCRFailedError(str(path), "OCR produced no recognizable words")

    # Compute confidence statistics
    confidences = [w.confidence for w in words]
    mean_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)

    # Reconstruct full text preserving line breaks
    lines: dict[int, list[str]] = {}
    for w in words:
        lines.setdefault(w.line_num, []).append(w.text)
    full_text = "\n".join(" ".join(line_words) for _, line_words in sorted(lines.items()))

    # Determine status based on confidence threshold
    if mean_conf < config.ocr_reject_threshold:
        status = PipelineStatus.OCR_LOW_CONFIDENCE
    else:
        status = PipelineStatus.SUCCESS

    return OCRResult(
        words=words,
        full_text=full_text,
        mean_confidence=mean_conf,
        min_confidence=min_conf,
        detected_script=detected_script,
        detected_language=detected_language,
        status=status,
        word_count=len(words),
    )
