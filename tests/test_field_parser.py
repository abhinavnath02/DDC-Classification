"""Tests for Stage 4: Field parsing (§2.4)."""
import pytest

from src.ocr_pipeline.field_parser import parse_fields
from src.ocr_pipeline.schemas import (
    BoundingBox,
    OCRResult,
    OCRWord,
    PipelineStatus,
)


def _make_ocr_result(lines: list[tuple[str, float, int, int]]) -> OCRResult:
    """Helper: build an OCRResult from (text, confidence, top, height) tuples.

    Each tuple represents one line of words.
    """
    words = []
    for line_num, (text, conf, top, height) in enumerate(lines, start=1):
        for word_num, word_text in enumerate(text.split(), start=1):
            words.append(OCRWord(
                text=word_text,
                confidence=conf,
                bbox=BoundingBox(left=50, top=top, width=len(word_text) * 15, height=height),
                line_num=line_num,
                word_num=word_num,
            ))

    full_text = "\n".join(text for text, _, _, _ in lines)
    confs = [w.confidence for w in words]

    return OCRResult(
        words=words,
        full_text=full_text,
        mean_confidence=sum(confs) / len(confs) if confs else 0,
        min_confidence=min(confs) if confs else 0,
        detected_script="Latin",
        detected_language="eng",
        status=PipelineStatus.SUCCESS,
        word_count=len(words),
    )


class TestFieldParser:
    """Field parser tests."""

    def test_title_extracted(self):
        """The largest text block becomes the title."""
        ocr = _make_ocr_result([
            ("The Great Gatsby", 95.0, 80, 48),
            ("A Novel", 90.0, 180, 28),
            ("by F. Scott Fitzgerald", 88.0, 280, 20),
        ])
        result = parse_fields(ocr)
        assert "Gatsby" in result.title.value or "Great" in result.title.value
        assert result.title.confidence > 0
        assert result.fields_present["title"] is True

    def test_author_by_prefix(self):
        """Author is detected via 'by' prefix."""
        ocr = _make_ocr_result([
            ("Clean Code", 95.0, 80, 48),
            ("by Robert C. Martin", 90.0, 200, 20),
        ])
        result = parse_fields(ocr)
        assert "Robert" in result.author.value or "Martin" in result.author.value
        assert result.author.confidence > 0
        assert result.fields_present["author"] is True

    def test_missing_subtitle_stays_missing(self):
        """If no subtitle signal, subtitle is empty — never fabricated."""
        ocr = _make_ocr_result([
            ("Moby Dick", 95.0, 150, 48),
        ])
        result = parse_fields(ocr)
        assert result.subtitle.value == ""
        assert result.subtitle.confidence == 0.0
        assert result.fields_present["subtitle"] is False

    def test_publisher_not_mistaken_for_title(self):
        """A publisher name is not identified as the title."""
        ocr = _make_ocr_result([
            ("Penguin Books", 92.0, 50, 30),
            ("War and Peace", 94.0, 120, 45),
            ("by Leo Tolstoy", 90.0, 220, 20),
        ])
        result = parse_fields(ocr)
        # Title should be "War and Peace", not "Penguin Books"
        assert "War" in result.title.value or "Peace" in result.title.value

    def test_per_field_confidence(self):
        """Each field has a confidence score, not just parse-or-fail."""
        ocr = _make_ocr_result([
            ("1984", 95.0, 80, 50),
            ("A Novel", 85.0, 160, 28),
            ("by George Orwell", 90.0, 250, 20),
        ])
        result = parse_fields(ocr)
        assert isinstance(result.title.confidence, float)
        assert isinstance(result.subtitle.confidence, float)
        assert isinstance(result.author.confidence, float)

    def test_empty_ocr_result(self):
        """Empty OCR produces all-empty fields."""
        ocr = OCRResult(
            words=[], full_text="", mean_confidence=0,
            min_confidence=0, detected_script=None,
            detected_language=None, status=PipelineStatus.SUCCESS,
            word_count=0,
        )
        result = parse_fields(ocr)
        assert result.title.value == ""
        assert result.subtitle.value == ""
        assert result.author.value == ""
        assert all(not v for v in result.fields_present.values())

    def test_confidence_source_documented(self):
        """Each parsed field documents its confidence source."""
        ocr = _make_ocr_result([
            ("The Catcher in the Rye", 93.0, 80, 48),
        ])
        result = parse_fields(ocr)
        assert result.title.source  # Not empty
