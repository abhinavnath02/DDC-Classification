"""Tests for Stage 5: raw_text assembly (§2.5)."""
import pytest

from src.ocr_pipeline.assemble import assemble_raw_text
from src.ocr_pipeline.schemas import ParsedField, ParsedFields


class TestAssemble:
    """raw_text assembly tests."""

    def test_title_only_shape(self):
        """Title-only input produces raw_text matching model_text() output."""
        parsed = ParsedFields(
            title=ParsedField(value="Moby Dick", confidence=0.9, source="test"),
            subtitle=ParsedField(value="", confidence=0.0, source="not_found"),
            author=ParsedField(value="Herman Melville", confidence=0.8, source="test"),
            fields_present={"title": True, "subtitle": False, "author": True},
        )
        result = assemble_raw_text(parsed)
        assert result.raw_text == "Moby Dick"
        assert result.text_availability == "title_only"
        assert result.fields_present["title"] is True
        assert result.fields_present["subtitle"] is False
        assert "subject_headings" in result.fields_absent

    def test_title_and_subtitle_shape(self):
        """Title + subtitle produces multiline raw_text."""
        parsed = ParsedFields(
            title=ParsedField(value="Clean Code", confidence=0.95, source="test"),
            subtitle=ParsedField(value="A Handbook of Agile Software Craftsmanship",
                                 confidence=0.7, source="test"),
            author=ParsedField(value="Robert C. Martin", confidence=0.8, source="test"),
            fields_present={"title": True, "subtitle": True, "author": True},
        )
        result = assemble_raw_text(parsed)
        assert "Clean Code" in result.raw_text
        assert "Handbook" in result.raw_text
        assert result.text_availability == "title_and_context"

    def test_subtitle_redundant_with_title(self):
        """If subtitle is redundant with title, model_text() drops it."""
        parsed = ParsedFields(
            title=ParsedField(value="Clean Code", confidence=0.95, source="test"),
            subtitle=ParsedField(value="Clean Code", confidence=0.4, source="test"),
            author=ParsedField(value="", confidence=0.0, source="not_found"),
            fields_present={"title": True, "subtitle": True, "author": False},
        )
        result = assemble_raw_text(parsed)
        # model_text() should not duplicate the subtitle if it's redundant
        lines = result.raw_text.strip().split("\n")
        # Count occurrences of "Clean Code" — should be exactly 1
        assert result.raw_text.lower().count("clean code") == 1

    def test_no_author_in_raw_text(self):
        """Author is never included in raw_text per README contract."""
        parsed = ParsedFields(
            title=ParsedField(value="Fahrenheit 451", confidence=0.9, source="test"),
            subtitle=ParsedField(value="", confidence=0.0, source="not_found"),
            author=ParsedField(value="Ray Bradbury", confidence=0.9, source="test"),
            fields_present={"title": True, "subtitle": False, "author": True},
        )
        result = assemble_raw_text(parsed)
        assert "Bradbury" not in result.raw_text
        assert "Ray" not in result.raw_text

    def test_fields_absent_tracked(self):
        """Fields not available from OCR are explicitly listed."""
        parsed = ParsedFields(
            title=ParsedField(value="Test Title", confidence=0.9, source="test"),
            subtitle=ParsedField(value="", confidence=0.0, source="not_found"),
            author=ParsedField(value="", confidence=0.0, source="not_found"),
            fields_present={"title": True, "subtitle": False, "author": False},
        )
        result = assemble_raw_text(parsed)
        assert "first_sentence" in result.fields_absent
        assert "subject_headings" in result.fields_absent

    def test_empty_title_produces_empty_raw_text(self):
        """Empty title produces empty raw_text (not fabricated)."""
        parsed = ParsedFields(
            title=ParsedField(value="", confidence=0.0, source="no_lines"),
            subtitle=ParsedField(value="", confidence=0.0, source="no_lines"),
            author=ParsedField(value="", confidence=0.0, source="no_lines"),
            fields_present={"title": False, "subtitle": False, "author": False},
        )
        result = assemble_raw_text(parsed)
        assert result.raw_text == ""
