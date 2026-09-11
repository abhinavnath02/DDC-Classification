"""Stage 5: raw_text assembly (§2.5).

Reuses the existing model_text() function from etl.py rather than
reimplementing the concatenation logic. This prevents the two
implementations from drifting out of sync.

OCR input will usually be thinner than catalog metadata:
- first_sentence is typically absent from a book cover/title page.
- subject_headings are typically absent (no catalog metadata).

These absent fields are explicitly tracked, not silently omitted.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

# Import model_text() from the existing ETL module — single source of truth.
# We add the repo root to sys.path only if needed for the import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ddc_data.etl import model_text, norm  # noqa: E402

from .schemas import AssembledText, ParsedFields


def assemble_raw_text(parsed: ParsedFields) -> AssembledText:
    """Build a raw_text string in the same shape the existing model expects.

    Uses model_text() from src/ddc_data/etl.py, which produces:
        title
        [subtitle if not redundant with title]
        [first_sentence if available]
        [Subjects: s1; s2; ... if available]

    For OCR input, first_sentence and subject_headings are almost always
    absent — they come from catalog metadata, not book covers.

    Parameters
    ----------
    parsed : output from the field parsing stage

    Returns
    -------
    AssembledText with the assembled raw_text and field coverage
    """
    # Build the row dict that model_text() expects
    row = {
        "title": parsed.title.value,
        "subtitle": parsed.subtitle.value,
        "first_sentence": "",          # Not available from OCR of a book cover
        "subject_headings": [],        # Not available from OCR of a book cover
    }

    # Call the single source of truth for field composition
    raw_text = model_text(row)

    # Determine text_availability per the existing logic
    has_context = bool(row["subtitle"]) or bool(row["subject_headings"]) or bool(row["first_sentence"])
    text_availability = "title_and_context" if has_context else "title_only"

    # Track which fields are present vs absent
    fields_present: Dict[str, bool] = {
        "title": bool(parsed.title.value),
        "subtitle": bool(parsed.subtitle.value),
        "first_sentence": False,       # Always absent from OCR
        "subject_headings": False,     # Always absent from OCR
    }
    fields_absent: List[str] = [k for k, v in fields_present.items() if not v]

    return AssembledText(
        raw_text=raw_text,
        text_availability=text_availability,
        fields_present=fields_present,
        fields_absent=fields_absent,
    )
