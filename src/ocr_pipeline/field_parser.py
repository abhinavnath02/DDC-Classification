"""Stage 4: Field parsing (§2.4).

Parses title/subtitle/author candidates from raw OCR text using
explicit RULE-BASED heuristics (not an ML model). Each parsed field
carries a per-field confidence score.

Key rules from the existing repo that apply here:
- Missing subtitle stays missing — never fabricated (README §85).
- Per-field confidence is emitted, not just parse-or-fail.
- A publisher's name must not be mistaken for the title.

This parser is documented as heuristic-based. It is the most likely
source of 'confidently wrong' output, which is why every field has
an explicit confidence and source attribution.
"""
from __future__ import annotations

import re
from typing import List

from .schemas import OCRResult, OCRWord, ParsedField, ParsedFields


# ── Heuristic helpers ───────────────────────────────────────────────────

# Common publisher names to avoid misidentifying as title/author
_PUBLISHER_PATTERNS = [
    r"\b(penguin|random\s*house|harper\s*collins|simon\s*&?\s*schuster|"
    r"macmillan|hachette|wiley|springer|elsevier|oxford\s*university\s*press|"
    r"cambridge\s*university\s*press|mcgraw[- ]hill|pearson|scholastic|"
    r"vintage|doubleday|knopf|pantheon|pocket\s*books|bantam|dell|"
    r"tor\s*books|daw\s*books|baen|orbit|ace\s*books|penguin\s*books|"
    r"edition|verlag|press|publishing|publishers|publications)\b"
]
_PUBLISHER_RE = re.compile("|".join(_PUBLISHER_PATTERNS), re.IGNORECASE)

# "by" prefix pattern for author detection
_BY_PATTERN = re.compile(r"^\s*by\s+", re.IGNORECASE)

# Common author-line patterns (e.g. "Written by ...", "Author: ...")
_AUTHOR_PREFIX_PATTERN = re.compile(
    r"^\s*(by|written\s+by|author[:\s])\s*", re.IGNORECASE
)


def _is_likely_publisher(text: str) -> bool:
    """Check if text looks like a publisher name."""
    return bool(_PUBLISHER_RE.search(text))


def _group_words_by_line(words: List[OCRWord]) -> List[List[OCRWord]]:
    """Group OCR words into lines, sorted by vertical position."""
    lines: dict[int, List[OCRWord]] = {}
    for w in words:
        lines.setdefault(w.line_num, []).append(w)
    # Sort lines by the top coordinate of their first word
    sorted_lines = sorted(lines.values(), key=lambda ws: ws[0].bbox.top)
    return sorted_lines


def _line_text(words: List[OCRWord]) -> str:
    """Reconstruct line text from words."""
    return " ".join(w.text for w in words)


def _line_mean_height(words: List[OCRWord]) -> float:
    """Mean height of words in a line (proxy for font size)."""
    if not words:
        return 0
    return sum(w.bbox.height for w in words) / len(words)


def _line_mean_confidence(words: List[OCRWord]) -> float:
    """Mean OCR confidence for a line."""
    if not words:
        return 0
    return sum(w.confidence for w in words) / len(words)


def parse_fields(ocr_result: OCRResult) -> ParsedFields:
    """Parse title, subtitle, and author from OCR output.

    Uses positional and font-size heuristics:
    - Title: the largest text block in the upper portion of the page.
    - Subtitle: a secondary text block near the title, smaller font.
    - Author: text preceded by "by" or in a smaller font below the title.

    Parameters
    ----------
    ocr_result : output from the OCR stage

    Returns
    -------
    ParsedFields with per-field confidence scores
    """
    lines = _group_words_by_line(ocr_result.words)

    if not lines:
        return ParsedFields(
            title=ParsedField(value="", confidence=0.0, source="no_lines"),
            subtitle=ParsedField(value="", confidence=0.0, source="no_lines"),
            author=ParsedField(value="", confidence=0.0, source="no_lines"),
            fields_present={"title": False, "subtitle": False, "author": False},
        )

    # Compute metrics for each line
    line_info = []
    for words in lines:
        text = _line_text(words)
        height = _line_mean_height(words)
        conf = _line_mean_confidence(words)
        is_publisher = _is_likely_publisher(text)
        has_author_prefix = bool(_AUTHOR_PREFIX_PATTERN.match(text))
        line_info.append({
            "words": words,
            "text": text.strip(),
            "height": height,
            "confidence": conf,
            "is_publisher": is_publisher,
            "has_author_prefix": has_author_prefix,
        })

    # Filter out very short lines (likely noise or page numbers)
    meaningful = [li for li in line_info if len(li["text"]) > 2 and not li["text"].isdigit()]
    if not meaningful:
        meaningful = line_info  # Fall back to all lines

    # ── Title: largest non-publisher text, preferring upper half ────────
    title_info = None
    title_candidates = [
        li for li in meaningful
        if not li["is_publisher"] and not li["has_author_prefix"]
    ]
    if not title_candidates:
        title_candidates = meaningful

    # Sort by font size (height) descending, then by position (top) ascending
    title_candidates.sort(key=lambda li: (-li["height"], li["words"][0].bbox.top))
    title_info = title_candidates[0]

    title_conf = min(1.0, title_info["confidence"] / 100.0 * 1.2)  # Scale OCR conf
    # Reduce confidence if title is suspiciously short
    if len(title_info["text"]) < 5:
        title_conf *= 0.5

    title = ParsedField(
        value=title_info["text"],
        confidence=round(title_conf, 3),
        source="positional_heuristic_largest_text",
    )

    # ── Author: look for "by" prefix or bottom-positioned small text ───
    author = ParsedField(value="", confidence=0.0, source="not_found")
    remaining = [li for li in meaningful if li is not title_info]

    for li in remaining:
        if li["has_author_prefix"]:
            author_text = _AUTHOR_PREFIX_PATTERN.sub("", li["text"]).strip()
            if author_text and not _is_likely_publisher(author_text):
                author_conf = min(1.0, li["confidence"] / 100.0 * 1.1)
                author = ParsedField(
                    value=author_text,
                    confidence=round(author_conf, 3),
                    source="by_prefix",
                )
                remaining = [r for r in remaining if r is not li]
                break

    # If no "by" prefix found, look for smaller text below the title
    if not author.value:
        below_title = [
            li for li in remaining
            if li["words"][0].bbox.top > title_info["words"][0].bbox.top
            and not li["is_publisher"]
            and li["height"] < title_info["height"] * 0.9
        ]
        if below_title:
            # The last small-text line before any publisher is likely the author
            for candidate in below_title:
                if len(candidate["text"]) > 2:
                    author_conf = min(1.0, candidate["confidence"] / 100.0 * 0.7)
                    author = ParsedField(
                        value=candidate["text"],
                        confidence=round(author_conf, 3),
                        source="positional_heuristic_below_title",
                    )
                    remaining = [r for r in remaining if r is not candidate]
                    break

    # ── Subtitle: secondary large text near the title, not publisher ───
    subtitle = ParsedField(value="", confidence=0.0, source="not_found")
    subtitle_candidates = [
        li for li in remaining
        if not li["is_publisher"]
        and not li["has_author_prefix"]
        and li is not title_info
        and len(li["text"]) > 3
    ]

    if subtitle_candidates:
        # Prefer text near the title and reasonably large
        subtitle_candidates.sort(
            key=lambda li: abs(li["words"][0].bbox.top - title_info["words"][0].bbox.top)
        )
        sub = subtitle_candidates[0]
        # Only accept if it's near the title (within 3x title height)
        gap = abs(sub["words"][0].bbox.top - title_info["words"][0].bbox.top)
        if gap < title_info["height"] * 5:
            sub_conf = min(1.0, sub["confidence"] / 100.0 * 0.8)
            subtitle = ParsedField(
                value=sub["text"],
                confidence=round(sub_conf, 3),
                source="positional_heuristic_near_title",
            )

    fields_present = {
        "title": bool(title.value),
        "subtitle": bool(subtitle.value),
        "author": bool(author.value),
    }

    return ParsedFields(
        title=title,
        subtitle=subtitle,
        author=author,
        fields_present=fields_present,
    )
