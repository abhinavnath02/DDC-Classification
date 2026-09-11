"""Gemini Vision OCR — uses Google's Gemini model for intelligent image understanding.

Instead of raw OCR, this module sends the book page image to Gemini and asks it
to extract structured fields (title, subtitle, author, description/summary).
This produces dramatically better results on artistic covers, stylized fonts,
and complex layouts that confuse traditional OCR engines like Tesseract.

Requires GEMINI_API_KEY environment variable to be set.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types


_EXTRACTION_PROMPT = """You are an expert librarian and book cataloger. Analyze this book page image and extract the following information.

**Instructions:**
- Extract all visible text from the image.
- Identify the book's title, subtitle (if any), and author (if visible).
- Provide a brief description/summary if there is descriptive text on the page.
- If you cannot confidently identify a field, leave it as an empty string.
- Return ONLY valid JSON, no markdown fencing, no explanation.

**Return format (strict JSON):**
{
  "title": "the book title",
  "subtitle": "the subtitle if present, else empty string",
  "author": "the author if present, else empty string",
  "description": "any descriptive/summary text visible on the page",
  "full_extracted_text": "all visible text on the page, preserving reading order",
  "ddc_class": "the 3-digit Dewey Decimal Classification string (e.g. '000', '100', '200', '300', '400', '500', '600', '700', '800', '900') that best describes the subject",
  "ddc_confidence": 0.90,
  "confidence": 0.95
}"""


def extract_with_gemini(
    image_path: str | Path,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash",
) -> dict:
    """Extract structured book fields from an image using Gemini Vision.

    Parameters
    ----------
    image_path : path to the book page image
    api_key : Gemini API key (falls back to GEMINI_API_KEY env var)
    model_name : which Gemini model to use

    Returns
    -------
    dict with keys: title, subtitle, author, description,
                    full_extracted_text, confidence

    Raises
    ------
    ValueError if no API key is found
    RuntimeError if the Gemini call fails
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    path = Path(image_path)
    image_bytes = path.read_bytes()

    # Determine MIME type
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    client = genai.Client(api_key=key)

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes,
                        )
                    ),
                    types.Part(text=_EXTRACTION_PROMPT),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ],
        ),
    )

    raw_text = response.text.strip()

    # Strip markdown code fencing if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw text as a fallback
        result = {
            "title": "",
            "subtitle": "",
            "author": "",
            "description": "",
            "full_extracted_text": raw_text,
            "ddc_class": "",
            "ddc_confidence": 0.0,
            "confidence": 0.3,
        }

    # Ensure all expected keys exist
    for key_name in ["title", "subtitle", "author", "description", "full_extracted_text", "ddc_class", "ddc_confidence", "confidence"]:
        if key_name not in result:
            result[key_name] = "" if "confidence" not in key_name else 0.5

    return result
