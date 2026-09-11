"""Stage 2: Image preprocessing (§2.2).

Deskew, crop to text region, normalize contrast. All operations are logged
so a reviewer can tell 'why does this text look wrong' from the record.
Derived images are written to separate paths — the original is never mutated.

If preprocessing fails (e.g. no text region detected), the item stops
with a preprocessing_failed status rather than being forced through OCR.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageFilter, ImageOps, ImageStat

from .config import PipelineConfig
from .exceptions import PreprocessingFailedError
from .schemas import PipelineStatus, PreprocessResult


def _detect_skew_angle(image: Image.Image) -> float:
    """Estimate skew angle using edge projection variance heuristic.

    Tries a range of small rotation angles and picks the one that
    maximizes row-projection variance (text lines produce strong peaks).
    Returns angle in degrees (positive = counter-clockwise).
    """
    gray = image.convert("L")
    # Use a smaller version for speed
    thumb = gray.copy()
    thumb.thumbnail((600, 600))
    best_angle = 0.0
    best_variance = 0.0

    for angle_tenths in range(-50, 51, 5):  # -5.0 to +5.0 degrees
        angle = angle_tenths / 10.0
        rotated = thumb.rotate(angle, fillcolor=255, expand=False)
        # Avoid Pillow 14 getdata() deprecation warning by converting to list directly or using getdata()
        pixels = list(rotated.getdata())
        width = rotated.width
        # Row projections (sum of dark pixels per row)
        rows = []
        for y in range(rotated.height):
            row_sum = sum(255 - pixels[y * width + x] for x in range(width))
            rows.append(row_sum)
        if len(rows) < 2:
            continue
        mean_val = sum(rows) / len(rows)
        variance = sum((r - mean_val) ** 2 for r in rows) / len(rows)
        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    return best_angle


def _crop_to_text_region(image: Image.Image) -> Tuple[Image.Image, bool]:
    """Crop to the bounding box of the text region using contrast analysis.

    Returns (cropped_image, True) if a text region was found, or
    (original_image, False) if no clear text region was detected.
    """
    gray = image.convert("L")
    # Threshold to binary
    threshold = 128
    stat = ImageStat.Stat(gray)
    if stat.mean[0] > 0:
        threshold = int(stat.mean[0] * 0.7)

    # Text is usually darker than the background. We want the text to be non-zero (255)
    # so getbbox() finds the bounding box of the text, not the background.
    binary = gray.point(lambda p: 255 if p < threshold else 0, "1")
    bbox = binary.getbbox()

    if bbox is None:
        return image, False

    # Add padding (5% of each dimension)
    pad_x = max(10, int(image.width * 0.05))
    pad_y = max(10, int(image.height * 0.05))
    left = max(0, bbox[0] - pad_x)
    top = max(0, bbox[1] - pad_y)
    right = min(image.width, bbox[2] + pad_x)
    bottom = min(image.height, bbox[3] + pad_y)

    cropped = image.crop((left, top, right, bottom))

    # Reject if the crop is too small (likely noise, not text)
    if cropped.width < 50 or cropped.height < 50:
        return image, False

    return cropped, True


def _normalize_contrast(image: Image.Image) -> Image.Image:
    """Histogram equalization for contrast normalization."""
    if image.mode == "RGBA":
        # Equalize only the RGB channels
        r, g, b, a = image.split()
        rgb = Image.merge("RGB", (r, g, b))
        eq = ImageOps.equalize(rgb)
        er, eg, eb = eq.split()
        return Image.merge("RGBA", (er, eg, eb, a))
    elif image.mode in ("L", "RGB"):
        return ImageOps.equalize(image)
    else:
        return ImageOps.equalize(image.convert("RGB"))


def preprocess_image(
    image_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
    image_id: str = "",
) -> PreprocessResult:
    """Preprocess a single image for OCR.

    Parameters
    ----------
    image_path : path to the image (original or archive copy)
    output_dir : directory to write the derived (preprocessed) image
    config : pipeline configuration
    image_id : identifier for naming the output file

    Returns
    -------
    PreprocessResult with the derived image path and applied operations

    Raises
    ------
    PreprocessingFailedError
        If no usable text region can be detected.
    """
    path = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    operations: List[str] = []

    try:
        image = Image.open(path)
        # Handle HEIC via pillow-heif (registered at import time)
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass  # HEIC not available — will fail on .heic files at Image.open
        image = Image.open(path)
    except Exception as exc:
        raise PreprocessingFailedError(str(path), f"Cannot open image: {exc}")

    # Convert to RGB if necessary (e.g. RGBA, P mode)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
        operations.append("convert_to_rgb")

    # 1. Deskew
    if config.deskew_enabled:
        angle = _detect_skew_angle(image)
        if abs(angle) > 0.3:  # Only apply if skew is noticeable
            image = image.rotate(angle, fillcolor=(255, 255, 255) if image.mode == "RGB" else 255, expand=True)
            operations.append(f"deskew_{angle:.1f}deg")

    # 2. Crop to text region
    if config.crop_to_text:
        cropped, found = _crop_to_text_region(image)
        if not found:
            raise PreprocessingFailedError(
                str(path),
                "No text region detected after crop analysis. "
                "The image may be blank, heavily degraded, or non-textual."
            )
        image = cropped
        operations.append("crop_to_text")

    # 3. Normalize contrast
    if config.contrast_normalize:
        image = _normalize_contrast(image)
        operations.append("contrast_normalize")

    # Save derived image
    stem = image_id or path.stem
    derived_path = out_dir / f"{stem}_preprocessed.png"
    image.save(str(derived_path), format="PNG")
    operations.append(f"saved_as_png:{derived_path.name}")

    return PreprocessResult(
        derived_image_path=derived_path,
        operations_applied=operations,
        status=PipelineStatus.SUCCESS,
    )
