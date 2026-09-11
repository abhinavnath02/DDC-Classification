"""Pipeline configuration loader. (Triggering reload)

Loads pipeline_config.yaml from the repository root and exposes all
thresholds, paths, and lists as typed attributes. All threshold values
come from this single file — no magic numbers in pipeline code.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from .exceptions import ConfigurationError

# Repository root is two levels above src/ocr_pipeline/
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "pipeline_config.yaml"


class PipelineConfig:
    """Immutable pipeline configuration loaded from YAML."""

    def __init__(self, config_path: Path | str | None = None):
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Config file must be a YAML mapping: {path}")
        self._raw = raw
        self._validate()

    # ── OCR settings ────────────────────────────────────────────────────

    @property
    def ocr_engine(self) -> str:
        return self._raw["ocr"]["engine"]

    @property
    def ocr_min_word_confidence(self) -> float:
        return float(self._raw["ocr"]["min_word_confidence"])

    @property
    def ocr_min_mean_confidence(self) -> float:
        return float(self._raw["ocr"]["min_mean_confidence"])

    @property
    def tesseract_cmd(self) -> str | None:
        return self._raw["ocr"].get("tesseract_cmd")

    @property
    def osd_min_script_confidence(self) -> float:
        return float(self._raw["ocr"].get("osd_min_script_confidence", 5.0))

    @property
    def tesseract_config(self) -> str:
        return self._raw["ocr"].get("tesseract_config", "--oem 3 --psm 3")

    # ── Preprocessing settings ──────────────────────────────────────────

    @property
    def deskew_enabled(self) -> bool:
        return bool(self._raw["preprocessing"]["deskew_enabled"])

    @property
    def contrast_normalize(self) -> bool:
        return bool(self._raw["preprocessing"]["contrast_normalize"])

    @property
    def crop_to_text(self) -> bool:
        return bool(self._raw["preprocessing"]["crop_to_text"])

    # ── Routing thresholds ──────────────────────────────────────────────

    @property
    def classification_auto_accept_threshold(self) -> float:
        return float(self._raw["routing"]["classification_auto_accept_threshold"])

    @property
    def classification_needs_review_threshold(self) -> float:
        return float(self._raw["routing"]["classification_needs_review_threshold"])

    @property
    def ocr_reject_threshold(self) -> float:
        return float(self._raw["routing"]["ocr_reject_threshold"])

    # ── Paths ───────────────────────────────────────────────────────────

    @property
    def model_artifact_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / self._raw["paths"]["model_artifact"]

    @property
    def image_inbox(self) -> Path:
        return Path(__file__).resolve().parents[2] / self._raw["paths"]["image_inbox"]

    @property
    def processed_output(self) -> Path:
        return Path(__file__).resolve().parents[2] / self._raw["paths"]["processed_output"]

    @property
    def review_queue_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / self._raw["paths"]["review_queue"]

    @property
    def archive_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / self._raw["paths"]["archive"]

    # ── Feature guard ───────────────────────────────────────────────────

    @property
    def supported_formats(self) -> List[str]:
        return list(self._raw["supported_formats"])

    @property
    def forbidden_model_features(self) -> List[str]:
        return list(self._raw["forbidden_model_features"])

    @property
    def valid_labels(self) -> List[str]:
        return list(self._raw["valid_labels"])

    # ── Routing thresholds as a dict (for logging) ──────────────────────

    def routing_thresholds_dict(self) -> dict:
        """Return all routing-related thresholds as a flat dict for audit logging."""
        return {
            "classification_auto_accept_threshold": self.classification_auto_accept_threshold,
            "classification_needs_review_threshold": self.classification_needs_review_threshold,
            "ocr_reject_threshold": self.ocr_reject_threshold,
            "ocr_min_mean_confidence": self.ocr_min_mean_confidence,
            "ocr_min_word_confidence": self.ocr_min_word_confidence,
        }

    # ── Validation ──────────────────────────────────────────────────────

    def _validate(self):
        """Check required keys and value constraints."""
        required_sections = ["ocr", "preprocessing", "routing", "paths",
                             "supported_formats", "forbidden_model_features", "valid_labels"]
        for section in required_sections:
            if section not in self._raw:
                raise ConfigurationError(f"Missing required config section: '{section}'")

        # Threshold ordering
        auto = self.classification_auto_accept_threshold
        review = self.classification_needs_review_threshold
        if not (0 <= review <= auto <= 1):
            raise ConfigurationError(
                f"Threshold ordering violated: needs_review ({review}) "
                f"must be <= auto_accept ({auto}), both in [0, 1]."
            )

        # Valid labels must be the ten DDC classes
        expected = {f"{i}00" for i in range(10)}
        if set(self.valid_labels) != expected:
            raise ConfigurationError(
                f"valid_labels must be exactly the ten DDC classes. "
                f"Got: {self.valid_labels}"
            )
