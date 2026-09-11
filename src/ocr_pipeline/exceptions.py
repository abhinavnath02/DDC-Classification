"""Typed exceptions for each pipeline failure mode.

Every failure raises a specific exception rather than silently returning
a default value. This ensures no stage swallows errors.
"""


class OCRPipelineError(Exception):
    """Base exception for all OCR pipeline errors."""


class UnsupportedFormatError(OCRPipelineError):
    """Raised when the input image format is not in the supported list."""

    def __init__(self, path: str, extension: str, supported: list):
        self.path = path
        self.extension = extension
        self.supported = supported
        super().__init__(
            f"Unsupported image format '{extension}' for file '{path}'. "
            f"Supported formats: {', '.join(supported)}"
        )


class PreprocessingFailedError(OCRPipelineError):
    """Raised when preprocessing cannot produce a usable image.

    For example, no text region detected after crop attempts.
    The item stops at this stage with preprocessing_failed status.
    """

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Preprocessing failed for '{path}': {reason}")


class OCRFailedError(OCRPipelineError):
    """Raised when OCR produces no output or the engine itself fails."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"OCR failed for '{path}': {reason}")


class NonLatinScriptError(OCRPipelineError):
    """Raised when a non-Latin script is detected and the current OCR
    engine configuration does not support it.

    Per §2.3: fail explicitly rather than emitting garbage.
    """

    def __init__(self, detected_script: str, detected_language: str = ""):
        self.detected_script = detected_script
        self.detected_language = detected_language
        super().__init__(
            f"Non-Latin script detected: {detected_script}"
            + (f" (language: {detected_language})" if detected_language else "")
            + ". Current OCR configuration does not support this script. "
            "Reconfigure with appropriate language packs or use a different engine."
        )


class ModelNotFoundError(OCRPipelineError):
    """Raised when the frozen model artifact cannot be loaded."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        super().__init__(
            f"Trained model not found at '{model_path}'. "
            "Run train_baseline.py first or provide a model artifact."
        )


class FeatureLeakageError(OCRPipelineError):
    """Raised when a forbidden feature is detected in model input.

    This is a hard requirement (§2.6, §3): the build must fail if a
    forbidden field ever reaches the model.
    """

    def __init__(self, forbidden_fields: list):
        self.forbidden_fields = forbidden_fields
        super().__init__(
            f"FEATURE LEAKAGE DETECTED: forbidden fields in model input: "
            f"{', '.join(forbidden_fields)}. "
            "These must never reach the model. See README §183 and pipeline_config.yaml."
        )


class ConfigurationError(OCRPipelineError):
    """Raised when pipeline configuration is invalid or missing."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Pipeline configuration error: {reason}")
