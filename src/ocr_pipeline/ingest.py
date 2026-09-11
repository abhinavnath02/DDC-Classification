"""Stage 1: Image ingest (§2.1).

Accepts image files, validates format, computes a content hash of the
raw image BEFORE any processing (audit trail), records provenance, and
copies the original to an archive path. The original image is never mutated.
"""
from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import PipelineConfig
from .exceptions import UnsupportedFormatError
from .schemas import IngestResult


def ingest_image(
    image_path: str | Path,
    config: PipelineConfig,
    batch_id: str | None = None,
    session_id: str | None = None,
) -> IngestResult:
    """Ingest a single image file.

    Parameters
    ----------
    image_path : path to the source image
    config : pipeline configuration
    batch_id : optional batch identifier; auto-generated if not supplied
    session_id : optional session identifier; auto-generated if not supplied

    Returns
    -------
    IngestResult with content hash, archive path, and provenance

    Raises
    ------
    UnsupportedFormatError
        If the file extension is not in the supported formats list.
    FileNotFoundError
        If the image file does not exist.
    """
    path = Path(image_path).resolve()

    # Validate existence
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Validate format
    ext = path.suffix.lower()
    if ext not in config.supported_formats:
        raise UnsupportedFormatError(str(path), ext, config.supported_formats)

    # Compute content hash BEFORE any processing — audit trail
    raw_bytes = path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Provenance
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    batch = batch_id or f"batch_{now.strftime('%Y%m%d_%H%M%S')}"
    session = session_id or str(uuid.uuid4())

    # Archive: copy original to archive/<batch>/<hash>_<original_name>
    archive_dir = config.archive_path / batch
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{content_hash[:12]}_{path.name}"
    archive_path = archive_dir / archive_name

    if not archive_path.exists():
        shutil.copy2(str(path), str(archive_path))

    return IngestResult(
        original_path=path,
        archive_path=archive_path,
        content_hash=content_hash,
        original_filename=path.name,
        upload_timestamp=timestamp,
        batch_id=batch,
        session_id=session,
        file_format=ext,
    )
