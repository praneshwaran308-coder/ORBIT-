"""Uploaded-file context shared with agents."""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("orbit.files")

MAX_PREVIEW_ROWS = 5


class FileValidationError(ValueError):
    """Raised when an uploaded file is not a usable CSV."""


@dataclass
class FileContext:
    """A validated CSV upload, passed verbatim to Data/ML agents.

    AI Agent receives a bounded, secret-free preview via ``ai_preview()``.
    The raw bytes are kept in memory so agents can read the frame even after
    the on-disk upload copy is removed (post-acceptance hygiene).
    """

    path: str
    filename: str
    size_bytes: int
    _raw: bytes = field(default=b"", repr=False)
    _frame: pd.DataFrame | None = field(default=None, repr=False)

    @classmethod
    def from_upload(cls, filename: str, raw: bytes) -> "FileContext":
        if not filename.lower().endswith(".csv"):
            raise FileValidationError("Only .csv files are supported in v1.")
        if not raw:
            raise FileValidationError("Uploaded file is empty.")
        if len(raw) > _max_upload_bytes():
            limit_mb = _max_upload_bytes() / (1024 * 1024)
            raise FileValidationError(f"CSV exceeds the {limit_mb:.0f} MB upload limit.")
        # Validate parseability up-front.
        try:
            pd.read_csv(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            raise FileValidationError(f"Could not parse CSV: {exc}") from exc
        # Sanitize: never trust the client-supplied name (path traversal),
        # and uniquify to avoid collisions between uploads.
        safe_name = Path(filename).name
        if not safe_name.lower().endswith(".csv") or safe_name in (".csv", ""):
            safe_name = "upload.csv"
        unique_name = f"{uuid.uuid4().hex[:8]}-{safe_name}"
        target = _uploads_dir() / unique_name
        target.write_bytes(raw)
        return cls(path=str(target), filename=safe_name, size_bytes=len(raw), _raw=raw)

    # -- data access -------------------------------------------------------
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            if self._raw:
                self._frame = pd.read_csv(io.BytesIO(self._raw))
            else:
                self._frame = pd.read_csv(self.path)
        return self._frame

    def profile(self) -> dict[str, Any]:
        """Compact summary used when routing tasks that mention files."""
        df = self.frame()
        return {
            "filename": self.filename,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }

    def ai_preview(self, max_chars: int = 2500) -> str:
        """Bounded textual preview for the AI Agent's prompt context."""
        df = self.frame()
        head = df.head(MAX_PREVIEW_ROWS).to_string(index=False)
        text = (
            f"File: {self.filename}\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns/dtypes: {self.profile()['dtypes']}\n"
            f"First {MAX_PREVIEW_ROWS} rows:\n{head}"
        )
        return text[:max_chars]


def _max_upload_bytes() -> int:
    from app.config import get_settings

    return get_settings().max_upload_bytes


def _uploads_dir():
    from app.config import DATA_DIR

    d = DATA_DIR / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d
