"""Small cross-platform filesystem helpers.

`force_rmtree` is a Windows-safe `shutil.rmtree`: Git pack files (and other artifacts) are marked
read-only, and on Windows `os.unlink` refuses to delete a read-only file — so a plain `rmtree` of a
genome repo (`.git/objects/pack/*`) raises `PermissionError`. The recovery handler clears the
read-only bit and retries, so wiping a recycled offspring slot / stale scratch works the same on
Windows as on Linux (the test run is a Windows host, TEST_RUN_PLAN §2).
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path


def _retry_after_chmod(func, path, _exc) -> None:
    """rmtree error handler: drop the read-only bit and retry the failed op (Windows)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:  # pragma: no cover - best-effort; a truly undeletable path re-raises elsewhere
        pass


def force_rmtree(path: Path | str) -> None:
    """`shutil.rmtree` that survives read-only files (e.g. Git objects) on Windows."""
    path = Path(path)
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry_after_chmod)
    else:  # pragma: no cover - exercised only on 3.11
        shutil.rmtree(path, onerror=_retry_after_chmod)
