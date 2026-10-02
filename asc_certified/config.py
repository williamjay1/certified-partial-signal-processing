"""Paths for the public code; no import of the author's private environment.

Repository inputs may be relocated. On the validated Windows backend, all
generated files and compiled caches stay on D:, and raw archives default to F:.
No directory is created merely by importing this module.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = "D:/MLWork/ASC_CertifiedProcessing_public"
DEFAULT_RAW = "F:/AcademicData/ASC_CertifiedProcessing/raw"


def data_root(value: str | Path | None = None) -> Path:
    return Path(value or os.environ.get("ASC_DATA_ROOT", REPOSITORY_ROOT / "data")).resolve()


def work_root(value: str | Path | None = None, *, create: bool = False) -> Path:
    candidate = value or os.environ.get("ASC_WORK_ROOT")
    if candidate is None:
        if os.name != "nt":
            raise RuntimeError("Non-Windows execution is unverified; set an explicit absolute ASC_WORK_ROOT to experiment.")
        candidate = DEFAULT_WORK
    path = Path(candidate)
    if not path.is_absolute():
        raise ValueError("ASC_WORK_ROOT must be an absolute path")
    path = path.resolve()
    if os.name == "nt" and path.drive.upper() != "D:":
        raise ValueError("Generated work, caches, training and outputs must remain on D: for this Windows study")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("An output path must be absolute; use a fresh D: directory")
    path = path.resolve()
    if os.name == "nt" and path.drive.upper() != "D:":
        raise ValueError("Research outputs must remain on D:")
    return path


def raw_root(value: str | Path | None = None) -> Path:
    candidate = value or os.environ.get("ASC_RAW_ROOT")
    if candidate is None:
        if os.name != "nt":
            raise RuntimeError("Set an explicit raw archive path on an unverified non-Windows system")
        candidate = DEFAULT_RAW
    path = Path(candidate)
    if not path.is_absolute():
        raise ValueError("Raw storage must use an absolute path")
    path = path.resolve()
    if os.name == "nt" and path.drive.upper() not in ("F:", "E:"):
        raise ValueError("Raw archive custody must stay on F: (preferred), or explicitly supplied E:")
    return path


def configure_runtime() -> Path:
    """Configure generated caches explicitly before importing Numba/plotters."""
    root = work_root(create=True)
    for name in ("cache/numba", "cache/python", "cache/matplotlib", "temp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(root / "cache/numba")
    os.environ["MPLCONFIGDIR"] = str(root / "cache/matplotlib")
    for name in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
        os.environ[name] = str(root / "temp")
    sys.pycache_prefix = str(root / "cache/python")
    return root
