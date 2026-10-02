"""Standard-library verification of every declared release file.

Reject traversal and paths escaping the requested root, including symlinks.
This validates byte identity, not biological validity or reproducibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: Path, root: Path) -> dict:
    root = root.resolve(strict=True)
    document = json.loads(manifest.read_text(encoding="utf-8-sig"))
    records = document.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest requires a nonempty files list")
    seen = set()
    checked = []
    for record in records:
        name = record.get("relative_path")
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("Use a nonempty relative POSIX path in relative_path")
        logical = PurePosixPath(name)
        if logical.is_absolute() or ".." in logical.parts or ":" in name:
            raise ValueError("Unsafe manifest path: " + name)
        identity = name.casefold()
        if identity in seen:
            raise ValueError("Duplicate manifest file: " + name)
        seen.add(identity)
        path = root.joinpath(*logical.parts).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Manifest path escapes release root or is not a file: " + name)
        actual_bytes = path.stat().st_size
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or actual_bytes != expected_bytes:
            raise ValueError("Size mismatch: " + name)
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("Invalid SHA-256 metadata: " + name)
        actual_hash = sha256(path)
        if actual_hash != expected_hash.lower():
            raise ValueError("SHA-256 mismatch: " + name)
        checked.append({"relative_path": name, "bytes": actual_bytes, "sha256": actual_hash})
    return {"status": "PASS_RELEASE_BYTE_VERIFICATION", "files_checked": len(checked),
            "bytes_checked": sum(r["bytes"] for r in checked), "files": checked}
