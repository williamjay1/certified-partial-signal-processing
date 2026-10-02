"""Optional upstream-only source download, never executed by replay or tests.

Keeps a fresh raw filename read-only even on failure; never overwrites or deletes
existing archives. Exact study hashes fail closed if upstream bytes change.
No parsing, preparation, fitting or test outcome access is performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import uuid
import zipfile

from .config import raw_root, work_root


def download(dataset: str, source_specification: Path, *, destination: Path | None = None) -> dict:
    specification = json.loads(source_specification.read_text(encoding="utf-8"))["sources"][dataset]
    raw = raw_root(destination)
    work = work_root(create=True)
    raw.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(specification["bytes"])
    free_raw, free_work = shutil.disk_usage(raw).free, shutil.disk_usage(work).free
    if free_raw < expected_bytes + 100_000_000 or free_work < 100_000_000:
        raise RuntimeError("Insufficient raw/work space; no drive substitution or overwrite permitted")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    target = raw / f"{dataset}__{tag}.zip"
    report_path = work / "downloads" / f"{dataset}__{tag}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"dataset": dataset, "upstream_url": specification["download_url"],
              "source_page": specification["source_page"], "raw_path": str(target),
              "expected_sha256": specification["sha256"], "expected_bytes": expected_bytes,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "raw_free_bytes_before": free_raw, "work_free_bytes_before": free_work,
              "dataset_content_members_opened": 0, "models_fitted": 0}
    sha = hashlib.sha256()
    try:
        request = urllib.request.Request(specification["download_url"], headers={"User-Agent": "CertifiedProcessing-source-retrieval/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, target.open("xb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                sha.update(chunk)
        record.update(actual_sha256=sha.hexdigest(), actual_bytes=target.stat().st_size)
        if record["actual_sha256"] != specification["sha256"] or record["actual_bytes"] != expected_bytes:
            raise ValueError("Upstream bytes differ from frozen study source; new file retained, not silently accepted")
        if not zipfile.is_zipfile(target):
            raise ValueError("Upstream response is not a ZIP archive")
        record["status"] = "PASS_FROZEN_SOURCE_BYTES"
    except BaseException as error:
        record.update(status="FAILED_FRESH_FILE_RETAINED", error_type=type(error).__name__, error=str(error))
        if target.exists():
            record.update(actual_bytes=target.stat().st_size, actual_sha256=sha.hexdigest())
        raise
    finally:
        if target.exists():
            os.chmod(target, 0o444)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return {**record, "download_report": str(report_path)}
