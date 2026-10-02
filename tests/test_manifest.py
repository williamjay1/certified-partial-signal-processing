"""Meaningful file-integrity and traversal tests; no network or model loading."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from asc_certified.config import work_root
from asc_certified.manifest import validate_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=work_root() / "temp")
        self.root = Path(self.temp.name)
        self.file = self.root / "sample.txt"
        self.file.write_bytes(b"independent integrity fixture\n")
        self.record = {"relative_path": "sample.txt", "bytes": self.file.stat().st_size,
                       "sha256": hashlib.sha256(self.file.read_bytes()).hexdigest()}
        self.manifest = self.root / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, records):
        self.manifest.write_text(json.dumps({"files": records}), encoding="utf-8")

    def test_valid_manifest(self):
        self.write_manifest([self.record])
        self.assertEqual(validate_manifest(self.manifest, self.root)["files_checked"], 1)

    def test_tamper_rejected_even_same_size(self):
        self.write_manifest([self.record])
        self.file.write_bytes(b"X" * self.record["bytes"])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_manifest(self.manifest, self.root)

    def test_parent_and_absolute_paths_rejected(self):
        for name in ("../sample.txt", "/sample.txt", "D:/sample.txt", "..\\sample.txt"):
            with self.subTest(name=name):
                self.write_manifest([{**self.record, "relative_path": name}])
                with self.assertRaises(ValueError):
                    validate_manifest(self.manifest, self.root)

    def test_duplicate_records_rejected(self):
        self.write_manifest([self.record, {**self.record, "relative_path": "SAMPLE.TXT"}])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest(self.manifest, self.root)
