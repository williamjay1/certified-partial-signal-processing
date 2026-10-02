"""Explicit, non-destructive public research entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import REPOSITORY_ROOT, configure_runtime, data_root, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Verify declared release bytes; standard library only")
    validate.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "release_manifest.json")
    validate.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    replay = commands.add_parser("replay", help="Recompute saved primary numbers without models or timing")
    replay.add_argument("--data-root", type=Path)
    replay.add_argument("--skip-summary-comparison", action="store_true", help="Explicitly skip comparison to saved summary JSON")
    replay.add_argument("--report", type=Path, help="Optional fresh D: JSON filename")
    bootstrap = commands.add_parser("bootstrap", help="Recompute the original shared-subject 10,000-sample analysis")
    bootstrap.add_argument("--data-root", type=Path)
    bootstrap.add_argument("--output", type=Path, required=True, help="Fresh absolute D: output directory")
    commands.add_parser("test", help="Run small numeric kernel/feature and manifest integrity tests")
    conformance = commands.add_parser("conformance", help="Finite synthetic probes of four deployment models; no timing")
    conformance.add_argument("--data-root", type=Path)
    conformance.add_argument("--report", type=Path, help="Optional fresh D: JSON filename")
    download = commands.add_parser("download-raw", help="Explicit upstream-only fresh raw download; no parsing or fitting")
    download.add_argument("--dataset", choices=("har", "hydraulic"), required=True)
    download.add_argument("--raw-root", type=Path, help="Absolute F: destination, or explicitly supplied E:")
    args = parser.parse_args()
    if args.command == "validate":
        from .manifest import validate_manifest
        result = validate_manifest(args.manifest, args.root)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    elif args.command == "replay":
        from .replay import replay as run_replay
        result = run_replay(data_root(args.data_root), compare_saved=not args.skip_summary_comparison)
        if args.report:
            report = output_path(args.report)
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "bootstrap":
        configure_runtime()
        from .saved_bootstrap import main as bootstrap_main
        output = output_path(args.output)
        bootstrap_main(data_root(args.data_root) / "results/har_certified_test/locked_execution_v1", output)
        print(json.dumps({"status": "RECOMPUTED_SAVED_SHARED_SUBJECT_BOOTSTRAP", "output": str(output),
                          "bootstrap_repetitions": 10000, "bootstrap_seed": 7302026,
                          "models_fitted": 0, "predictions_rerun": 0, "timing_rerun": 0}))
    elif args.command == "test":
        configure_runtime()
        import unittest
        suite = unittest.defaultTestLoader.discover(str(REPOSITORY_ROOT / "tests"))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            return 1
    elif args.command == "conformance":
        configure_runtime()
        from .conformance import run_conformance
        result = run_conformance(data_root(args.data_root), REPOSITORY_ROOT / "config/deployed_models.json")
        if args.report:
            report = output_path(args.report)
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "download-raw":
        from .download_raw import download
        result = download(args.dataset, REPOSITORY_ROOT / "config/raw_sources.json", destination=args.raw_root)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
