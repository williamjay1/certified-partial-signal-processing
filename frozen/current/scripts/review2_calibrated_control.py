"""Read-only availability audit for an unexecuted CRC-inspired follow-up.

This program deliberately does not select a stopping rule, load a model, read
official test outputs, fit, predict, or time anything.  Full final-schedule
development candidates are absent; combining uniform-A development outcomes
with final-C test outcomes would change the rule being calibrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(r"D:\MLWork\ASC_CertifiedProcessing_20261002")
for path in (
    Path(r"D:\MLWork\ASC_SelectiveReplanning_20260930\env_core"),
    Path(r"D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages"),
):
    sys.path.insert(0, str(path))
import numpy as np

SEEDS = (17, 29, 43)
PROPOSED_TOLERANCES = (0.1, 0.05, 0.025, 0.01)
FINAL_CHECKPOINTS = (12, 16, 18)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inspect_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {
            "path": str(path),
            "sha256": digest(path),
            "shapes": {name: list(data[name].shape) for name in data.files},
            "subjects": np.unique(data["subjects"]).tolist(),
            "window_count": int(len(data["development_indices"])),
            "methods": data["method_names"].tolist(),
        }
        if "deltas" in data.files:
            result["tolerances"] = [
                float(value) if np.isfinite(value) else None
                for value in data["deltas"]
            ]
        return result


def audit() -> dict:
    schedule_path = (
        ROOT / "results/har_checkpoint_development/schedule_development_v1"
        / "selected_schedule.json"
    )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if tuple(schedule["checkpoints"]) != FINAL_CHECKPOINTS:
        raise ValueError("The final-schedule provenance changed; re-audit explicitly")
    records = []
    for seed in SEEDS:
        old_folder = ROOT / "results/har_compute_development/corrected_execution" / f"seed{seed}"
        stop_folder = ROOT / "results/har_stopping_controls_development/frozen_controls_C_v1" / f"seed{seed}"
        checkpoint_folder = ROOT / "results/har_checkpoint_development/schedule_development_v1" / f"seed{seed}"
        old_lock_path = old_folder / "frozen_order_and_cohort_before_development.json"
        old_lock = json.loads(old_lock_path.read_text(encoding="utf-8"))
        old = inspect_npz(old_folder / "full_development_outcomes.npz")
        final_selected = inspect_npz(stop_folder / "all_development_selected_outcomes.npz")
        cohort = inspect_npz(checkpoint_folder / "cohort_outcomes.npz")
        assert old["window_count"] == final_selected["window_count"]
        assert old["subjects"] == final_selected["subjects"]
        assert len(old["subjects"]) == 4 and cohort["window_count"] == 300
        assert len(old["methods"]) == 17 and len(final_selected["methods"]) == 6
        records.append({
            "seed": seed,
            "model_sha256": old_lock["model_sha256"],
            "temperature": old_lock["temperature"],
            "original_order_lock_sha256": digest(old_lock_path),
            "complete_uniform_A_candidates": old,
            "complete_final_C_selected_rules": final_selected,
            "checkpoint_comparison_cohort_only": cohort,
            "missing_complete_final_C_tight_tolerances": [0.1, 0.05, 0.01],
            "selection_executed": False,
            "same900_followup_metrics_available": False,
            "all2947_followup_metrics_available": False,
        })
    return {
        "status": "NOT_EXECUTED_INCOMPATIBLE_DEVELOPMENT_CANDIDATES",
        "audit_only": True,
        "script_sha256": digest(Path(__file__)),
        "selected_schedule_sha256": digest(schedule_path),
        "final_checkpoints": list(FINAL_CHECKPOINTS),
        "proposed_tolerances": list(PROPOSED_TOLERANCES),
        "terminal_candidate": "fixed compiled complete reference, zero relative loss",
        "proposed_bound_B": 0.1,
        "subject_count_n": 4,
        "target_delta": 0.025,
        "independent_calibration": False,
        "official_test_outputs_read_by_this_audit": False,
        "new_fits": 0,
        "new_predictions": 0,
        "new_timings": 0,
        "reason": (
            "Full candidate grid was saved for uniform A, but final C only has "
            "the primary .025 rule and complete controls on all four subjects. "
            "The C comparison grid has only the 300-window cohort. "
            "No rule is selected from mismatched data or from test outcomes."
        ),
        "seeds": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(audit(), indent=2, ensure_ascii=False, allow_nan=False))
