"""Read-only reconstruction of locked primary outcomes from released arrays.

Never loads model pickles, fits a model, reruns predictions, or measures time.
Every-label certificates and subject ownership are checked from saved vectors.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

SEEDS = (17, 29, 43)
PRIMARY_DELTA = .025
HAR_RELATIVE = Path("results/har_certified_test/locked_execution_v1")
HYDRAULIC_RELATIVE = Path("results/hydraulic_external_confirmation/20261002T061642Z_3208a13f")


def require(condition, message: str):
    if not bool(condition):
        raise ValueError(message)


def half_brier(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    require(p.ndim == 2 and y.shape == (len(p),), "Probability/label shape mismatch")
    require(np.isfinite(p).all() and (p >= -2e-12).all(), "Invalid probabilities")
    require(np.max(np.abs(p.sum(-1) - 1.)) <= 2e-12, "Probabilities must sum to one")
    require((y >= 0).all() and (y < p.shape[1]).all(), "Label out of range")
    return .5 * np.square(p - np.eye(p.shape[1])[y]).sum(-1)


def every_label_excess(p, q):
    require(p.shape == q.shape, "Reference/output shape mismatch")
    return .5 * (np.square(p).sum(-1) - np.square(q).sum(-1))[:, None] - p + q


def select(z, method: str, delta: float | None):
    names, deltas = z["method_names"], z["deltas"]
    selected = names == method
    if delta is not None:
        selected = selected & (deltas == delta)
    ids = np.flatnonzero(selected)
    require(len(ids) == 1, f"Need exactly one saved method {method}, delta={delta}")
    return int(ids[0])


def _close(actual: float, expected: float, label: str):
    require(np.isclose(actual, expected, rtol=2e-12, atol=2e-12), "Saved-summary mismatch: " + label)


def replay(data: Path, *, compare_saved: bool = True) -> dict:
    part_means, full_means, risks, full_risks, systems = [], [], [], [], []
    common_cohort = common_labels = common_subjects = None
    for seed in SEEDS:
        folder = data / HAR_RELATIVE / f"seed{seed}"
        with np.load(folder / "timing_cohort_outcomes.npz", allow_pickle=False) as z:
            index = select(z, "tight_minimax", PRIMARY_DELTA)
            p = z["probabilities"][index].copy()
            q = z["reference_probabilities"].copy()
            y = z["labels"].copy()
            cases = z["test_indices"].copy()
            subjects = z["subjects"].copy()
            certificate = z["certificates"][index].copy()
        require(len(cases) == 900 and len(np.unique(cases)) == 900, "Expected 900 unique timed windows")
        require(len(np.unique(subjects)) == 9, "Expected the same nine held-out subjects")
        require(np.all(np.unique(subjects, return_counts=True)[1] == 100), "Expected 100 timed windows per subject")
        if common_cohort is None:
            common_cohort, common_labels, common_subjects = cases, y, subjects
        else:
            require(np.array_equal(cases, common_cohort) and np.array_equal(y, common_labels)
                    and np.array_equal(subjects, common_subjects), "Seeds must reuse identical subjects/cases/labels")
        ell, base = half_brier(p, y), half_brier(q, y)
        excess = every_label_excess(p, q).max(-1)
        require(excess.max() <= PRIMARY_DELTA + 2e-12, "Every-label tolerance violation")
        require(np.max(excess - certificate) <= 2e-12, "Emitted certificate violation")
        times = {"tight_minimax": {}, "native_full": {}, "compiled_full": {}}
        probabilities = {int(case): p[i] for i, case in enumerate(cases)}
        for line in (folder / "timing_case_results.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            method = r["name"]
            if method not in times or (method == "tight_minimax" and r["delta"] != PRIMARY_DELTA):
                continue
            case, repetition = int(r["case"]), int(r["rep"])
            identity = case, repetition
            require(case in probabilities and repetition in (0, 1, 2), "Unexpected timing identity")
            require(identity not in times[method], "Duplicate timing call")
            complete_ms = float(r["complete_ms"])
            require(np.isfinite(complete_ms) and complete_ms > 0, "Invalid complete-call time")
            times[method][identity] = complete_ms
            if method == "tight_minimax":
                require(np.max(np.abs(np.asarray(r["probability"]) - probabilities[case])) <= 2e-12,
                        "Timing/outcome probability disagreement")
        require(all(len(t) == 2700 for t in times.values()), "Expected 2700 calls per method per system")
        identities = set(times["tight_minimax"])
        require(all(set(t) == identities for t in times.values()), "Timing methods are not paired")
        full_backend = min(("native_full", "compiled_full"), key=lambda name: np.mean(list(times[name].values())))
        full = float(np.mean(list(times[full_backend].values())))
        partial = float(np.mean(list(times["tight_minimax"].values())))
        part_means.append(partial); full_means.append(full)
        risks.append(float(ell.mean())); full_risks.append(float(base.mean()))
        system = {"seed": seed, "timed_windows": 900, "raw_calls_per_method": 2700,
                  "partial_ms": partial, "faster_full_ms": full, "faster_full_backend": full_backend,
                  "time_reduction": 1 - partial / full, "half_brier": float(ell.mean()),
                  "reference_half_brier": float(base.mean()), "risk_excess": float((ell - base).mean()),
                  "maximum_every_label_excess": float(excess.max()),
                  "maximum_certificate_violation_signed": float(np.max(excess - certificate)),
                  "latency_quantiles": {name: {"p50_ms": float(np.quantile(list(t.values()), .5)),
                                                "p95_ms": float(np.quantile(list(t.values()), .95)),
                                                "p99_ms": float(np.quantile(list(t.values()), .99)),
                                                "max_ms": float(np.max(list(t.values())))}
                                        for name, t in times.items()}}
        systems.append(system)
    pooled = {"partial_ms": float(np.mean(part_means)), "faster_full_ms": float(np.mean(full_means)),
              "time_reduction": float(1 - np.mean(part_means) / np.mean(full_means)),
              "half_brier": float(np.mean(risks)), "reference_half_brier": float(np.mean(full_risks)),
              "risk_excess": float(np.mean(risks) - np.mean(full_risks))}
    hydro = data / HYDRAULIC_RELATIVE
    with np.load(hydro / "outcomes.npz", allow_pickle=False) as z:
        i, j = select(z, "tight_minimax", PRIMARY_DELTA), select(z, "compiled_full", None)
        p, q, y = z["probabilities"][i], z["reference_probability"], z["labels"]
        ell, base = half_brier(p, y), half_brier(q, y)
        excess = every_label_excess(p, q).max(-1)
        units, times = z["proxy_segments"], z["complete_ms"]
        require(len(y) == 435 and len(np.unique(units)) == 48, "Expected 435 cycles in 48 proxy segments")
        require(np.isfinite(times).all() and (times > 0).all(), "Invalid Hydraulic timing")
        require(excess.max() <= PRIMARY_DELTA + 2e-12, "Hydraulic every-label tolerance violation")
        certificate = z["certificates"][i]
        require(np.max(excess[:, None] - certificate) <= 2e-12 if certificate.ndim == 2
                else np.max(excess - certificate) <= 2e-12, "Hydraulic emitted certificate violation")
        hydraulic = {"cycles": 435, "proxy_segments": 48, "partial_ms": float(times[i].mean()),
                     "compiled_full_ms": float(times[j].mean()),
                     "time_reduction": float(1 - times[i].mean() / times[j].mean()),
                     "half_brier": float(ell.mean()), "reference_half_brier": float(base.mean()),
                     "risk_excess": float((ell - base).mean()),
                     "maximum_every_label_excess": float(excess.max())}
    if compare_saved:
        analysis = json.loads((data / "results/final_analysis/har_locked_v1/heldout_analysis.json").read_text(encoding="utf-8-sig"))
        primary = next(r for r in analysis["pooled"] if r["name"] == "tight_minimax" and r["delta"] == PRIMARY_DELTA)
        for actual_name, expected_name in (("partial_ms", "complete_mean_ms"), ("time_reduction", "time_reduction"),
                                           ("half_brier", "timed_cohort_risk"), ("risk_excess", "timed_cohort_risk_excess")):
            _close(pooled[actual_name], primary[expected_name], "HAR " + actual_name)
        expected = json.loads((hydro / "executed_results.json").read_text(encoding="utf-8-sig"))
        row = next(r for r in expected["all_methods"] if r["method"] == "tight_minimax" and r["delta"] == PRIMARY_DELTA)
        _close(hydraulic["partial_ms"], row["complete_ms"]["mean"], "Hydraulic complete time")
        _close(hydraulic["time_reduction"], row["complete_mean_time_reduction_vs_faster_full"], "Hydraulic time reduction")
    return {"status": "PASS_SAVED_NUMERICAL_REPLAY", "models_fitted": 0, "predictions_rerun": 0,
            "timing_rerun": 0, "summary_comparison_executed": compare_saved,
            "HAR_systems": systems, "HAR_pooled": pooled, "Hydraulic": hydraulic,
            "inference_limit": "Saved complete times are machine-specific observations; replay does not rerun or guarantee latency."}
