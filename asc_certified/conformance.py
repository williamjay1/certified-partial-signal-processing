"""Finite synthetic probes of the four released deployment models.

This is a numerical regression check on published model bytes. Its synthetic
inputs are not additional HAR or Hydraulic data, and no latency is measured.
Model files use Python serialization; load only the trusted published artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .certificate_kernel import FrozenForest, choose_probability, probability_intervals, stable_softmax
from .leaf_mask_kernel import FrozenLeafMasks
from .manifest import sha256
from .replay import every_label_excess, require


def run_conformance(data: Path, specification: Path) -> dict:
    import joblib
    from threadpoolctl import threadpool_limits
    registry = json.loads(specification.read_text(encoding="utf-8"))
    results = []
    with threadpool_limits(limits=1):
        for entry in registry["models"]:
            path = data / entry["data_relative_path"]
            require(sha256(path) == entry["sha256"], "Frozen deployment model hash mismatch")
            bundle = joblib.load(path)
            model = bundle["model"] if entry["bundle"] else bundle
            temperature = float(bundle["temperature"]) if entry["bundle"] else 1.
            require(temperature == entry["temperature"], "Frozen temperature mismatch")
            group = np.repeat(np.arange(entry["blocks"]), np.tile([11, 6], entry["blocks"] // 2)).astype(np.int64)
            require(model.n_features_in_ == len(group), "Frozen feature-schema mismatch")
            forest = FrozenForest(model, group, entry["blocks"])
            # A copied baseline is required: scaling an aliased ndarray would
            # modify the native comparator held in memory.
            forest.baseline = forest.baseline.copy()
            if temperature != 1.:
                forest.values /= temperature
                forest.baseline /= temperature
                forest.static_lower /= temperature
                forest.static_upper /= temperature
                forest.root_abs /= temperature
            leaf = FrozenLeafMasks(forest)
            offsets = np.r_[0, np.cumsum(np.tile([11, 6], entry["blocks"] // 2))]
            rng = np.random.default_rng(10022026 + entry["probe_seed"])
            x = np.vstack((np.zeros(len(group)), rng.normal(size=(6, len(group))), np.full(len(group), np.nan)))
            native_scores = model._raw_predict(x) / temperature
            native = np.stack([stable_softmax(row) for row in native_scores])
            maximum_difference = maximum_interval_gap = maximum_certificate_violation = 0.
            checkpoints_checked = 0
            for case, actual in enumerate(x):
                exact = forest.exact(actual)
                maximum_difference = max(maximum_difference, float(np.max(np.abs(exact - native[case]))))
                obs, masks = np.zeros(len(group)), leaf.initial.copy()
                for block in forest.used:
                    start, stop = offsets[block:block + 2]
                    obs[start:stop] = actual[start:stop]
                    leaf.update(int(block), obs, masks)
                    lo, hi, _ = leaf.score_intervals(masks)
                    maximum_interval_gap = max(maximum_interval_gap,
                        float(np.max(lo - native_scores[case])), float(np.max(native_scores[case] - hi)))
                    l, u = probability_intervals(lo, hi)
                    require(np.all(exact >= l - 2e-12) and np.all(exact <= u + 2e-12), "Completed probabilities outside partial bounds")
                    for variant in (0, 1, 2):
                        p, bound = choose_probability(l, u, variant)
                        gap = float(every_label_excess(p[None, :], exact[None, :]).max() - bound)
                        maximum_certificate_violation = max(maximum_certificate_violation, gap)
                        require(gap <= 2e-12, "Finite-probe every-label certificate violation")
                        checkpoints_checked += 1
                final = forest.exact(obs)
                require(np.max(np.abs(final - exact)) <= 2e-12, "Complete selected-coordinate output mismatch")
            require(maximum_difference <= 2e-12, "Native/compiled probe probability mismatch")
            require(maximum_interval_gap <= 2e-12, "Native scores outside leaf bounds")
            results.append({"model": entry["name"], "model_sha256": entry["sha256"],
                            "temperature": temperature, "synthetic_probes": len(x),
                            "certificate_checks": checkpoints_checked,
                            "maximum_native_compiled_probability_difference": maximum_difference,
                            "maximum_score_interval_violation_nonnegative": maximum_interval_gap,
                            "maximum_certificate_violation_nonnegative": maximum_certificate_violation})
    return {"status": "PASS_FINITE_SYNTHETIC_MODEL_CONFORMANCE", "models": results,
            "real_dataset_cases_evaluated": 0, "models_fitted": 0, "latencies_measured": 0,
            "limitation": "Finite synthetic regression probes do not prove all-input floating-point soundness or reproduce empirical timing."}
