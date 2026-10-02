"""Correctness-only check of leaf masks on existing predictor-fitting features."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
OLD = Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['PYTHONPYCACHEPREFIX'] = str(ROOT / 'temp' / 'pycache')
sys.path[:0] = [str(ROOT / 'research'), str(ROOT / 'scripts'),
                str(OLD / 'env_core'), str(OLD / 'env_packages'), r'D:\ptenv\Lib\site-packages']
import joblib
import numpy as np
from certificate_kernel import FrozenForest, probability_intervals, choose_probability
from leaf_mask_kernel import FrozenLeafMasks


def main():
    model = joblib.load(r'D:\MLWork\ASC_ComputeValue_20261002\results\redesign_diagnostic\full_hgb_seed17.joblib')
    widths = np.tile([11, 6], 14)
    groups = np.repeat(np.arange(28), widths)
    slices = np.r_[0, np.cumsum(widths)]
    forest = FrozenForest(model, groups, 28)
    masks_forest = FrozenLeafMasks(forest)
    with np.load(r'D:\MLWork\ASC_ComputeValue_20261002\cache\hydraulic_development_raw_v1.npz', allow_pickle=False) as z:
        X = z['X']
        fitting = z['predictor_fit']
    rng = np.random.default_rng(20261002)
    cases = rng.choice(fitting, 24, replace=False)
    worst = {'logit': 0., 'native_probability': 0., 'interval_enclosure': 0., 'certificate_excess': 0.}
    prefixes = 0
    for case in cases:
        q = model.predict_proba(X[case:case + 1])[0]
        actual_scores = model._raw_predict(X[case:case + 1])[0]
        observed = np.zeros(X.shape[1], np.float32)
        known = np.zeros(28, np.bool_)
        lower, upper = forest.static_lower.copy(), forest.static_upper.copy()
        masks = masks_forest.initial.copy()
        for block in rng.permutation(forest.used):
            observed[slices[block]:slices[block + 1]] = X[case, slices[block]:slices[block + 1]]
            known[block] = True
            forest.update(int(block), observed, known, lower, upper)
            masks_forest.update(int(block), observed, masks)
            ref_lo, ref_hi, _, _, ref_collapsed = forest.intervals(lower, upper)
            lo, hi, collapsed = masks_forest.score_intervals(masks)
            worst['logit'] = max(worst['logit'], float(np.max(np.abs(lo - ref_lo))), float(np.max(np.abs(hi - ref_hi))))
            if collapsed != ref_collapsed:
                raise AssertionError('Collapse predicates differ')
            if np.any(actual_scores < lo) or np.any(actual_scores > hi):
                raise AssertionError('Complete scores excluded')
            l, u = probability_intervals(lo, hi)
            worst['interval_enclosure'] = max(worst['interval_enclosure'], float(max(np.max(l - q), np.max(q - u), 0.)))
            p, B = choose_probability(l, u, 0)
            actual_excess = .5 * (p @ p - q @ q) - (p - q)
            worst['certificate_excess'] = max(worst['certificate_excess'], float(np.max(actual_excess - B)))
            prefixes += 1
        worst['native_probability'] = max(worst['native_probability'], float(np.max(np.abs(forest.exact(observed) - q))))
    if any(v > 2e-12 for v in worst.values()):
        raise AssertionError(worst)
    bit_cases = [1 << i for i in range(63)] + [(1 << 63) - 1]
    from leaf_mask_kernel import highest_bit
    for x in bit_cases:
        if highest_bit(np.uint64(x)) != x.bit_length() - 1:
            raise AssertionError('Bit-index error')
    out = {'status': 'PASS', 'fitting_cases': len(cases), 'tested_prefixes': prefixes,
           'maximum_discrepancies': worst, 'labels_or_nonfitting_partitions_used': 0,
           'principle': 'Candidate masks enforce exactly known path split constraints; opposite-subtree masks leave outside leaves unchanged.',
           'scope': 'Numerical correctness development check, no timing or new fitting.'}
    dest = ROOT / 'research' / 'LEAF_MASK_CORRECTNESS.json'
    dest.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
