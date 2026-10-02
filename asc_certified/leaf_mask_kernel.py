"""Candidate-leaf mask partial evaluator; QuickScorer-inspired control kernel.

This is a runtime optimization, not a claim of a new tree evaluation principle.
Only previously observed features are supplied to update(). Supports numeric
multiclass forests with <=63 leaves per tree. Validation remains mandatory.
"""
from .config import configure_runtime
configure_runtime()
import numpy as np
from numba import njit


@njit(cache=True)
def highest_bit(x):
    i = 0
    if x >= 4294967296:
        x >>= 32
        i += 32
    if x >= 65536:
        x >>= 16
        i += 16
    if x >= 256:
        x >>= 8
        i += 8
    if x >= 16:
        x >>= 4
        i += 4
    if x >= 4:
        x >>= 2
        i += 2
    if x >= 2:
        i += 1
    return i


@njit(cache=True)
def update_masks(observed, active, start, stop, feature, threshold, missing_left,
                 node_tree, wrong_if_left, wrong_if_right, masks):
    for i in range(start, stop):
        n = active[i]
        v = observed[feature[n]]
        goes_left = missing_left[n] if np.isnan(v) else v <= threshold[n]
        wrong = wrong_if_left[n] if goes_left else wrong_if_right[n]
        t = node_tree[n]
        masks[t] &= ~wrong
        if masks[t] == 0:
            return False
    return True


@njit(cache=True)
def mask_intervals(masks, leaf_values, classes, baseline, root_abs):
    lo = baseline.copy()
    hi = baseline.copy()
    absolute = np.abs(baseline.copy())
    counts = np.zeros(len(baseline), np.int64)
    collapsed = True
    for t in range(len(masks)):
        m = masks[t]
        if m == 0:
            # A correct update on a real completed input cannot empty a tree.
            return np.full(len(baseline), -np.inf), np.full(len(baseline), np.inf), False
        low_bit = m & (~m + np.uint64(1))
        low = leaf_values[t, highest_bit(low_bit)]
        high = leaf_values[t, highest_bit(m)]
        c = classes[t]
        lo[c] += low
        hi[c] += high
        absolute[c] += root_abs[t]
        counts[c] += 1
        if low != high:
            collapsed = False
    for c in range(len(baseline)):
        neps = counts[c] * np.finfo(np.float64).eps
        guard = 8.0 * neps / (1.0 - neps) * absolute[c] + 1e-12
        lo[c] = np.nextafter(lo[c] - guard, -np.inf)
        hi[c] = np.nextafter(hi[c] + guard, np.inf)
    return lo, hi, collapsed


class FrozenLeafMasks:
    def __init__(self, forest):
        self.forest = forest
        n = len(forest.feature)
        nt = len(forest.roots)
        self.node_tree = np.empty(n, np.int64)
        subtree = np.zeros(n, np.uint64)
        leaf_values = []
        initial = []
        for t in range(nt):
            first = int(forest.roots[t])
            stop = int(forest.roots[t + 1]) if t + 1 < nt else n
            self.node_tree[first:stop] = t
            ids = np.flatnonzero(forest.leaf[first:stop]) + first
            if len(ids) > 63 or len(ids) == 0:
                raise ValueError('Leaf-mask kernel requires between 1 and 63 leaves per tree')
            ids = ids[np.argsort(forest.values[ids], kind='stable')]
            vals = forest.values[ids]
            if not np.isfinite(vals).all():
                raise ValueError('Nonfinite leaf outputs are unsupported')
            leaf_values.append(vals)
            initial.append(np.uint64((1 << len(ids)) - 1))
            for j, node in enumerate(ids):
                subtree[node] = np.uint64(1 << j)
            for node in range(stop - 1, first - 1, -1):
                if not forest.leaf[node]:
                    if not (forest.left[node] > node and forest.right[node] > node):
                        raise ValueError('Forest nodes must be parent-before-children')
                    subtree[node] = subtree[forest.left[node]] | subtree[forest.right[node]]
            if subtree[first] != initial[-1]:
                raise AssertionError('Incomplete tree leaf mapping')
        self.leaf_values = np.zeros((nt, max(map(len, leaf_values))), np.float64)
        for t, vals in enumerate(leaf_values):
            self.leaf_values[t, :len(vals)] = vals
        self.initial = np.asarray(initial, np.uint64)
        self.wrong_if_left = np.zeros(n, np.uint64)
        self.wrong_if_right = np.zeros(n, np.uint64)
        interior = np.flatnonzero(~forest.leaf)
        self.wrong_if_left[interior] = subtree[forest.right[interior]]
        self.wrong_if_right[interior] = subtree[forest.left[interior]]
        active = []
        offsets = [0]
        for block in range(len(forest.gains)):
            active.extend(np.flatnonzero((forest.group == block) & ~forest.leaf).tolist())
            offsets.append(len(active))
        self.active = np.asarray(active, np.int64)
        self.offsets = np.asarray(offsets, np.int64)

    def update(self, block, observed, masks):
        f = self.forest
        ok = update_masks(observed, self.active, self.offsets[block], self.offsets[block + 1],
                          f.feature, f.threshold, f.missing_left, self.node_tree,
                          self.wrong_if_left, self.wrong_if_right, masks)
        if not ok:
            raise AssertionError('Actual observed split outcomes removed every leaf')

    def score_intervals(self, masks):
        f = self.forest
        return mask_intervals(masks, self.leaf_values, f.classes, f.baseline, f.root_abs)
