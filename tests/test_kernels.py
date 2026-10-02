"""Small deterministic numerical probes, not additional empirical study data.

Finite probes do not establish all-input floating-point soundness. Mathematical
limits and empirical diagnostics remain those stated in the manuscript.
"""
from types import SimpleNamespace
import unittest
import numpy as np

from asc_certified import certificate_kernel as ck
from asc_certified import leaf_mask_kernel as lm


class CertificateTests(unittest.TestCase):
    def test_simplex_projection_kkt_and_translation(self):
        for a in (np.array([.2, .3, -.1]), np.array([3., 1., -1.]), np.array([-8., -8., -8.])):
            p = ck.simplex_projection(a)
            self.assertGreaterEqual(float(p.min()), 0.)
            self.assertAlmostEqual(float(p.sum()), 1., places=14)
            active = p > 1e-13
            theta = float(np.mean(a[active] - p[active]))
            np.testing.assert_allclose(a[active] - p[active], theta, atol=2e-13)
            self.assertTrue(np.all(a[~active] <= theta + 2e-13))
            np.testing.assert_allclose(p, ck.simplex_projection(a + 5.), atol=2e-13)

    def test_uninformative_simplex_has_known_minimax_value(self):
        for k in (3, 4, 6):
            p, bound = ck.choose_probability(np.zeros(k), np.ones(k), 0)
            np.testing.assert_allclose(p, np.ones(k) / k, atol=1e-12)
            expected = .5 * (1. - 1. / k)
            self.assertGreaterEqual(bound, expected)
            self.assertLess(bound - expected, 3e-10)

    def test_point_uncertainty_nearly_exact_and_every_label_bound(self):
        q = np.array([.1, .3, .6])
        p, bound = ck.choose_probability(q, q, 0)
        np.testing.assert_allclose(p, q, atol=2e-11)
        excess = .5 * (p @ p - q @ q) - p + q
        self.assertLessEqual(float(excess.max()), bound)
        self.assertLess(bound, 1e-9)

    def test_conservative_bound_covers_feasible_probability_probes(self):
        rng = np.random.default_rng(9102026)
        for k in (3, 4, 6):
            feasible = rng.dirichlet(np.ones(k), size=160)
            l, u = feasible.min(0), feasible.max(0)
            for variant in (0, 1, 2):
                p, bound = ck.choose_probability(l, u, variant)
                excess = .5 * (p @ p - np.square(feasible).sum(-1))[:, None] - p + feasible
                self.assertLessEqual(float(excess.max()), bound + 2e-12)
                self.assertTrue(np.isfinite(p).all())
                self.assertAlmostEqual(float(p.sum()), 1., places=13)

    def test_softmax_intervals_contain_completed_scores(self):
        rng = np.random.default_rng(9102027)
        for scale in (1., 1000.):
            lo = np.array([-2., .2, 1.1]) * scale
            hi = lo + np.array([.8, 1., .4]) * scale
            l, u = ck.probability_intervals(lo, hi)
            for row in rng.uniform(lo, hi, size=(40, 3)):
                p = ck.stable_softmax(row)
                self.assertTrue(np.all(p >= l - 2e-12) and np.all(p <= u + 2e-12))


class LeafMaskTests(unittest.TestCase):
    @staticmethod
    def tiny_forest():
        # A two-feature tree: feature0 chooses leaf1 or feature1 branch2.
        return SimpleNamespace(feature=np.array([0, 0, 1, 0, 0]),
            roots=np.array([0]), leaf=np.array([False, True, False, True, True]),
            left=np.array([1, 0, 3, 0, 0]), right=np.array([2, 0, 4, 0, 0]),
            threshold=np.array([0., 0., 1., 0., 0.]),
            missing_left=np.array([True, False, False, False, False]),
            group=np.array([0, 0, 1, 0, 0]), values=np.array([0., -2., 0., .5, 3.]),
            gains=np.ones(2), classes=np.array([1]), baseline=np.zeros(3), root_abs=np.array([3.]))

    def test_partial_masks_contain_only_feasible_leaf_values(self):
        masks = lm.FrozenLeafMasks(self.tiny_forest())
        state = masks.initial.copy()
        obs = np.array([2., .2])
        lo, hi, collapsed = masks.score_intervals(state)
        self.assertFalse(collapsed)
        self.assertLessEqual(lo[1], -2.)
        self.assertGreaterEqual(hi[1], 3.)
        masks.update(0, obs, state)
        lo, hi, collapsed = masks.score_intervals(state)
        self.assertFalse(collapsed)
        self.assertGreater(lo[1], .49)
        self.assertLess(hi[1], 3.01)
        masks.update(1, obs, state)
        lo, hi, collapsed = masks.score_intervals(state)
        self.assertTrue(collapsed)
        self.assertLessEqual(lo[1], .5)
        self.assertGreaterEqual(hi[1], .5)

    def test_missing_branch_and_observation_order(self):
        masks = lm.FrozenLeafMasks(self.tiny_forest())
        for obs, target in ((np.array([np.nan, 2.]), -2.), (np.array([2., np.nan]), 3.)):
            finals = []
            for order in ((0, 1), (1, 0)):
                state = masks.initial.copy()
                for block in order:
                    masks.update(block, obs, state)
                lo, hi, collapsed = masks.score_intervals(state)
                self.assertTrue(collapsed)
                self.assertLessEqual(lo[1], target)
                self.assertGreaterEqual(hi[1], target)
                finals.append(state)
            np.testing.assert_array_equal(*finals)

    def test_empty_mask_fails_closed(self):
        masks = lm.FrozenLeafMasks(self.tiny_forest())
        lo, hi, collapsed = masks.score_intervals(np.zeros(1, dtype=np.uint64))
        self.assertTrue(np.isneginf(lo).all() and np.isposinf(hi).all())
        self.assertFalse(collapsed)
