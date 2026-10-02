"""Synthetic signals verify original operator units and observation discipline."""
import unittest
import numpy as np

from asc_certified.har_features import HARBlocks, OFFSETS, whole_subject_splits
from asc_certified.hydraulic_features import RawBlocks


class FeatureTests(unittest.TestCase):
    def test_har_constant_and_frequency_units(self):
        raw = np.ones((1, 9, 128), dtype=np.float64) * 2.
        blocks = HARBlocks(raw)
        stats, spectral = blocks.values(0, 0), blocks.values(0, 1)
        np.testing.assert_allclose(stats[:10], [2., 0., 2., 2., 2., 2., 2., 2., 2., 0.], atol=1e-6)
        self.assertAlmostEqual(float(stats[10]), 2.)
        np.testing.assert_allclose(spectral, np.zeros(6), atol=1e-6)
        raw[0, 0] = np.sin(2 * np.pi * 8 * np.arange(128) / 128)
        frequency = HARBlocks(raw).values(0, 1)
        self.assertAlmostEqual(float(frequency[4]), 8 * 50 / 128, places=6)
        self.assertAlmostEqual(float(frequency[1]), 8 * 50 / 128, places=5)

    def test_selected_block_scaling_and_duplicate_rejection(self):
        raw = np.ones((1, 9, 128), dtype=np.float64) * 2.
        blocks = HARBlocks(raw, center=np.ones(153), scale=np.ones(153) * 2)
        obs, mask = np.zeros(153), np.zeros(18, dtype=bool)
        blocks.reveal(0, 0, obs, mask)
        np.testing.assert_allclose(obs[:11], (blocks.values(0, 0) - 1) / 2)
        np.testing.assert_array_equal(obs[OFFSETS[1]:], 0.)
        with self.assertRaises(AssertionError):
            blocks.reveal(0, 0, obs, mask)

    def test_all_windows_of_subject_keep_one_owner(self):
        subjects = np.repeat(np.arange(1, 22), 5)
        arrays, record = whole_subject_splits(subjects)
        for seed in (17, 29, 43):
            fit = arrays[f"fit_indices_seed{seed}"]
            dev = arrays[f"development_indices_seed{seed}"]
            self.assertEqual(len(np.unique(subjects[fit])), 17)
            self.assertEqual(len(np.unique(subjects[dev])), 4)
            self.assertFalse(np.intersect1d(subjects[fit], subjects[dev]).size)
            self.assertTrue(record[str(seed)]["no_subject_crosses_fit_development"])

    def test_hydraulic_frequency_is_per_60_second_cycle(self):
        samples = 600
        wave = np.sin(2 * np.pi * 6 * np.arange(samples) / samples)[None, :]
        blocks = RawBlocks([wave.copy() for _ in range(14)])
        spectral = blocks.values(0, 1)
        self.assertAlmostEqual(float(spectral[4]), 6 / 60, places=7)
        self.assertAlmostEqual(float(spectral[1]), 6 / 60, places=7)
