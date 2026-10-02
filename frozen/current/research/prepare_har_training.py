"""Prepare official HAR training windows only, after the root feasibility gate.

This file does not download, fit a predictor, inspect test outcomes, or create a
test cache. The caller authorizes execution only after the recorded gate passes.
An inner ZIP may be read as an opaque container; no test member is opened.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

WORK = Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
OLD = Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
# Use the installed D-drive NumPy dependencies; do not import the Torch wrapper.
sys.path.insert(0, str(OLD / 'env_packages'))
sys.path.insert(0, str(OLD / 'env_core'))
import numpy as np

SEEDS = (17, 29, 43)
FS_HZ = 50.0
WINDOW_SAMPLES = 128
WINDOW_SECONDS = WINDOW_SAMPLES / FS_HZ
SIGNALS = tuple(f'{kind}_{axis}' for kind in ('total_acc', 'body_acc', 'body_gyro')
                for axis in ('x', 'y', 'z'))
STATS = ('mean', 'std', 'min', 'max', 'q25', 'q50', 'q75', 'first', 'last',
         'slope_per_second', 'rms')
SPECTRAL = ('log_power', 'centroid_hz', 'spread_hz', 'spectral_entropy',
            'dominant_hz', 'low_third_power_fraction')
WIDTHS = np.tile(np.asarray([11, 6], dtype=np.int64), 9)
OFFSETS = np.r_[0, np.cumsum(WIDTHS)]
GROUP_INDEX = np.repeat(np.arange(18), WIDTHS)
FEATURE_NAMES = tuple(f'{s}__{f}' for s in SIGNALS for f in (*STATS, *SPECTRAL))
TRAIN_SUFFIXES = ('train/subject_train.txt', 'train/y_train.txt',
                  *(f'train/Inertial Signals/{s}_train.txt' for s in SIGNALS))
ACTIVITIES = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
              4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_member(name: str) -> str:
    return name.replace('\\', '/').strip('/')


def find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    suffix = normalized_member(suffix)
    matches = [name for name in archive.namelist()
               if normalized_member(name) == suffix
               or normalized_member(name).endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one training member {suffix!r}, got {len(matches)}')
    return matches[0]


@contextmanager
def training_archive(path: Path, audit: dict):
    """Handle either the direct dataset ZIP or its official outer ZIP wrapper.

    Listing ZIP members reads metadata only. For a nested release, the inner ZIP
    is materialized in memory as compressed container bytes. Its test members
    remain unopened and are never decoded, parsed, or written to a cache.
    """
    with zipfile.ZipFile(path, 'r') as outer:
        names = [normalized_member(n) for n in outer.namelist()]
        direct = any(n == TRAIN_SUFFIXES[0] or n.endswith('/' + TRAIN_SUFFIXES[0])
                     for n in names)
        if direct:
            audit['layout'] = 'direct_training_tree'
            yield outer
            return
        candidates = [n for n in outer.namelist()
                      if normalized_member(n).split('/')[-1].lower() == 'uci har dataset.zip']
        if len(candidates) != 1:
            raise ValueError('No direct training tree or unique UCI HAR Dataset.zip container')
        container = candidates[0]
        audit['layout'] = 'nested_dataset_zip'
        audit['opaque_zip_container_opened'] = container
        with io.BytesIO(outer.read(container)) as buffer:
            with zipfile.ZipFile(buffer, 'r') as inner:
                yield inner


def load_training_signals(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    audit = {'opened_training_members': [], 'opened_test_members': [],
             'test_outcome_files_parsed': 0, 'test_signal_files_parsed': 0}
    with training_archive(path, audit) as archive:
        def read(suffix: str, dtype):
            if suffix not in TRAIN_SUFFIXES:
                raise AssertionError('Read outside the explicit training whitelist')
            member = find_member(archive, suffix)
            if 'test' in normalized_member(member).split('/'):
                raise AssertionError('Test member access is forbidden')
            audit['opened_training_members'].append(member)
            with archive.open(member, 'r') as stream:
                return np.loadtxt(stream, dtype=dtype)
        subjects = np.atleast_1d(read(TRAIN_SUFFIXES[0], np.int64))
        original_labels = np.atleast_1d(read(TRAIN_SUFFIXES[1], np.int64))
        channels = [np.atleast_2d(read(suffix, np.float64)) for suffix in TRAIN_SUFFIXES[2:]]
    n = len(subjects)
    if n != 7352 or original_labels.shape != (n,):
        raise ValueError('Unexpected official HAR training release size')
    if any(a.shape != (n, WINDOW_SAMPLES) for a in channels):
        raise ValueError('Training signal shape must be (7352, 128) for every channel')
    raw = np.stack(channels, axis=1)
    if not np.isfinite(raw).all():
        raise ValueError('Training signals contain nonfinite values')
    if not np.isin(original_labels, list(ACTIVITIES)).all():
        raise ValueError('Unknown HAR activity encoding')
    if not np.isin(subjects, np.arange(1, 31)).all() or len(np.unique(subjects)) != 21:
        raise ValueError('Expected 21 official training subjects with IDs in 1..30')
    labels = original_labels - 1
    assert len(audit['opened_training_members']) == 11 and not audit['opened_test_members']
    return raw, labels, subjects, audit


class HARBlocks:
    """The same statistic/spectral definitions as RawBlocks, with HAR time units.

    raw_train is (case, signal, sample); even blocks are eleven statistics and
    odd blocks are six spectral outputs. Each call executes its own operator.
    """
    def __init__(self, raw_train, center=None, scale=None):
        self.raw = np.asarray(raw_train)
        if self.raw.ndim != 3 or self.raw.shape[1:] != (9, WINDOW_SAMPLES):
            raise ValueError('Expected raw windows of shape (n, 9, 128)')
        self.center = np.zeros(153, np.float32) if center is None else np.asarray(center, np.float32)
        self.scale = np.ones(153, np.float32) if scale is None else np.asarray(scale, np.float32)
        if self.center.shape != (153,) or self.scale.shape != (153,):
            raise ValueError('Scaler arrays must have 153 coordinates')
        if not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError('Invalid fit-only feature scaler')
        self.t = (np.arange(WINDOW_SAMPLES) - (WINDOW_SAMPLES - 1) / 2) / FS_HZ
        self.tt = float(self.t @ self.t)
        self.freq = np.fft.rfftfreq(WINDOW_SAMPLES, d=1 / FS_HZ)

    def values(self, case: int, block: int) -> np.ndarray:
        if not 0 <= int(block) < 18:
            raise IndexError('HAR block must be in 0..17')
        x = self.raw[int(case), int(block) // 2]
        if int(block) % 2 == 0:
            q = np.quantile(x, [.25, .5, .75])
            out = [x.mean(), x.std(), x.min(), x.max(), *q, x[0], x[-1],
                   float(x @ self.t) / self.tt, np.sqrt(np.mean(x * x))]
        else:
            power = np.abs(np.fft.rfft(x - x.mean())) ** 2
            total = float(power.sum())
            norm = power / max(total, 1e-30)
            centroid = float(norm @ self.freq)
            spread = np.sqrt(float(norm @ ((self.freq - centroid) ** 2)))
            entropy = float(-(norm * np.log(np.maximum(norm, 1e-30))).sum() / np.log(len(norm)))
            out = [np.log1p(total / WINDOW_SAMPLES ** 2), centroid, spread, entropy,
                   self.freq[int(np.argmax(power))], norm[self.freq <= self.freq[-1] / 3].sum()]
        return np.asarray(out, dtype=np.float32)

    def reveal(self, case: int, block: int, obs: np.ndarray, mask: np.ndarray):
        if mask[block] > .5:
            raise AssertionError('A HAR block must not be computed twice')
        sl = slice(int(OFFSETS[block]), int(OFFSETS[block + 1]))
        obs[sl] = (self.values(case, block) - self.center[sl]) / self.scale[sl]
        mask[block] = 1


def whole_subject_splits(subjects: np.ndarray) -> tuple[dict, dict]:
    ids = np.sort(np.unique(subjects))
    if len(ids) != 21:
        raise ValueError('Subject splitting expects all 21 official training subjects')
    arrays, record = {}, {}
    for seed in SEEDS:
        shuffled = np.random.default_rng(seed).permutation(ids)
        fit_subjects, development_subjects = shuffled[:17], shuffled[17:]
        fit = np.flatnonzero(np.isin(subjects, fit_subjects))
        development = np.flatnonzero(np.isin(subjects, development_subjects))
        assert not np.intersect1d(fit_subjects, development_subjects).size
        assert not np.intersect1d(fit, development).size
        assert np.array_equal(np.sort(np.r_[fit, development]), np.arange(len(subjects)))
        # All windows of a subject have one owner, including overlapping windows.
        for subject in ids:
            rows = np.flatnonzero(subjects == subject)
            assert np.isin(rows, fit).all() or np.isin(rows, development).all()
        arrays[f'fit_indices_seed{seed}'] = fit
        arrays[f'development_indices_seed{seed}'] = development
        arrays[f'fit_subjects_seed{seed}'] = fit_subjects
        arrays[f'development_subjects_seed{seed}'] = development_subjects
        record[str(seed)] = {'shuffle_seed': seed, 'shuffled_subject_ids': shuffled.tolist(),
                            'fit_subject_ids': fit_subjects.tolist(),
                            'development_subject_ids': development_subjects.tolist(),
                            'fit_windows': len(fit), 'development_windows': len(development),
                            'no_subject_crosses_fit_development': True}
    return arrays, record


def prepare(manifest_path: Path, output: Path | None = None) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    source = Path(manifest['path']).resolve(strict=True)
    if source.drive.upper() != 'F:' or not source.is_file():
        raise ValueError('This loader requires the manifest-authorized F-drive source ZIP')
    if manifest.get('sha256') is None or digest(source) != manifest['sha256']:
        raise ValueError('Source ZIP does not match its immutable download manifest')
    if shutil.disk_usage('D:\\').free < 2_000_000_000:
        raise RuntimeError('Insufficient D-drive space for training-only work')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = WORK / 'cache' / f'har_train_only__{tag}.npz' if output is None else output.resolve()
    if output.drive.upper() != 'D:' or output.suffix.lower() != '.npz':
        raise ValueError('Prepared cache must be an absolute D-drive NPZ path')
    if output.exists():
        raise FileExistsError('Existing caches are not overwritten; choose a fresh output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, labels, subjects, archive_audit = load_training_signals(source)
    split_arrays, splits = whole_subject_splits(subjects)
    operators = HARBlocks(raw)
    features = np.empty((len(raw), 153), np.float32)
    for case in range(len(raw)):
        for block in range(18):
            features[case, OFFSETS[block]:OFFSETS[block + 1]] = operators.values(case, block)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite training features')
    scaler = {}
    for seed in SEEDS:
        fitting = features[split_arrays[f'fit_indices_seed{seed}']].astype(np.float64)
        center = fitting.mean(0).astype(np.float32)
        scale = fitting.std(0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        scaler[f'scaler_center_seed{seed}'] = center
        scaler[f'scaler_scale_seed{seed}'] = scale
    payload = dict(raw_train=raw, features_train=features, labels_train=labels,
                   subjects=subjects, original_train_row=np.arange(len(raw)),
                   signal_names=np.asarray(SIGNALS), feature_names=np.asarray(FEATURE_NAMES),
                   block_widths=WIDTHS, block_offsets=OFFSETS, group_index=GROUP_INDEX,
                   sampling_hz=np.asarray(FS_HZ), window_seconds=np.asarray(WINDOW_SECONDS),
                   splits=np.asarray(json.dumps(splits, sort_keys=True)), **split_arrays, **scaler)
    # Exclusive creation also protects against a concurrent filename collision.
    with output.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_manifest': str(manifest_path.resolve()),
              'source_manifest_sha256': digest(manifest_path), 'source_zip': str(source),
              'source_zip_sha256': manifest['sha256'], 'cache': str(output),
              'cache_sha256': digest(output), 'preparation_script_sha256': digest(Path(__file__)),
              'raw_train_shape': list(raw.shape), 'features_train_shape': list(features.shape),
              'labels_shape': list(labels.shape), 'training_subject_ids': np.unique(subjects).tolist(),
              'signals': list(SIGNALS), 'sampling_hz': FS_HZ, 'window_samples': WINDOW_SAMPLES,
              'window_seconds': WINDOW_SECONDS, 'sample_time_span_seconds': (WINDOW_SAMPLES - 1) / FS_HZ,
              'statistics': list(STATS), 'spectral': list(SPECTRAL), 'blocks': 18,
              'label_map': {str(k): {'encoded': k - 1, 'name': v} for k, v in ACTIVITIES.items()},
              'scaler_fit': 'Each seed uses only its 17 fitting subjects; no development/test windows.',
              'split_rule': 'Sorted training subject IDs, seeded RNG permutation, first17 fit/last4 development; no outcome-based selection.',
              'splits': splits, 'archive_audit': archive_audit,
              'outer_test_disjointness_checked': False,
              'outer_test_note': 'No test subject, label, feature, or signal member was opened. Formal outer verification occurs only after method freeze.',
              'measurement_boundary': 'Official prefiltered/windowed inertial arrays; released signal window already in host memory.',
              'predictor_fitted': False, 'test_cache_created': False}
    report_path = WORK / 'datasets' / f'har_training_preparation_manifest__{tag}.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return output, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='D-drive manifest from research/download_har.py; contains path and sha256')
    parser.add_argument('--output', type=Path, default=None, help='Optional fresh absolute D-drive NPZ filename')
    args = parser.parse_args()
    cache, report = prepare(args.manifest, args.output)
    print(json.dumps({'cache': str(cache), 'preparation_manifest': str(report),
                      'test_members_opened': 0, 'predictor_fitted': False}), flush=True)


if __name__ == '__main__':
    main()
