"""Prepare held-out HAR windows only after an explicit pre-test freeze lock.

This module is not executed during loader preparation. It fits no preprocessing
or predictor and opens only the eleven test-content members after validating
the caller's frozen lock. Raw sources remain read-only on F; output stays on D.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

from prepare_har_training import (
    WORK, SEEDS, FS_HZ, WINDOW_SAMPLES, WINDOW_SECONDS, SIGNALS, STATS, SPECTRAL,
    WIDTHS, OFFSETS, GROUP_INDEX, FEATURE_NAMES, ACTIVITIES, HARBlocks,
    digest, normalized_member, find_member, training_archive, np,
)

REQUIRED_STAGE = 'FROZEN_BEFORE_TEST_ACCESS'
TEST_SUFFIXES = ('test/subject_test.txt', 'test/y_test.txt',
                 *(f'test/Inertial Signals/{signal}_test.txt' for signal in SIGNALS))


def read_test_lock(path: Path) -> tuple[dict, str]:
    """This check precedes any source-archive access or output creation."""
    if not path.is_file():
        raise FileNotFoundError('The pre-test freeze lock does not exist; test access refused')
    lock = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(lock, dict) or lock.get('stage') != REQUIRED_STAGE:
        raise PermissionError(f'Test access requires stage={REQUIRED_STAGE!r}')
    return lock, digest(path)


def read_training_ownership(cache: Path) -> tuple[np.ndarray, dict, str]:
    if cache.drive.upper() != 'D:' or not cache.is_file():
        raise ValueError('Training ownership must come from the existing D-drive training cache')
    sha = digest(cache)
    with np.load(cache, allow_pickle=False) as saved:
        subjects = saved['subjects']
        if subjects.shape != (7352,) or len(np.unique(subjects)) != 21:
            raise ValueError('Expected 7352 training windows from 21 subjects')
        if not np.isin(subjects, np.arange(1, 31)).all():
            raise ValueError('Unexpected training subject IDs')
        if not np.array_equal(saved['group_index'], GROUP_INDEX):
            raise ValueError('Training/test feature schema mismatch')
        if not np.array_equal(saved['feature_names'], np.asarray(FEATURE_NAMES)):
            raise ValueError('Training/test operator-coordinate mismatch')
        scaler_keys = {}
        for seed in SEEDS:
            center_key, scale_key = f'scaler_center_seed{seed}', f'scaler_scale_seed{seed}'
            center, scale = saved[center_key], saved[scale_key]
            if center.shape != (153,) or scale.shape != (153,):
                raise ValueError('Missing or incompatible fit-only training scaler')
            if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Invalid saved training scaler')
            scaler_keys[str(seed)] = {'center_key': center_key, 'scale_key': scale_key}
    return np.unique(subjects), scaler_keys, sha


def load_test_signals(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    audit = {'opened_test_members': [], 'opened_training_members': [],
             'official_561_feature_members_opened': [], 'test_outcome_files_parsed': 1,
             'test_signal_files_parsed': 9}
    with training_archive(source, audit) as archive:
        def read(suffix: str, dtype):
            if suffix not in TEST_SUFFIXES:
                raise AssertionError('Content access outside the eleven-member test whitelist')
            member = find_member(archive, suffix)
            if 'train' in normalized_member(member).split('/'):
                raise AssertionError('Training archive content must not be reopened')
            audit['opened_test_members'].append(member)
            with archive.open(member, 'r') as stream:
                return np.loadtxt(stream, dtype=dtype)
        subjects = np.atleast_1d(read(TEST_SUFFIXES[0], np.int64))
        original_labels = np.atleast_1d(read(TEST_SUFFIXES[1], np.int64))
        channels = [np.atleast_2d(read(suffix, np.float64)) for suffix in TEST_SUFFIXES[2:]]
    if subjects.shape != (2947,) or original_labels.shape != (2947,):
        raise ValueError('Unexpected official HAR test release size')
    if any(channel.shape != (2947, WINDOW_SAMPLES) for channel in channels):
        raise ValueError('Every official test channel must have 2947 windows of 128 readings')
    if len(np.unique(subjects)) != 9 or not np.isin(subjects, np.arange(1, 31)).all():
        raise ValueError('Expected nine held-out subjects with IDs in 1..30')
    if not np.array_equal(np.unique(original_labels), np.arange(1, 7)):
        raise ValueError('The test release must contain all six original activity labels')
    raw = np.stack(channels, axis=1)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite test inertial values')
    assert len(audit['opened_test_members']) == 11
    assert not audit['opened_training_members'] and not audit['official_561_feature_members_opened']
    return raw, original_labels - 1, subjects, audit


def prepare(manifest_path: Path, test_lock_path: Path, train_cache: Path,
            output: Path | None = None) -> tuple[Path, Path]:
    lock, lock_sha = read_test_lock(test_lock_path)
    # The training ownership/scalers are inspected before opening any test data.
    train_cache = train_cache.resolve(strict=True)
    train_subjects, scaler_keys, train_sha = read_training_ownership(train_cache)
    if 'train_cache_sha256' in lock and lock['train_cache_sha256'] != train_sha:
        raise ValueError('Training cache differs from the frozen test lock')
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    source = Path(manifest['path']).resolve(strict=True)
    if source.drive.upper() != 'F:' or not source.is_file():
        raise ValueError('The manifest must authorize an immutable F-drive source ZIP')
    if manifest.get('sha256') is None or digest(source) != manifest['sha256']:
        raise ValueError('Source ZIP checksum does not match the download manifest')
    if 'source_sha256' in lock and lock['source_sha256'] != manifest['sha256']:
        raise ValueError('Source archive differs from the frozen test lock')
    if 'test_loader_sha256' in lock and lock['test_loader_sha256'] != digest(Path(__file__)):
        raise ValueError('Test loader differs from the frozen test lock')
    if shutil.disk_usage('D:\\').free < 2_000_000_000:
        raise RuntimeError('Insufficient D-drive space for test preparation')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = WORK / 'cache' / f'har_test_frozen__{tag}.npz' if output is None else output.resolve()
    if output.drive.upper() != 'D:' or output.suffix.lower() != '.npz':
        raise ValueError('Test caches must use a fresh absolute D-drive NPZ filename')
    if output.exists():
        raise FileExistsError('An existing cache is never overwritten')
    # Recheck the lock immediately before releasing test content, rather than
    # relying on a path that may have changed since the initial validation.
    _, current_lock_sha = read_test_lock(test_lock_path)
    if current_lock_sha != lock_sha:
        raise RuntimeError('The test lock changed before test access')
    raw, labels, subjects, archive_audit = load_test_signals(source)
    test_subjects = np.unique(subjects)
    if np.intersect1d(train_subjects, test_subjects).size:
        raise AssertionError('Official training/test subject leakage')
    if not np.array_equal(np.sort(np.r_[train_subjects, test_subjects]), np.arange(1, 31)):
        raise AssertionError('Official outer split does not cover the 30 distinct subjects')
    features = np.empty((len(raw), 153), np.float32)
    operators = HARBlocks(raw)
    for case in range(len(raw)):
        for block in range(18):
            features[case, OFFSETS[block]:OFFSETS[block + 1]] = operators.values(case, block)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite test features')
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(raw_test=raw, features_test=features, labels_test=labels, subjects=subjects,
                   original_test_row=np.arange(len(raw)), signal_names=np.asarray(SIGNALS),
                   feature_names=np.asarray(FEATURE_NAMES), block_widths=WIDTHS,
                   block_offsets=OFFSETS, group_index=GROUP_INDEX,
                   sampling_hz=np.asarray(FS_HZ), window_seconds=np.asarray(WINDOW_SECONDS),
                   train_cache_sha256=np.asarray(train_sha), test_lock_sha256=np.asarray(lock_sha))
    with output.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'test_lock': str(test_lock_path.resolve()), 'test_lock_sha256': lock_sha,
              'test_lock_stage': lock['stage'], 'source_manifest': str(manifest_path),
              'source_manifest_sha256': digest(manifest_path), 'source_zip': str(source),
              'source_zip_sha256': manifest['sha256'], 'train_cache': str(train_cache),
              'train_cache_sha256': train_sha, 'cache': str(output), 'cache_sha256': digest(output),
              'test_loader_sha256': digest(Path(__file__)),
              'training_operator_script_sha256': digest(Path(__file__).with_name('prepare_har_training.py')),
              'raw_test_shape': list(raw.shape), 'features_test_shape': list(features.shape),
              'train_subject_ids': train_subjects.tolist(), 'test_subject_ids': test_subjects.tolist(),
              'outer_subject_overlap': 0, 'label_counts': np.bincount(labels, minlength=6).tolist(),
              'label_map': {str(k): {'encoded': k - 1, 'name': v} for k, v in ACTIVITIES.items()},
              'signals': list(SIGNALS), 'sampling_hz': FS_HZ, 'window_samples': WINDOW_SAMPLES,
              'window_seconds': WINDOW_SECONDS, 'statistics': list(STATS), 'spectral': list(SPECTRAL),
              'archive_audit': archive_audit, 'official_561_features_unopened': True,
              'test_features_standardized': False, 'preprocessing_fitted': False,
              'predictor_fitted': False, 'future_scaler_source': str(train_cache),
              'future_scaler_keys': scaler_keys,
              'evaluation_rule': 'For each seed, reuse its frozen fit-only training center/scale and classifier. No test-based order, tolerance, model, or preprocessing selection.',
              'measurement_boundary': 'Author-prefiltered/windowed inertial arrays; online software timing starts with released window in host memory.'}
    report_path = WORK / 'datasets' / f'har_test_preparation_manifest__{tag}.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return output, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Immutable official source download manifest')
    parser.add_argument('--test-lock', type=Path, required=True, help='Root-owned frozen pre-test-access lock')
    parser.add_argument('--train-cache', type=Path, required=True, help='Existing training-only NPZ for outer ownership and frozen scaler references')
    parser.add_argument('--output', type=Path, default=None, help='Optional fresh absolute D-drive test-cache filename')
    args = parser.parse_args()
    cache, report = prepare(args.manifest, args.test_lock, args.train_cache, args.output)
    print(json.dumps({'cache': str(cache), 'preparation_manifest': str(report),
                      'opened_test_members': 11, 'official_561_features_opened': 0,
                      'preprocessing_fitted': False, 'predictor_fitted': False}), flush=True)


if __name__ == '__main__':
    main()
