"""Fixed, recorded HAR train-only full predictor selection; no ZIP/test access."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import uuid
import warnings

ROOT = Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
OLD = Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
TEMP = ROOT / 'temp' / 'har_models'
TEMP.mkdir(parents=True, exist_ok=True)
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ASC_THREADS'):
    os.environ[name] = '1'
for name in ('TEMP', 'TMP', 'TMPDIR', 'JOBLIB_TEMP_FOLDER'):
    os.environ[name] = str(TEMP)
os.environ['PYTHONPYCACHEPREFIX'] = str(ROOT / 'temp' / 'pycache_har_models')
sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
sys.path.insert(0, str(OLD / 'env_packages'))
sys.path.insert(0, str(OLD / 'env_core'))
sys.path.append(r'D:\ptenv\Lib\site-packages')

import joblib
import numpy as np
import scipy
from scipy.special import softmax
import sklearn
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

CACHE = ROOT / 'cache' / 'har_train_only__20261002T051217Z_533d2ce1.npz'
MANIFEST = ROOT / 'datasets' / 'har_training_preparation_manifest__20261002T051217Z_533d2ce1.json'
PLAN = ROOT / 'research' / 'HAR_MODEL_PLAN.md'
SEEDS = (17, 29, 43)
TEMPERATURES = (.75, 1., 1.25, 1.5, 2., 3.)
CLASSES = np.arange(6)
TIE_TOL = 1e-12


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=jsonable), encoding='utf-8')


def candidates(seed):
    result = []
    for iterations in (100, 150):
        for leaves in (15, 31):
            params = dict(max_iter=iterations, max_leaf_nodes=leaves,
                          learning_rate=.1, min_samples_leaf=30,
                          l2_regularization=1., early_stopping=False,
                          random_state=seed)
            result.append((f'hgb_i{iterations}_l{leaves}', 'HGB', params,
                           HistGradientBoostingClassifier))
    result.append(('random_forest', 'RF', dict(n_estimators=400, min_samples_leaf=2,
                    n_jobs=1, random_state=seed), RandomForestClassifier))
    result.append(('extra_trees', 'ET', dict(n_estimators=400, min_samples_leaf=2,
                    n_jobs=1, random_state=seed), ExtraTreesClassifier))
    lr_params = dict(C=1., max_iter=1000, solver='lbfgs', random_state=seed)
    if 'multi_class' in inspect.signature(LogisticRegression).parameters:
        lr_params['multi_class'] = 'multinomial'
    result.append(('logistic', 'LR', lr_params, LogisticRegression))
    result.append(('rbf_svc', 'SVC', dict(C=10., gamma='scale', probability=True,
                    random_state=seed), SVC))
    assert len(result) == 8
    return result


def metrics(y, p):
    if p.shape != (len(y), 6) or not np.isfinite(p).all():
        raise ValueError('Expected finite six-class probability array')
    if np.min(p) < -1e-15 or not np.allclose(p.sum(axis=1), 1., atol=1e-12):
        raise ValueError('Invalid probability simplex')
    target = np.eye(6)[y]
    losses = .5 * np.sum((p - target) ** 2, axis=1)
    return dict(half_brier=float(losses.mean()),
                accuracy=float(np.mean(p.argmax(1) == y)),
                log_loss=float(log_loss(y, p, labels=CLASSES))), losses


def choose_temperature(trials):
    low = min(t['half_brier'] for t in trials)
    tied = [t for t in trials if t['half_brier'] <= low + TIE_TOL]
    return min(tied, key=lambda t: (abs(t['temperature'] - 1.), t['temperature']))


def model_shape(model, family):
    if family == 'HGB':
        trees = [tree for stage in model._predictors for tree in stage]
        nodes = [tree.nodes for tree in trees]
        return dict(n_features=int(model.n_features_in_), iterations=int(model.n_iter_),
                    classes=len(model.classes_), trees=len(trees),
                    nodes=sum(len(a) for a in nodes),
                    leaves=sum(int(a['is_leaf'].sum()) for a in nodes),
                    max_depth=max(int(a['depth'].max()) for a in nodes))
    if family in ('RF', 'ET'):
        return dict(n_features=int(model.n_features_in_), classes=len(model.classes_),
                    trees=len(model.estimators_),
                    nodes=sum(t.tree_.node_count for t in model.estimators_),
                    leaves=sum(t.tree_.n_leaves for t in model.estimators_),
                    max_depth=max(t.tree_.max_depth for t in model.estimators_))
    if family == 'LR':
        return dict(n_features=int(model.n_features_in_), classes=len(model.classes_),
                    coefficient_shape=list(model.coef_.shape),
                    iterations=model.n_iter_.tolist())
    return dict(n_features=int(model.n_features_in_), classes=len(model.classes_),
                support_vectors=int(model.support_.size),
                support_per_class=model.n_support_.tolist(),
                internal_probability_calibration='LIBSVM fitting-data-only five-fold CV, not subject-grouped')


def load_and_audit():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    if CACHE.drive.upper() != 'D:' or digest(CACHE) != manifest['cache_sha256']:
        raise ValueError('Training cache path/hash mismatch')
    if manifest['archive_audit']['opened_test_members'] or manifest['test_cache_created']:
        raise ValueError('Preparation manifest test boundary mismatch')
    with np.load(CACHE, allow_pickle=False) as data:
        # raw_train is intentionally never requested: fitting uses cached complete features.
        x = data['features_train'].copy()
        y = data['labels_train'].copy()
        subjects = data['subjects'].copy()
        metadata = dict(cache_keys=data.files, x_shape=list(x.shape),
                        y_shape=list(y.shape), raw_array_loaded=False)
        arrays = {}
        for seed in SEEDS:
            arrays[seed] = {key: data[f'{key}_seed{seed}'].copy() for key in (
                'fit_indices', 'development_indices', 'fit_subjects',
                'development_subjects', 'scaler_center', 'scaler_scale')}
    if x.shape != (7352, 153) or y.shape != (7352,) or subjects.shape != (7352,):
        raise ValueError('Unexpected cache dimensions')
    if not np.isfinite(x).all() or not np.array_equal(np.unique(y), CLASSES):
        raise ValueError('Invalid cached features/labels')
    splits = {}
    for seed, a in arrays.items():
        fit, dev = a['fit_indices'], a['development_indices']
        fs, ds = a['fit_subjects'], a['development_subjects']
        if len(fs) != 17 or len(ds) != 4 or np.intersect1d(fs, ds).size:
            raise ValueError('Invalid saved subject membership')
        if np.intersect1d(fit, dev).size or not np.array_equal(np.sort(np.r_[fit, dev]), np.arange(7352)):
            raise ValueError('Saved fit/dev rows do not partition training rows')
        if not np.isin(subjects[fit], fs).all() or not np.isin(subjects[dev], ds).all():
            raise ValueError('Subject boundary crossed')
        if sorted(fs.tolist()) != sorted(manifest['splits'][str(seed)]['fit_subject_ids']):
            raise ValueError('Subject membership does not match preparation manifest')
        center = x[fit].astype(np.float64).mean(0).astype(np.float32)
        scale = x[fit].astype(np.float64).std(0).astype(np.float32)
        scale[scale < 1e-8] = 1.
        if not np.array_equal(center, a['scaler_center']) or not np.array_equal(scale, a['scaler_scale']):
            raise ValueError('Saved scaler is not exactly the fitting-row scaler')
        splits[str(seed)] = dict(fit_subjects=fs.tolist(), development_subjects=ds.tolist(),
                                fit_rows=len(fit), development_rows=len(dev),
                                fit_class_counts=np.bincount(y[fit], minlength=6).tolist(),
                                development_class_counts=np.bincount(y[dev], minlength=6).tolist(),
                                saved_scaler_fit_only_verified=True)
    return x, y, subjects, arrays, metadata, splits


def prospective():
    if shutil.disk_usage('D:\\').free < 2_000_000_000:
        raise RuntimeError('D drive has less than 2GB free; no output relocation permitted')
    if not PLAN.is_file():
        raise FileNotFoundError('Prospective HAR_MODEL_PLAN.md must exist before fitting')
    x, y, subjects, arrays, metadata, splits = load_and_audit()
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    result_dir = ROOT / 'results' / 'har_model_selection' / tag
    model_dir = ROOT / 'checkpoints' / 'har_models' / tag
    result_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    registry = dict(created_utc=utc(), status='prospective_not_fitted',
                    expected_model_count=24, cache=str(CACHE), cache_sha256=digest(CACHE),
                    preparation_manifest=str(MANIFEST), preparation_manifest_sha256=digest(MANIFEST),
                    plan=str(PLAN), plan_sha256=digest(PLAN), script_sha256=digest(__file__),
                    seeds=list(SEEDS), temperatures=list(TEMPERATURES),
                    primary_metric='multiclass half-Brier', tie_tolerance=TIE_TOL,
                    hgb_tie_rule='min configured leaves cap, then lower iterations, then lower max_leaf_nodes',
                    temperature_tie_rule='closest to 1, then lower T',
                    candidates=[dict(seed=seed, candidate=cid, family=family, parameters=params)
                                for seed in SEEDS for cid, family, params, _ in candidates(seed)],
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
                    threads=threadpool_info(), d_free_gib=shutil.disk_usage('D:\\').free/2**30,
                    output_dir=str(result_dir), checkpoint_dir=str(model_dir),
                    metadata=metadata, splits=splits,
                    test_data_accessed=False, source_zip_opened=False,
                    fixed_full_feature_input=True, partial_processing_time_used_in_selection=False)
    dump(result_dir / 'prospective_registry.json', registry)
    print(json.dumps(dict(status='prospective_not_fitted', result_dir=str(result_dir),
                          registry=str(result_dir / 'prospective_registry.json'))), flush=True)
    return result_dir, model_dir, registry, x, y, subjects, arrays


def execute(prepared):
    result_dir, model_dir, registry, x, y, subjects, arrays = prepared
    records, selected = [], {}
    started_all = time.perf_counter()
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            a = arrays[seed]
            fit, dev = a['fit_indices'], a['development_indices']
            center, scale = a['scaler_center'], a['scaler_scale']
            xs = (x - center) / scale
            if xs.dtype != np.float32:
                raise AssertionError('Expected saved float32 scaling arithmetic')
            seed_records = []
            for cid, family, params, constructor in candidates(seed):
                record = dict(seed=seed, candidate=cid, family=family, parameters=params,
                              started_utc=utc(), status='started',
                              class_counts=registry['splits'][str(seed)], warnings=[])
                try:
                    model = constructor(**params)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        t = time.perf_counter()
                        model.fit(xs[fit], y[fit])
                        record['fit_seconds'] = time.perf_counter() - t
                        record['warnings'] = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
                    if not np.array_equal(model.classes_, CLASSES):
                        raise AssertionError('All six classes must preserve fixed probability column order')
                    t = time.perf_counter()
                    probability_raw = model.predict_proba(xs[dev])
                    record['development_probability_seconds'] = time.perf_counter() - t
                    logp = np.log(np.clip(probability_raw, 1e-15, 1.))
                    if family == 'HGB':
                        logits = model._raw_predict(xs[dev])
                        equivalence = float(np.max(np.abs(softmax(logits, axis=1) - probability_raw)))
                        record['hgb_raw_logits_native_probability_max_abs_error'] = equivalence
                        if equivalence > 1e-12:
                            raise AssertionError('HGB raw logits do not reproduce native probabilities')
                    else:
                        logits = logp
                    trials = []
                    for temperature in TEMPERATURES:
                        p = softmax(logits / temperature, axis=1)
                        score, _ = metrics(y[dev], p)
                        trials.append(dict(temperature=temperature, **score))
                    chosen = choose_temperature(trials)
                    pchosen = softmax(logits / chosen['temperature'], axis=1)
                    scores, losses = metrics(y[dev], pchosen)
                    raw_scores, raw_losses = metrics(y[dev], probability_raw)
                    record.update(status='success', completed_utc=utc(), temperature_trials=trials,
                                  chosen_temperature=chosen['temperature'], selected_metrics=scores,
                                  raw_metrics=raw_scores, model_shape=model_shape(model, family))
                    model_path = model_dir / f'seed{seed}__{cid}.joblib'
                    bundle = dict(model=model, center=center, scale=scale, temperature=chosen['temperature'],
                                  seed=seed, candidate=cid, family=family, parameters=params,
                                  feature_dimension=153, class_order=CLASSES, fit_indices=fit,
                                  development_indices=dev, fit_subjects=a['fit_subjects'],
                                  development_subjects=a['development_subjects'],
                                  cache_sha256=registry['cache_sha256'],
                                  probability_transform='HGB raw logits / T; others log(clip(native p,1e-15,1)) / T; stable softmax',
                                  selection_metric='development multiclass half-Brier',
                                  sklearn_version=sklearn.__version__, official_test_accessed=False)
                    joblib.dump(bundle, model_path, compress=3)
                    record.update(model_path=str(model_path), model_sha256=digest(model_path),
                                  model_bytes=model_path.stat().st_size)
                    predictions = result_dir / f'seed{seed}__{cid}__development.npz'
                    with predictions.open('xb') as stream:
                        np.savez_compressed(stream, original_train_rows=dev, subject_ids=subjects[dev],
                                            labels=y[dev], raw_probability=probability_raw,
                                            selected_probability=pchosen, raw_half_brier=raw_losses,
                                            selected_half_brier=losses,
                                            selected_temperature=np.asarray(chosen['temperature']))
                    record['predictions_path'] = str(predictions)
                except Exception as error:
                    record.update(status='failed', completed_utc=utc(),
                                  error_type=type(error).__name__, error_message=str(error),
                                  traceback=traceback.format_exc())
                records.append(record)
                seed_records.append(record)
                dump(result_dir / 'candidate_records.json', records)
                print(json.dumps({k: record.get(k) for k in ('seed', 'candidate', 'status', 'fit_seconds',
                                                           'chosen_temperature', 'selected_metrics', 'error_message')}), flush=True)
            successful = [r for r in seed_records if r['family'] == 'HGB' and r['status'] == 'success']
            if not successful:
                selected[str(seed)] = dict(status='no_successful_hgb')
                continue
            low = min(r['selected_metrics']['half_brier'] for r in successful)
            tied = [r for r in successful if r['selected_metrics']['half_brier'] <= low + TIE_TOL]
            chosen = min(tied, key=lambda r: (r['parameters']['max_iter'] * 6 * r['parameters']['max_leaf_nodes'],
                                               r['parameters']['max_iter'], r['parameters']['max_leaf_nodes']))
            selected[str(seed)] = {k: chosen[k] for k in ('seed', 'candidate', 'model_path', 'model_sha256',
                                                       'chosen_temperature', 'parameters', 'selected_metrics',
                                                       'raw_metrics', 'model_shape', 'fit_seconds')}
            selected[str(seed)]['status'] = 'development_selected_frozen_hgb'
            dump(result_dir / 'selected_hgb.json', selected)
    final = dict(created_utc=utc(), status='complete', planned_models=24, attempted_models=len(records),
                 successful_models=sum(r['status'] == 'success' for r in records),
                 failed_models=sum(r['status'] == 'failed' for r in records),
                 total_seconds=time.perf_counter() - started_all, selected_hgb=selected,
                 prospective_registry=str(result_dir / 'prospective_registry.json'),
                 plan_sha256_at_finish=digest(PLAN), test_data_accessed=False,
                 warning='Development data used for model/temperature selection. No independent-test result.',
                 competence={})
    if len(records) != 24:
        raise AssertionError('Exactly 24 fits must be attempted')
    for seed in SEEDS:
        r = selected[str(seed)]
        sr = [item for item in records if item['seed'] == seed and item['status'] == 'success']
        if r['status'] == 'no_successful_hgb':
            continue
        best = min(sr, key=lambda item: item['selected_metrics']['half_brier'])
        final['competence'][str(seed)] = dict(best_full_candidate=best['candidate'],
            best_full_family=best['family'], best_full_half_brier=best['selected_metrics']['half_brier'],
            selected_hgb_half_brier=r['selected_metrics']['half_brier'],
            hgb_excess_vs_best= r['selected_metrics']['half_brier']-best['selected_metrics']['half_brier'])
    dump(result_dir / 'summary.json', final)
    with (result_dir / 'candidate_table.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['seed','candidate','family','status','temperature','half_brier','accuracy','log_loss',
                  'raw_half_brier','raw_accuracy','raw_log_loss','fit_seconds','model_bytes','warning_count','error']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {k:r[k] for k in ('seed','candidate','family','status')}
            row.update(temperature=r.get('chosen_temperature'),
                       fit_seconds=r.get('fit_seconds'), model_bytes=r.get('model_bytes'),
                       warning_count=len(r['warnings']), error=r.get('error_message'))
            row.update(r.get('selected_metrics', {}))
            row.update({f'raw_{k}':v for k,v in r.get('raw_metrics',{}).items()})
            writer.writerow(row)
    write_report(result_dir, final, records)
    print(json.dumps(dict(status='complete', summary=str(result_dir/'summary.json'),
                          selected_hgb=str(result_dir/'selected_hgb.json'),
                          successful_models=final['successful_models'], failed_models=final['failed_models'])), flush=True)


def write_report(result_dir, final, records):
    lines = ['# HAR development full model selection', '',
             f"Completed {final['successful_models']}/24 fixed models; {final['failed_models']} failures. No official test file or source ZIP was opened.", '',
             'Development metrics below include model/temperature selection on held-out subjects. They do not estimate independent official-test performance.', '',
             '| Seed | Model | T | Half-Brier | Accuracy | Log loss | Fit s |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for r in records:
        if r['status'] == 'success':
            s = r['selected_metrics']
            lines.append(f"| {r['seed']} | {r['candidate']} | {r['chosen_temperature']:g} | {s['half_brier']:.6f} | {s['accuracy']:.6f} | {s['log_loss']:.6f} | {r['fit_seconds']:.3f} |")
        else:
            lines.append(f"| {r['seed']} | {r['candidate']} FAILED | — | — | — | — | — |")
    lines += ['', '## Selected complete HGB references', '']
    for seed, r in final['selected_hgb'].items():
        if r['status'] != 'no_successful_hgb':
            lines.append(f"- Seed {seed}: {r['candidate']}, T={r['chosen_temperature']:g}, half-Brier={r['selected_metrics']['half_brier']:.6f}; checkpoint `{r['model_path']}`.")
    lines += ['', '## Full predictor competence and boundaries', '']
    for seed, c in final['competence'].items():
        lines.append(f"- Seed {seed}: strongest full model on the selected development score is {c['best_full_candidate']} ({c['best_full_half_brier']:.6f}); chosen HGB risk excess is {c['hgb_excess_vs_best']:.6f}.")
    lines += ['', 'All full models use the same recomputed 153 features and same saved subject-separated fitting/development rows. Every supplied scaler was verified exactly against fitting rows; it was not refit. HGB uses six-class additive logits divided by selected T. RF/ET/LR/SVC use log native probabilities clipped at 1e-15 divided by T. Positive temperature preserves each model probability argmax.', '',
              'SVC probability=True performs LIBSVM internal fitting-data-only calibration folds. Those folds are not subject-grouped and may contain overlapping windows from the same fitting subject. The four development subjects remain untouched by predictor fitting; all reported temperature choices nevertheless reuse them.', '',
              'The complete baseline comparison is limited to this fixed registry; it is not an exhaustive HAR benchmark or deep model comparison. HGB selected from four configurations is a reference predictor for certification, not a guarantee of best possible activity recognition. The method must be evaluated against frozen full references and cannot claim that its risk certificate makes an inferior reference optimal.', '',
              'The three development splits reuse official training subjects and may overlap across seeds. Treat them as sensitivity replications, not three independent test cohorts. No significance, subject-population generalization, or independent-test claim is made here.', '',
              'Warnings and failures are retained in candidate_records.json; all calibration trials, raw T=1 metrics, class counts, shape, package/thread records, and hashes are included there or in prospective_registry.json. Development per-window probabilities/losses are preserved for later paired analysis. Training times describe these fits and are not the selective processing timing experiment.']
    (result_dir/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Fit fixed registry after prospective records are saved')
    args = parser.parse_args()
    prepared = prospective()
    if args.execute:
        execute(prepared)


if __name__ == '__main__':
    main()
