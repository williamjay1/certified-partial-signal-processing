"""Evaluate 15 previously frozen complete HAR predictors after root test lock.

No fit, scaler estimation, tuning, model selection, bootstrap, ZIP, or download
operation is implemented. This module must not be executed before the root lock.

Post-lock technical recovery: only the secondary log-loss metric converts and
normalizes a separate probability copy to float64. Predictor probabilities,
half-Brier, accuracy, model, temperature, scaler, and softmax remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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
RUN = ROOT / 'results' / 'har_model_selection' / '20261002T052208Z_75415f24'
SELECTION = RUN / 'selected_hgb.json'
RECORDS = RUN / 'candidate_records.json'
SEEDS = (17, 29, 43)
FAMILIES = ('HGB', 'RF', 'ET', 'LR', 'SVC')
STAGE = 'FROZEN_BEFORE_TEST_ACCESS'
ORIGINAL_REFERENCE = ROOT / 'scripts' / 'har_reference_test.py'
FAILED_REFERENCE_RUN = ROOT / 'results' / 'har_reference_test' / 'locked_execution_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding='utf-8')


def require_d_path(path, *, existing=False):
    path = Path(path).resolve(strict=existing)
    if path.drive.upper() != 'D:':
        raise ValueError(f'Requires absolute D-drive path: {path}')
    return path


def validate_root_lock(path):
    lock = read_json(path)
    if lock.get('stage') != STAGE:
        raise PermissionError('Root test lock has not frozen the method before test access')
    if lock.get('har_method_frozen') is not True or lock.get('official_test_access_authorized') is not True:
        raise PermissionError('Official test access is not authorized by the root lock')
    if str(lock.get('selected_hgb_sha256','')).lower() != digest(SELECTION):
        raise PermissionError('Frozen selected HGB manifest hash does not match')
    if 'model_selection_records_sha256' in lock and str(lock['model_selection_records_sha256']).lower() != digest(RECORDS):
        raise PermissionError('Frozen baseline model selection records changed')
    if 'reference_script_sha256' in lock and str(lock['reference_script_sha256']).lower() != digest(ORIGINAL_REFERENCE):
        raise PermissionError('Original reference evaluation script changed since root freeze')
    return lock


def frozen_model_records():
    selected = read_json(SELECTION)
    records = read_json(RECORDS)
    frozen = []
    for seed in SEEDS:
        if selected[str(seed)].get('status') != 'development_selected_frozen_hgb':
            raise ValueError('Each seed must have one successful development-selected HGB')
        for family in FAMILIES:
            if family == 'HGB':
                matched = [r for r in records if r['seed'] == seed and r['family'] == family
                           and r['candidate'] == selected[str(seed)]['candidate']]
            else:
                matched = [r for r in records if r['seed'] == seed and r['family'] == family]
            if len(matched) != 1 or matched[0].get('status') != 'success':
                raise ValueError(f'Expected exactly one frozen successful {family} for seed {seed}')
            r = matched[0]
            if family == 'HGB':
                for key in ('model_path', 'model_sha256', 'chosen_temperature'):
                    if r[key] != selected[str(seed)][key]:
                        raise ValueError('Selected HGB entry differs from its candidate record')
            path = require_d_path(r['model_path'], existing=True)
            if digest(path) != r['model_sha256']:
                raise ValueError(f'Frozen model artifact changed: {path.name}')
            frozen.append({key:r[key] for key in ('seed','candidate','family','model_path',
                            'model_sha256','chosen_temperature','parameters','model_shape')})
    if len(frozen) != 15:
        raise AssertionError('There must be exactly 15 frozen complete models')
    return frozen


def initialize_runtime():
    temp = ROOT / 'temp' / 'har_reference_test'
    temp.mkdir(parents=True, exist_ok=True)
    for name in ('TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER'):
        os.environ[name] = str(temp)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','ASC_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONPYCACHEPREFIX'] = str(ROOT / 'temp' / 'pycache_har_reference_test')
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    sys.path.insert(0, str(OLD/'env_packages'))
    sys.path.insert(0, str(OLD/'env_core'))
    sys.path.append(r'D:\ptenv\Lib\site-packages')
    import joblib
    import numpy as np
    import scipy
    from scipy.special import softmax
    import sklearn
    from sklearn.metrics import log_loss
    from threadpoolctl import threadpool_info, threadpool_limits
    return joblib, np, scipy, softmax, sklearn, log_loss, threadpool_info, threadpool_limits


def evaluate(args):
    # Validate the immutable root lock and all model choices before touching test
    # arrays. No test signal array or archive path is ever requested by this file.
    lock_path = require_d_path(args.test_lock, existing=True)
    lock = validate_root_lock(lock_path)
    lock_sha = digest(lock_path)
    frozen = frozen_model_records()
    if shutil.disk_usage('D:\\').free < 2_000_000_000:
        raise RuntimeError('D-drive space is insufficient; do not relocate outputs')
    test_cache = require_d_path(args.test_cache, existing=True)
    cache_sha = digest(test_cache)
    test_manifest = None
    if args.test_manifest:
        manifest_path = require_d_path(args.test_manifest, existing=True)
        test_manifest = read_json(manifest_path)
        if str(test_manifest.get('cache_sha256','')).lower() != cache_sha:
            raise ValueError('Post-access test preparation manifest/cache SHA mismatch')
        if 'cache' in test_manifest and require_d_path(test_manifest['cache'], existing=True) != test_cache:
            raise ValueError('Post-access manifest names a different test cache')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    result_dir = require_d_path(args.output) if args.output else ROOT/'results'/'har_reference_test'/tag
    result_dir.mkdir(parents=True, exist_ok=False)
    joblib, np, scipy, softmax, sklearn, log_loss, threadpool_info, threadpool_limits = initialize_runtime()
    run = dict(started_utc=datetime.now(timezone.utc).isoformat(), stage='AUTHORIZED_TEST_EVALUATION',
               root_lock_path=str(lock_path), root_lock_sha256=lock_sha, root_lock=lock,
               selected_hgb_sha256=digest(SELECTION), model_selection_records_sha256=digest(RECORDS),
               reference_script_sha256=digest(ORIGINAL_REFERENCE), test_cache=str(test_cache), test_cache_sha256=cache_sha,
               recovery_stage='POSTLOCK_SECONDARY_METRIC_CONFORMANCE_REPAIR',
               recovery_script=str(Path(__file__)),recovery_script_sha256=digest(__file__),
               original_reference_script=str(ORIGINAL_REFERENCE),failed_reference_run=str(FAILED_REFERENCE_RUN),
               original_failure_manifest_sha256=digest(FAILED_REFERENCE_RUN/'evaluation_manifest.json'),
               metric_repair='Separate float64 row-normalized copy for secondary log-loss only; ORIGINAL p retained for probabilities, half-Brier and accuracy',
               conformance_tolerance=1e-12,
               post_access_manifest_verified=test_manifest is not None,
               seeds=list(SEEDS), families=list(FAMILIES), planned_models=15,
               frozen_models=frozen, fitting_operations=0, test_model_selection=False,
               test_temperature_selection=False, raw_test_array_loaded=False,
               bootstrap_executed=False, planned_shared_subject_bootstrap_repetitions=10000,
               planned_shared_subject_bootstrap_seed=7302026,
               bootstrap_main_estimand='uniform shared subject resampling; sum loss / sum cases within each seed, then mean across seeds',
               equal_subject_estimand='separate sensitivity: mean subject means, then mean across seeds',
               versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                             sklearn=sklearn.__version__,joblib=joblib.__version__),threads=threadpool_info())
    dump(result_dir/'evaluation_manifest.json',run)
    with np.load(test_cache,allow_pickle=False) as data:
        # Do not request data['raw_test'].
        x=data['features_test'].copy(); y=data['labels_test'].copy(); subjects=data['subjects'].copy()
        cache_keys=data.files
    if x.shape!=(2947,153) or y.shape!=(2947,) or subjects.shape!=(2947,):
        raise ValueError('Expected official HAR held-out cache shapes (2947,153),(2947,),(2947,)')
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Require finite float32 recomputed test features')
    if not np.issubdtype(y.dtype,np.integer) or not np.array_equal(np.unique(y),np.arange(6)):
        raise ValueError('Require encoded integer labels 0..5')
    ids=np.unique(subjects)
    if len(ids)!=9 or not np.issubdtype(subjects.dtype,np.integer) or not np.isin(ids,np.arange(1,31)).all():
        raise ValueError('Expected exactly nine held-out integer subject IDs')
    for subject in ids:
        if not np.array_equal(np.unique(y[subjects==subject]),np.arange(6)):
            raise ValueError('Every official held-out subject must contain all six activity classes')
    cases=np.arange(len(y),dtype=np.int64)
    probabilities=np.empty((3,5,2947,6),dtype=np.float64)
    risk=np.empty((3,5,2947),dtype=np.float64)
    negative_logp=np.empty_like(risk)
    correct=np.empty_like(risk,dtype=np.bool_)
    temperatures=np.empty((3,5),dtype=np.float64)
    target=np.eye(6)[y]
    per_model=[]; per_subject=[]; failures=[]
    started=time.perf_counter()
    with threadpool_limits(limits=1):
        for r in frozen:
            si=SEEDS.index(r['seed']); fi=FAMILIES.index(r['family'])
            try:
                bundle=joblib.load(r['model_path']); model=bundle['model']
                if bundle['family']!=r['family'] or bundle['seed']!=r['seed'] or bundle['candidate']!=r['candidate']:
                    raise ValueError('Frozen bundle identity mismatch')
                if float(bundle['temperature']) != r['chosen_temperature']:
                    raise ValueError('Frozen bundle temperature mismatch')
                if bundle['feature_dimension']!=153 or not np.array_equal(model.classes_,np.arange(6)):
                    raise ValueError('Frozen bundle feature/class encoding mismatch')
                if not np.array_equal(bundle['class_order'],np.arange(6)):
                    raise ValueError('Frozen probability column order mismatch')
                train_subjects=np.r_[bundle['fit_subjects'],bundle['development_subjects']]
                if np.intersect1d(ids,train_subjects).size:
                    raise ValueError('Official test subjects cross frozen fitting/development subjects')
                center=np.asarray(bundle['center']); scale=np.asarray(bundle['scale'])
                if center.shape!=(153,) or scale.shape!=(153,) or center.dtype!=np.float32 or scale.dtype!=np.float32:
                    raise ValueError('Require saved fitting-only float32 scaler')
                if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale<=0):
                    raise ValueError('Invalid frozen scaler')
                xs=(x-center)/scale; temperature=float(bundle['temperature'])
                temperatures[si,fi]=temperature
                prediction_started=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    logits=model._raw_predict(xs) if r['family']=='HGB' else np.log(np.clip(model.predict_proba(xs),1e-15,1.))
                    p=softmax(logits/temperature,axis=1)
                prediction_seconds=time.perf_counter()-prediction_started
                if p.shape!=(2947,6) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1.,atol=1e-12):
                    raise ValueError('Predictions are not finite six-class simplex points')
                probabilities[si,fi]=p
                risk[si,fi]=.5*np.sum((p-target)**2,axis=1)
                correct[si,fi]=p.argmax(1)==y
                # Metric-only normalization fixes float32/log-loss clipping and
                # accumulation consistency without changing original predictions.
                metric_p64=np.asarray(p,dtype=np.float64)
                metric_p64=metric_p64/metric_p64.sum(axis=1,keepdims=True)
                negative_logp[si,fi]=-np.log(np.clip(metric_p64[np.arange(2947),y],np.finfo(np.float64).eps,1.))
                score=dict(seed=r['seed'],family=r['family'],candidate=r['candidate'],temperature=temperature,
                           windows=2947,subjects=9,half_brier=float(risk[si,fi].mean()),
                           accuracy=float(correct[si,fi].mean()),log_loss=float(log_loss(y,metric_p64,labels=np.arange(6))),
                           original_probability_dtype=str(p.dtype),log_loss_metric_dtype='float64',
                           maximum_metric_only_normalization_difference=float(np.max(np.abs(metric_p64-p))),
                           prediction_seconds=prediction_seconds,warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
                if abs(score['log_loss']-float(negative_logp[si,fi].mean()))>1e-12:
                    raise ValueError('Metric-only float64 case log-loss array does not reproduce sklearn log_loss')
                per_model.append(score)
                for subject in ids:
                    rows=subjects==subject; count=int(rows.sum())
                    per_subject.append(dict(seed=r['seed'],family=r['family'],candidate=r['candidate'],subject=int(subject),
                        windows=count,temperature=temperature,class_counts=np.bincount(y[rows],minlength=6).tolist(),
                        half_brier=float(risk[si,fi,rows].mean()),half_brier_sum=float(risk[si,fi,rows].sum()),
                        accuracy=float(correct[si,fi,rows].mean()),correct_count=int(correct[si,fi,rows].sum()),
                        log_loss=float(negative_logp[si,fi,rows].mean()),log_loss_sum=float(negative_logp[si,fi,rows].sum())))
                print(json.dumps({k:score[k] for k in ('seed','family','temperature','half_brier','accuracy','log_loss')}),flush=True)
            except Exception as error:
                failure=dict(seed=r['seed'],family=r['family'],candidate=r['candidate'],status='failed',
                             error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
                failures.append(failure)
                print(json.dumps(failure),flush=True)
            dump(result_dir/'model_records_partial.json',dict(successes=per_model,failures=failures))
    run.update(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
               cache_keys=cache_keys,held_out_subject_ids=ids.tolist(),subject_case_counts={str(s):int((subjects==s).sum()) for s in ids},
               successful_models=len(per_model),failed_models=len(failures),status='failed' if failures else 'complete')
    if digest(lock_path)!=lock_sha:
        raise PermissionError('Root test lock changed during evaluation')
    dump(result_dir/'evaluation_manifest.json',run)
    dump(result_dir/'full_model_metrics.json',dict(models=per_model,failures=failures))
    dump(result_dir/'subject_model_metrics.json',per_subject)
    if failures:
        raise RuntimeError('At least one frozen predictor failed; all failures recorded, no partial final tensor emitted')
    paired=risk[:,0,None,:]-risk[:,1:,:]
    with (result_dir/'reference_outcomes.npz').open('xb') as stream:
        np.savez_compressed(stream,case_id=cases,subjects=subjects,labels=y,seeds=np.asarray(SEEDS),
                            families=np.asarray(FAMILIES),temperatures=temperatures,
                            probabilities=probabilities,half_brier=risk,correct=correct,
                            negative_log_probability=negative_logp,hgb_minus_baseline_half_brier=paired,
                            baseline_families=np.asarray(FAMILIES[1:]))
    subject_paired=[]
    subject_counts=np.asarray([(subjects==s).sum() for s in ids],dtype=np.int64)
    subject_risk_sums=np.empty((3,5,9)); subject_correct_counts=np.empty((3,5,9),dtype=np.int64)
    subject_logloss_sums=np.empty((3,5,9)); subject_difference_sums=np.empty((3,4,9))
    for sj,subject in enumerate(ids):
        rows=subjects==subject
        subject_risk_sums[:,:,sj]=risk[:,:,rows].sum(2)
        subject_correct_counts[:,:,sj]=correct[:,:,rows].sum(2)
        subject_logloss_sums[:,:,sj]=negative_logp[:,:,rows].sum(2)
        subject_difference_sums[:,:,sj]=paired[:,:,rows].sum(2)
        for si,seed in enumerate(SEEDS):
            for bi,family in enumerate(FAMILIES[1:]):
                subject_paired.append(dict(seed=seed,baseline_family=family,subject=int(subject),
                    windows=int(rows.sum()),hgb_minus_baseline_half_brier_sum=float(paired[si,bi,rows].sum()),
                    hgb_minus_baseline_half_brier_mean=float(paired[si,bi,rows].mean())))
    with (result_dir/'shared_subject_bootstrap_inputs.npz').open('xb') as stream:
        np.savez_compressed(stream,subject_ids=ids,subject_case_counts=subject_counts,
                            seeds=np.asarray(SEEDS),families=np.asarray(FAMILIES),
                            subject_half_brier_sums=subject_risk_sums,
                            subject_correct_counts=subject_correct_counts,
                            subject_log_loss_sums=subject_logloss_sums,
                            baseline_families=np.asarray(FAMILIES[1:]),
                            subject_hgb_minus_baseline_half_brier_sums=subject_difference_sums)
    dump(result_dir/'paired_subject_risk_differences.json',subject_paired)
    with (result_dir/'case_outcomes.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['seed','family','case_id','subject','label','temperature','half_brier','correct','negative_log_probability',
                'p0','p1','p2','p3','p4','p5']
        writer=csv.writer(stream); writer.writerow(fields)
        for si,seed in enumerate(SEEDS):
            for fi,family in enumerate(FAMILIES):
                for case in cases:
                    writer.writerow([seed,family,int(case),int(subjects[case]),int(y[case]),temperatures[si,fi],
                                     risk[si,fi,case],int(correct[si,fi,case]),negative_logp[si,fi,case],
                                     *probabilities[si,fi,case].tolist()])
    competence=[]
    for si,seed in enumerate(SEEDS):
        scores=[s for s in per_model if s['seed']==seed]
        descriptive_best=min(scores,key=lambda s:s['half_brier'])
        hgb=[s for s in scores if s['family']=='HGB'][0]
        competence.append(dict(seed=seed,frozen_hgb_half_brier=hgb['half_brier'],
            descriptive_lowest_test_risk_family=descriptive_best['family'],
            descriptive_lowest_test_risk=descriptive_best['half_brier'],
            hgb_excess_vs_descriptive_lowest=hgb['half_brier']-descriptive_best['half_brier'],
            reference_replaced=False,
            paired_risk_differences={family:float(paired[si,bi].mean()) for bi,family in enumerate(FAMILIES[1:])}))
    dump(result_dir/'reference_competence.json',dict(competence=competence,
        statement='Complete predictor comparison only; none of these test outcomes changes the frozen reference.',
        seed_averaged_window_weighted_metrics={family:{metric:float(np.mean([s[metric] for s in per_model if s['family']==family]))
                                              for metric in ('half_brier','accuracy','log_loss')} for family in FAMILIES},
        independent_subject_clusters=9,seed_subject_pairs_are_not_independent=27,
        inference='Root will use common nine-subject weighted bootstrap; no bootstrap was run here.'))
    run['outputs']={name:dict(path=str(result_dir/name),sha256=digest(result_dir/name)) for name in (
        'reference_outcomes.npz','shared_subject_bootstrap_inputs.npz','full_model_metrics.json',
        'subject_model_metrics.json','paired_subject_risk_differences.json','case_outcomes.csv','reference_competence.json')}
    dump(result_dir/'evaluation_manifest.json',run)
    print(json.dumps(dict(status='complete',result_dir=str(result_dir),models=15,subjects=9,
                          bootstrap_executed=False,root_lock_unchanged=True)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-cache',type=Path,required=True)
    parser.add_argument('--test-lock',type=Path,required=True)
    parser.add_argument('--test-manifest',type=Path,default=None,
                        help='Optional post-access loader manifest with cache_sha256')
    parser.add_argument('--output',type=Path,default=None,
                        help='Fresh D-drive output directory; never overwrite an existing directory')
    evaluate(parser.parse_args())


if __name__=='__main__':
    main()
