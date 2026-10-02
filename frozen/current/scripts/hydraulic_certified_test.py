"""Lock-guarded external confirmation of the frozen Hydraulic v4 pipeline.

Preparation only until root explicitly authorizes execution. Never import this
driver to inspect test data: main first validates a frozen lock and every input
hash, then opens the original prepared split and the immutable signal archive.
The original project's prepared test features were evaluated previously; this
is external confirmation of the new raw-processing method, not a pristine
overall-research test. No fitting, feature order or tolerance selection occurs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import uuid
import zipfile

ROOT = Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
SOURCE = Path(r'D:\MLWork\ASC_ComputeValue_20261002')
OLD = Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
KIND = 'HYDRAULIC_EXTERNAL_CONFIRMATION'
STAGE = 'FROZEN_BEFORE_TEST_ACCESS'
DELTAS = [0.0, .01, .025, .05, .1]
VARIANTS = {'tight_minimax': 0, 'coordinate_minimax': 1, 'tight_midpoint': 2}
SENSORS = ['PS1','PS2','PS3','PS4','PS5','PS6','EPS1','FS1','FS2','TS1','TS2','TS3','TS4','VS1']
SIGNAL_SAMPLES = [6000]*7 + [600]*2 + [60]*5
REQUIRED_CODE = [
    Path(__file__), ROOT/'scripts'/'run_leaf_mask_pilot.py',
    ROOT/'scripts'/'run_hydraulic_prototype.py', ROOT/'scripts'/'certificate_kernel.py',
    ROOT/'research'/'leaf_mask_kernel.py', SOURCE/'scripts'/'raw_compute.py',
    OLD/'scripts'/'methods.py',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def dump_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def d_path(path):
    result = Path(path).resolve()
    if result.drive.upper() != 'D:':
        raise ValueError('All work inputs/outputs must remain on D: ' + str(result))
    return result


def locked_ref(section, key, raw=False):
    ref = section.get(key)
    if not isinstance(ref, dict) or set(('path','sha256')) - set(ref):
        raise ValueError('Lock requires ' + key + ' {path,sha256}')
    path = Path(ref['path']).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if raw:
        if path.drive.upper() not in ('F:','E:'):
            raise ValueError('Raw input must be immutable F/E source')
    else:
        d_path(path)
    expected = str(ref['sha256']).lower()
    if len(expected) != 64 or digest(path) != expected:
        raise RuntimeError('Frozen hash mismatch: ' + key)
    return path


def read_lock(path):
    path = d_path(path)
    if not path.is_file():
        raise FileNotFoundError('No frozen test-access lock: ' + str(path))
    lock_sha = digest(path)
    lock = json.loads(path.read_text(encoding='utf-8'))
    if lock.get('stage') != STAGE:
        raise RuntimeError('Refusing test access: lock stage must equal ' + STAGE)
    section = lock.get('hydraulic', lock.get('hydraulic_external_confirmation', lock))
    if section.get('kind') != KIND:
        raise RuntimeError('Refusing test access: wrong Hydraulic lock kind')
    if section.get('seed') != 17 or section.get('batch_size') != 4:
        raise ValueError('Frozen v4 requires seed17 and batch_size4')
    if section.get('deltas') != DELTAS or section.get('variants') != VARIANTS:
        raise ValueError('Frozen five tolerances/three variants differ')
    if section.get('primary_delta') != .025:
        raise ValueError('Lock must retain the development-preselected primary delta .025')
    if section.get('repetitions') != 3 or not isinstance(section.get('timing_seed'), int):
        raise ValueError('Lock must specify three repetitions and an integer timing_seed')
    if section.get('controls', []):
        raise ValueError('This v1 external driver freezes main variants/full controls only; extra controls require a reviewed driver and new preaccess lock')
    order = section.get('order')
    if not isinstance(order, list) or len(order) != 28 or sorted(order) != list(range(28)):
        raise ValueError('Frozen order must cover exactly the 28 real blocks')
    scripts = section.get('scripts')
    if not isinstance(scripts, dict):
        raise ValueError('Lock requires scripts {absolute_path:sha256}')
    normalized = {str(Path(p).resolve()).casefold(): str(h).lower() for p,h in scripts.items()}
    for script in REQUIRED_CODE:
        resolved = script.resolve()
        expected = normalized.get(str(resolved).casefold())
        if expected is None or digest(resolved) != expected:
            raise RuntimeError('Missing/mismatched frozen script: ' + str(resolved))
    paths = {key: locked_ref(section,key,key=='raw_source') for key in
             ('model','development_cache','prepared_split','raw_source','v4_configuration','development_manifest')}
    if paths['prepared_split'] != (OLD/'datasets'/'hydraulic_valve.npz').resolve():
        raise ValueError('Use the original frozen hydraulic_valve.npz split')
    if paths['development_cache'] != (SOURCE/'cache'/'hydraulic_development_raw_v1.npz').resolve():
        raise ValueError('Use the original907-fit development cache')
    v4 = json.loads(paths['v4_configuration'].read_text(encoding='utf-8'))
    for key in ('order','deltas','variants','repetitions'):
        if v4.get(key) != section[key]:
            raise ValueError('Lock differs from v4 development configuration: ' + key)
    if v4.get('checkpoints') != [4,8,12,16,20,24,28]:
        raise ValueError('v4 checkpoint schedule differs')
    if v4.get('model_sha256') != section['model']['sha256']:
        raise ValueError('v4 model identity differs')
    for file_key, v4_key in [('certificate_kernel.py','kernel_sha256'),
                              ('leaf_mask_kernel.py','leaf_helper_sha256'),
                              ('run_leaf_mask_pilot.py','driver_sha256'),
                              ('run_hydraulic_prototype.py','parent_driver_sha256')]:
        matching = [p for p in REQUIRED_CODE if p.name == file_key]
        if len(matching) != 1 or digest(matching[0]) != v4.get(v4_key):
            raise RuntimeError('Current implementation differs from frozen v4: ' + file_key)
    dev = json.loads(paths['development_manifest'].read_text(encoding='utf-8'))
    if dev.get('scaler_fit') != 'predictor_fit only' or dev.get('development_rows') != 1770:
        raise ValueError('Development scaler/sample provenance invalid')
    if dev.get('cache_sha256') != section['development_cache']['sha256']:
        raise ValueError('Development manifest/cache identity differs')
    if dev.get('source_sha256') != section['raw_source']['sha256']:
        raise ValueError('Development and external raw sources differ')
    if int(dev.get('split',{}).get('predictor_fit',{}).get('n',-1)) != 907:
        raise ValueError('907-fit scaler provenance missing')
    return path, lock_sha, section, paths


def configure_runtime():
    temp = ROOT/'temp'/'hydraulic_external_confirmation'
    for path in (temp, temp/'pycache', ROOT/'cache'/'compiled', ROOT/'cache'/'torch'):
        path.mkdir(parents=True, exist_ok=True)
    for key in ('ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(TEMP=str(temp), TMP=str(temp), PYTHONPYCACHEPREFIX=str(temp/'pycache'),
                      NUMBA_CACHE_DIR=str(ROOT/'cache'/'compiled'), TORCH_HOME=str(ROOT/'cache'/'torch'))
    sys.pycache_prefix = str(temp/'pycache')
    for path in (ROOT/'scripts', ROOT/'research', OLD/'env_core', OLD/'env_packages'):
        sys.path.insert(0, str(path))


def load_split_and_scaler(paths, np):
    # NpzFile opens members lazily. Only the members below are decoded.
    with np.load(paths['development_cache'], allow_pickle=False) as z:
        development_rows = z['original_rows'].copy()
        development_units = z['units'].copy()
        fit = z['predictor_fit'].copy()
        validation = z['predictor_val'].copy()
        center, scale = z['center'].copy(), z['scale'].copy()
    if development_rows.shape != (1770,) or len(np.unique(development_rows)) != 1770:
        raise AssertionError('Development row universe changed')
    if development_units.shape != (1770,) or len(fit) != 907 or len(validation) != 223:
        raise AssertionError('907/223 fit/development split changed')
    if np.intersect1d(fit, validation).size or np.intersect1d(development_units[fit],development_units[validation]).size:
        raise AssertionError('Fit and internal-development proxy units overlap')
    if center.shape != (238,) or scale.shape != (238,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise AssertionError('Frozen scaler invalid')
    with np.load(paths['prepared_split'], allow_pickle=False) as z:
        train_rows, val_rows, test_rows = z['train_idx'].copy(), z['val_idx'].copy(), z['test_idx'].copy()
        all_units = z['split_group']
        test_units = all_units[test_rows].copy()
        if not np.array_equal(all_units[development_rows], development_units):
            raise AssertionError('Prepared and development proxy unit IDs differ')
        labels = z['y'][test_rows].copy()
    outer_development = np.sort(np.r_[train_rows,val_rows])
    if not np.array_equal(outer_development,development_rows):
        raise AssertionError('1770 development universe does not equal original train+val')
    if test_rows.shape != (435,) or len(np.unique(test_rows)) != 435 or not np.array_equal(test_rows,np.sort(test_rows)):
        raise AssertionError('Original test_idx must be435 distinct sorted rows')
    if np.any(test_rows < 0) or np.any(test_rows >= 2205):
        raise AssertionError('Test source row index out of range')
    if np.intersect1d(test_rows,development_rows).size or np.intersect1d(test_units,development_units).size:
        raise AssertionError('External test overlaps original development rows/proxy segments')
    if labels.shape != (435,) or not np.array_equal(np.unique(labels),np.arange(4)):
        raise AssertionError('Expected all four Hydraulic valve classes')
    provenance = {'external_cases':435, 'original_development_cases':1770,
                  'predictor_fit_cases':907,'internal_validation_cases':223,
                  'external_proxy_segments':int(len(np.unique(test_units))),
                  'development_proxy_segments':int(len(np.unique(development_units))),
                  'test_vs_all1770_row_overlap':0,'test_vs_all1770_proxy_overlap':0,
                  'test_vs907_fit_proxy_overlap':0,'test_vs223_internal_proxy_overlap':0,
                  'proxy_note':'Contiguous same-condition segments are proxy units, not independently certified experimental runs',
                  'scaler_source':'Frozen907-fit center/scale; no fit/re-estimation',
                  'decoded_development_cache_members':['original_rows','units','predictor_fit','predictor_val','center','scale'],
                  'decoded_prepared_split_members':['train_idx','val_idx','test_idx','split_group','y'],
                  'old_test_feature_matrix_decoded':False}
    return test_rows.astype(np.int64), test_units, labels.astype(np.int64), center, scale, provenance


def parse_selected_signals(archive, selected_rows, np):
    selected = set(map(int, selected_rows))
    raw, opened = [], []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for sensor, width in zip(SENSORS,SIGNAL_SAMPLES):
            members = [n for n in names if PurePosixPath(n).name.casefold() == (sensor+'.txt').casefold()]
            if len(members) != 1:
                raise ValueError('Expected one physical channel member: ' + sensor)
            member = members[0]
            values, indices, scanned = [], [], 0
            with z.open(member) as stream:
                for row, line in enumerate(stream):
                    scanned = row+1
                    if row in selected:
                        value = np.fromstring(line.decode('ascii'),sep=' ',dtype=np.float64)
                        if value.shape != (width,) or not np.isfinite(value).all():
                            raise ValueError('Invalid selected raw row: '+sensor+'/'+str(row))
                        values.append(value); indices.append(row)
            if scanned != 2205 or indices != selected_rows.tolist():
                raise AssertionError('Source row count/order mismatch: '+sensor)
            array = np.stack(values)
            if array.shape != (435,width):
                raise AssertionError('Unexpected selected raw shape')
            raw.append(array); opened.append(member)
            print('PARSED_LOCKED_HYDRAULIC_TEST',sensor,array.shape,flush=True)
    return raw, opened


def reference_features(raw, center, scale, rc, np):
    unscaled = np.empty((435,238),np.float32)
    operators = rc.RawBlocks(raw)
    for case in range(435):
        for block in range(28):
            unscaled[case,rc.OFFSETS[block]:rc.OFFSETS[block+1]] = operators.values(case,block)
    # Only reference/audit arrays: the online pipeline never receives these.
    full = (unscaled-center)/scale
    return unscaled, full


def correctness_checks(pipe, full, reference, native_scores, variants, np, label_losses):
    maxima = {'probability_interval_violation_signed':-1.0,
              'score_interval_violation_signed':-1.0,
              'every_label_certificate_violation_signed':-1.0}
    checked = 0
    for case in range(435):
        for variant in variants.values():
            traced = pipe.partial(case,0.0,variant,trace=True)
            for stage in traced['trace']:
                checked += 1
                q, s = reference[case],native_scores[case]
                maxima['probability_interval_violation_signed'] = max(maxima['probability_interval_violation_signed'],
                    float(np.max(stage['prob_lower']-q)),float(np.max(q-stage['prob_upper'])))
                maxima['score_interval_violation_signed'] = max(maxima['score_interval_violation_signed'],
                    float(np.max(stage['score_lower']-s)),float(np.max(s-stage['score_upper'])))
                excess = label_losses(stage['probability'])-label_losses(q)
                maxima['every_label_certificate_violation_signed'] = max(maxima['every_label_certificate_violation_signed'],
                    float(np.max(excess)-stage['certificate']))
    if maxima['probability_interval_violation_signed'] > 2e-12 or maxima['score_interval_violation_signed'] > 2e-11 or maxima['every_label_certificate_violation_signed'] > 2e-12:
        raise AssertionError('Locked external prefix correctness check failed: '+str(maxima))
    return {'status':'PASS_NUMERICAL_CHECKS_NOT_FORMAL_FLOATING_POINT_PROOF',
            'all_external_cases_checked':435,'variant_count':len(variants),
            'prefix_stage_checks':checked,**maxima}


def timed_cases(pipe, reference, labels, original_rows, units, np, base, section, out):
    specs = [(name,float(delta)) for name in VARIANTS for delta in DELTAS] + [('native_full',None),('compiled_full',None)]
    n_methods = len(specs)
    probabilities = np.empty((n_methods,435,4),np.float64)
    certificates = np.empty((n_methods,435),np.float64)
    counts = np.empty((n_methods,435),np.int16)
    fallback = np.empty((n_methods,435),np.bool_)
    collapse = np.empty((n_methods,435),np.bool_)
    elapsed = np.empty((n_methods,435,3),np.float64)
    rng = np.random.default_rng(section['timing_seed'])
    max_certificate_violation, max_tolerance_violation = -1.0, -1.0
    rows_written = 0
    with (out/'case_results.jsonl').open('x',encoding='utf-8') as stream:
        for case in range(435):
            for rep in range(3):
                for spec_id in rng.permutation(n_methods):
                    name,delta = specs[spec_id]
                    result = pipe.partial(case,delta,VARIANTS[name]) if name in VARIANTS else pipe.full(case,name=='compiled_full')
                    p = result.pop('probability')
                    implementation_variant = result.pop('method')
                    if not np.isfinite(p).all() or np.any(p < 0) or abs(float(p.sum())-1.0)>1e-10:
                        raise AssertionError('Invalid emitted probability')
                    elapsed[spec_id,case,rep] = result['complete_ms']
                    if rep == 0:
                        probabilities[spec_id,case] = p
                        certificates[spec_id,case] = result['certificate']
                        counts[spec_id,case] = result['blocks']
                        fallback[spec_id,case] = result['fallback']
                        collapse[spec_id,case] = result['collapsed']
                    elif not np.array_equal(probabilities[spec_id,case],p) or counts[spec_id,case] != result['blocks'] or certificates[spec_id,case] != result['certificate']:
                        raise AssertionError('Repeated online output/stopping decision changed')
                    loss = base.label_losses(p)
                    excess = loss-base.label_losses(reference[case])
                    certificate_violation = float(excess.max()-result['certificate'])
                    tolerance_violation = None if delta is None else float(excess.max()-delta)
                    max_certificate_violation = max(max_certificate_violation,certificate_violation)
                    if delta is not None:
                        max_tolerance_violation = max(max_tolerance_violation,tolerance_violation)
                    row = {'external_position':case,'original_row':int(original_rows[case]),
                           'proxy_segment':int(units[case]),'label':int(labels[case]),
                           'repetition':rep,'method':name,'delta':delta,'implementation_variant':implementation_variant,
                           'probability':p.tolist(),'reference_probability':reference[case].tolist(),
                           'half_brier':float(loss[labels[case]]),'accuracy':int(p.argmax()==labels[case]),
                           'reference_loss_excess':float(excess[labels[case]]),
                           'max_every_label_excess':float(excess.max()),
                           'certificate_violation_signed':certificate_violation,
                           'tolerance_violation_signed':tolerance_violation,**result}
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    rows_written += 1
                stream.flush()
            if (case+1)%25 == 0 or case == 434:
                print('TIMED_LOCKED_HYDRAULIC_EXTERNAL',case+1,'of435',flush=True)
    if rows_written != 435*3*17 or max_certificate_violation > 2e-12 or max_tolerance_violation > 2e-12:
        raise AssertionError('Incomplete run or every-label certificate/tolerance violation')
    aggregate = []
    ref_losses = np.asarray([base.label_losses(q)[y] for q,y in zip(reference,labels)])
    for spec_id,(name,delta) in enumerate(specs):
        p = probabilities[spec_id]
        loss = np.asarray([base.label_losses(a)[y] for a,y in zip(p,labels)])
        all_excess = np.stack([base.label_losses(a)-base.label_losses(q) for a,q in zip(p,reference)])
        aggregate.append({'method':name,'delta':delta,'external_cases':435,'timing_repetitions':3,
                          'complete_ms':base.summary(elapsed[spec_id].reshape(-1)),
                          'per_case_median_complete_ms':base.summary(np.median(elapsed[spec_id],axis=1)),
                          'half_brier':float(loss.mean()),'accuracy':float(np.mean(p.argmax(1)==labels)),
                          'mean_reference_loss_excess':float((loss-ref_losses).mean()),
                          'maximum_every_label_excess':float(all_excess.max()),
                          'maximum_certificate_violation_signed':float(np.max(all_excess.max(1)-certificates[spec_id])),
                          'maximum_tolerance_violation_signed':None if delta is None else float(all_excess.max()-delta),
                          'mean_block_count':float(counts[spec_id].mean()),
                          'block_count_distribution':{str(k):int(np.sum(counts[spec_id]==k)) for k in np.unique(counts[spec_id])},
                          'full_fallback_rate':float(fallback[spec_id].mean()),
                          'exact_collapse_rate':float(collapse[spec_id].mean()),
                          'reference_class_agreement':float(np.mean(p.argmax(1)==reference.argmax(1)))})
    faster = min((r for r in aggregate if r['method'] in ('native_full','compiled_full')),key=lambda r:r['complete_ms']['mean'])
    for row in aggregate:
        row['faster_full_comparator'] = faster['method']
        row['complete_mean_time_reduction_vs_faster_full'] = 1.0-row['complete_ms']['mean']/faster['complete_ms']['mean']
    outputs = {'probabilities':probabilities,'certificates':certificates,'block_counts':counts,
               'full_fallback':fallback,'exact_collapse':collapse,'complete_ms':elapsed,
               'reference_probability':reference,'labels':labels,'original_rows':original_rows,'proxy_segments':units,
               'method_names':np.asarray([x[0] for x in specs]),
               'deltas':np.asarray([np.nan if x[1] is None else x[1] for x in specs])}
    with (out/'outcomes.npz').open('xb') as stream:
        np.savez_compressed(stream,**outputs)
    return aggregate, rows_written, max_certificate_violation, max_tolerance_violation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-lock', required=True)
    parser.add_argument('--output-root', default=str(ROOT/'results'/'hydraulic_external_confirmation'))
    args = parser.parse_args()
    lock_path, lock_sha, section, paths = read_lock(args.test_lock)
    free = shutil.disk_usage('D:\\').free
    if free < 2*1024**3:
        raise RuntimeError('Insufficient D space; no storage substitution')
    configure_runtime()
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits, threadpool_info
    import run_leaf_mask_pilot as leaf
    import certificate_kernel as ck
    base,rc = leaf.base,leaf.base.rc
    if rc.SENSORS != SENSORS or len(rc.WIDTHS) != 28 or int(rc.OFFSETS[-1]) != 238:
        raise AssertionError('Raw feature schema changed')
    # Recheck before the first decode of any reserved array/member.
    if digest(lock_path) != lock_sha:
        raise RuntimeError('Lock changed before external data access')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out = d_path(args.output_root)/stamp
    out.mkdir(parents=True,exist_ok=False)
    cache = ROOT/'cache'/('hydraulic_external_raw__'+stamp+'.npz')
    if cache.exists():
        raise FileExistsError(cache)
    started = time.perf_counter()
    config = {'phase':KIND,'test_lock_path':str(lock_path),'test_lock_sha256':lock_sha,
              'frozen':section,'D_free_bytes_before_compute':free,
              'raw_window_access_note':'Only435 selected rows parsed; sequential ZIP line scan skips all other rows without numeric decoding',
              'timing_inclusions':'Resident raw input to probability: feature operations, scaling, allocations, leaf masks, intervals, certificate, routing and exact fallback',
              'timing_exclusions':['raw disk read/decompression','offline reference construction','training','JIT warmup','outside-timer label/reference checks'],
              'prior_test_exposure_note':'Old prepared-feature test outcomes were evaluated in the earlier project; these raw windows were not used to tune the certified processing method',
              'new_model_fits':0,'new_scaler_fits':0,'new_order_selection':0,'new_tolerance_selection':0}
    dump_new(out/'frozen_configuration_before_external_access.json',config)
    try:
        with threadpool_limits(limits=1):
            rows,units,labels,center,scale,split_record = load_split_and_scaler(paths,np)
            if digest(lock_path) != lock_sha:
                raise RuntimeError('Lock changed before raw ZIP access')
            raw,opened = parse_selected_signals(paths['raw_source'],rows,np)
            unscaled,full = reference_features(raw,center,scale,rc,np)
            with cache.open('xb') as stream:
                np.savez_compressed(stream,original_rows=rows,proxy_segments=units,labels=labels,
                                    features_unscaled=unscaled,reference_X=full,center=center,scale=scale,
                                    group_index=rc.GROUP_INDEX,block_widths=rc.WIDTHS,
                                    **{'raw_'+s:a for s,a in zip(SENSORS,raw)})
            cache_record = {'phase':KIND,'source':section['raw_source'],
                            'test_lock_sha256':lock_sha,'cache_path':str(cache),'cache_sha256':digest(cache),
                            'raw_shapes':{s:list(a.shape) for s,a in zip(SENSORS,raw)},
                            'features_shape':list(full.shape),'opened_physical_channel_members':opened,
                            'profile_or_virtual_channel_opened':False,'selected_numeric_rows_per_channel':435,
                            'unselected_rows_numeric_decoded':0,'source_unchanged':digest(paths['raw_source'])==section['raw_source']['sha256'],
                            **split_record}
            if not cache_record['source_unchanged']:
                raise RuntimeError('Immutable source changed during parsing')
            dump_new(out/'raw_cache_manifest.json',cache_record)
            model = joblib.load(paths['model'])
            if not np.array_equal(model.classes_,np.arange(4)) or model.n_features_in_ != 238:
                raise AssertionError('Frozen classifier schema differs')
            forest = ck.FrozenForest(model,rc.GROUP_INDEX)
            blocks = rc.RawBlocks(raw,center,scale)
            pipe = leaf.LeafPipeline(model,forest,blocks,section['order'])
            if forest.batch_ends.tolist() != [4,8,12,16,20,24,28]:
                raise AssertionError('Online checkpoints differ')
            reference = np.stack([forest.exact(np.asarray(x,np.float64)) for x in full])
            native = model.predict_proba(full)
            native_scores = model._raw_predict(full)
            mismatch = float(np.max(np.abs(reference-native)))
            if mismatch > 2e-12:
                raise AssertionError('Native/compiled reference mismatch on external cases')
            conformance = {'all435_native_compiled_max_probability_difference':mismatch,
                           'same_frozen_predictor':True,'reference_is_never_online_policy_input':True}
            dump_new(out/'native_reference_conformance.json',conformance)
            # Fixed warmup on the first source row, no outcome-dependent choice.
            for variant in VARIANTS.values():
                pipe.partial(0,.025,variant)
            pipe.full(0,True);pipe.full(0,False)
            correctness = correctness_checks(pipe,full,reference,native_scores,VARIANTS,np,base.label_losses)
            dump_new(out/'all435_prefix_correctness.json',correctness)
            aggregate,n_written,cert_violation,tol_violation = timed_cases(pipe,reference,labels,rows,units,np,base,section,out)
            runtime = threadpool_info()
        result = {'phase':KIND,'status':'COMPLETE','configuration':config,
                  'split_provenance':split_record,'reference_conformance':conformance,
                  'correctness':correctness,'all_methods':aggregate,
                  'external_cases':435,'repetitions':3,'jsonl_rows':n_written,
                  'maximum_emitted_certificate_violation_signed':cert_violation,
                  'maximum_emitted_tolerance_violation_signed':tol_violation,
                  'cache':cache_record,'single_thread_runtime':runtime,
                  'execution_seconds':time.perf_counter()-started,
                  'outcomes_sha256':digest(out/'outcomes.npz'),
                  'case_results_sha256':digest(out/'case_results.jsonl'),
                  'numerical_limit':'Exact-arithmetic theorem plus engineering guards and empirical checks; no formal all-input machine-arithmetic proof',
                  'test_status_limit':'External confirmation after frozen certified-method development; not a pristine test of the entire earlier research programme'}
        dump_new(out/'executed_results.json',result)
        print('HYDRAULIC_EXTERNAL_COMPLETE',str(out),flush=True)
    except BaseException as exc:
        dump_new(out/'failure.json',{'phase':KIND,'status':'FAILED_RETAINED_NO_METHOD_RETUNING',
                                   'error_type':type(exc).__name__,'message':str(exc),
                                   'test_lock_sha256':lock_sha,'elapsed_seconds':time.perf_counter()-started})
        raise


if __name__ == '__main__':
    main()
