"""Post-test descriptive complete HAR timing of fifteen frozen model bundles.

Each call recomputes all eighteen genuine raw feature blocks, applies the saved
fit-only scaler, and returns the frozen native probability/temperature output.
Three calibrated HGB compiled implementations are additionally measured. No fit,
selection, raw-source access, preprocessing estimation or policy is implemented.
Only execute after the coordinated component-profile CPU release.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, platform, shutil, sys, time, traceback, uuid
from pathlib import Path

ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
OLD=Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
SEEDS=(17,29,43)
FAMILIES=('HGB','RF','ET','LR','SVC','HGB_compiled')
COHORT_SEED=7302026
TIMING_SEED=7342026
REPETITIONS=3

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def on_d(path,existing=False):
    p=Path(path).resolve(strict=existing)
    if p.drive.upper()!='D:':raise ValueError('Work input/output must stay on D')
    return p

def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def runtime():
    temp=ROOT/'temp'/'review2_full_model_timing';temp.mkdir(parents=True,exist_ok=True)
    for k in ('TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER'):os.environ[k]=str(temp)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','ASC_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
    sys.pycache_prefix=str(temp/'pycache');os.environ['PYTHONPYCACHEPREFIX']=sys.pycache_prefix
    os.environ['NUMBA_CACHE_DIR']=str(ROOT/'cache'/'compiled')
    for p in (ROOT/'research',ROOT/'scripts',OLD/'env_packages',OLD/'env_core'):sys.path.insert(0,str(p))
    sys.path.append(r'D:\ptenv\Lib\site-packages')

def quantiles(np,values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
                p95=float(np.quantile(values,.95)),p99=float(np.quantile(values,.99)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--test-lock',required=True,type=Path)
    ap.add_argument('--test-cache',required=True,type=Path)
    ap.add_argument('--formal-run',required=True,type=Path)
    ap.add_argument('--reference-run',required=True,type=Path)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    lock_path=on_d(args.test_lock,True);lock=read(lock_path);lock_sha=sha(lock_path)
    if lock.get('stage')!='FROZEN_BEFORE_TEST_ACCESS' or not lock.get('official_test_access_authorized') or not lock.get('har_method_frozen'):
        raise PermissionError('Requires the root frozen authorized test lock')
    if shutil.disk_usage('D:\\').free<2*1024**3:raise RuntimeError('Insufficient D space')
    source_scripts={ROOT/'research'/'prepare_har_training.py':'har_operator_sha256',
        ROOT/'scripts'/'har_development_benchmark.py':'har_compute_script_sha256',
        ROOT/'scripts'/'certificate_kernel.py':'certificate_kernel_sha256',
        ROOT/'research'/'leaf_mask_kernel.py':'leaf_helper_sha256'}
    for path,key in source_scripts.items():
        if sha(path)!=lock[key]:raise PermissionError('Frozen shared implementation hash changed: '+str(path))
    cache=on_d(args.test_cache,True);formal=on_d(args.formal_run,True);prior=on_d(args.reference_run,True)
    prior_manifest=read(prior/'evaluation_manifest.json')
    if prior_manifest.get('status')!='complete' or prior_manifest.get('successful_models')!=15:
        raise ValueError('Prior fifteen-model reference evaluation is incomplete')
    if sha(cache)!=prior_manifest['test_cache_sha256'] or prior_manifest['root_lock_sha256']!=lock_sha:
        raise PermissionError('Different frozen raw test cache/lock')
    prior_outcome=prior/'reference_outcomes.npz'
    if sha(prior_outcome)!=prior_manifest['outputs']['reference_outcomes.npz']['sha256']:
        raise PermissionError('Prior reference outcome artifact hash changed')
    cohort_record=read(formal/'frozen_shared_timing_cohort_before_predictions.json')
    cases_input=cohort_record['test_indices']
    if len(cases_input)!=900 or cohort_record['seed']!=COHORT_SEED:raise ValueError('Requires original common900 cohort')
    runtime()
    import numpy as np, joblib, scipy, sklearn
    from scipy.special import softmax
    from threadpoolctl import threadpool_limits,threadpool_info
    import har_reference_test as registry
    import har_development_benchmark as shared
    registry.validate_root_lock(lock_path)
    frozen=registry.frozen_model_records()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=on_d(args.output) if args.output else ROOT/'results'/'review2_full_model_timing'/stamp
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(phase='POST_TEST_DESCRIPTIVE_COMPLETE_MODEL_TIMING',test_lock_sha256=lock_sha,
        scientific_parameters_changed=False,model_fits=0,temperature_fits=0,scaler_fits=0,
        raw_source_access=False,timing_seed=TIMING_SEED,repetitions=REPETITIONS,
        systems=list(SEEDS),families=list(FAMILIES),native_models=15,compiled_hgb_models=3,
        cases=900,planned_timed_rows=48600,cohort=cohort_record,
        test_cache_path=str(cache),test_cache_sha256=sha(cache),reference_outcomes_sha256=sha(prior_outcome),
        script_sha256=sha(__file__),frozen_models=frozen,
        timing_inclusions='Allocation; all18 real statistics/FFT blocks; saved float32 standardization; native family prediction; frozen temperature transformation; returned probability',
        timing_exclusions='Cache/file I/O; model loading; JIT; warmup; prior-array/cached-feature comparisons; labels; validation; result serialization',
        order='Each system uses its frozen eighteen-block HGB order for all five families; each case/repetition randomly mixes all eighteen systems/implementations',
        probability_conformance_tolerance=dict(LR=float(16*np.finfo(np.float32).eps),others=2e-12),
        probability_conformance_note='LR retains native float32 prediction: singleton versus stored batched BLAS rounding may differ. No probability is replaced or renormalized; require identical argmax.',
        predictor_dtype=dict(HGB='float64 input holding float32 feature/scaler values',RF='float32',ET='float32',LR='float32',SVC='float32'),
        formal_main_unchanged=True,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        processor=os.environ.get('PROCESSOR_IDENTIFIER'),D_free_bytes=shutil.disk_usage('D:\\').free)
    dump(out/'protocol_before_diagnostic.json',protocol)
    started=time.perf_counter()
    try:
        with np.load(cache,allow_pickle=False) as z:
            raw=z['raw_test'];features=z['features_test'];labels=z['labels_test'];subjects=z['subjects']
            if str(z['test_lock_sha256'])!=lock_sha:raise PermissionError('Embedded cache freeze mismatch')
        with np.load(prior_outcome,allow_pickle=False) as z:
            old_probability=z['probabilities'];old_seeds=z['seeds'];old_families=z['families']
            if not np.array_equal(z['labels'],labels) or not np.array_equal(z['subjects'],subjects):raise AssertionError('Reference ownership mismatch')
        if raw.shape!=(2947,9,128) or features.shape!=(2947,153) or old_probability.shape!=(3,5,2947,6):raise ValueError('HAR shape mismatch')
        if not np.array_equal(old_seeds,SEEDS) or not np.array_equal(old_families,FAMILIES[:5]):raise ValueError('Reference family/system order changed')
        cases=np.asarray(cases_input,np.int64)
        rng_cohort=np.random.default_rng(COHORT_SEED);recreated=[]
        for subject in np.unique(subjects):
            ix=np.flatnonzero(subjects==subject);recreated.extend(rng_cohort.choice(ix,min(100,len(ix)),replace=False))
        if not np.array_equal(cases,np.sort(recreated)):raise AssertionError('Cohort differs from original outcome-free rule')
        target=np.eye(6)[labels[cases]]
        pipelines=[];frozen_full_reference=[];formal_full_times=[]
        with threadpool_limits(limits=1):
            for record in frozen:
                si=SEEDS.index(record['seed']);fi=FAMILIES.index(record['family'])
                bundle=joblib.load(record['model_path']);model=bundle['model']
                if bundle['seed']!=record['seed'] or bundle['family']!=record['family'] or bundle['candidate']!=record['candidate']:raise ValueError('Model identity mismatch')
                if float(bundle['temperature'])!=record['chosen_temperature'] or not np.array_equal(model.classes_,np.arange(6)):raise ValueError('Frozen temperature/encoding mismatch')
                if getattr(model,'n_jobs',1) not in (None,1):raise ValueError('Native model is not configured for single-thread prediction')
                center,scale=bundle['center'],bundle['scale']
                if center.dtype!=np.float32 or scale.dtype!=np.float32 or center.shape!=(153,) or scale.shape!=(153,):raise ValueError('Saved scaler representation mismatch')
                blocks=shared.har.HARBlocks(raw,center,scale)
                order=np.asarray(lock['orders'][str(record['seed'])]['order'],np.int64)
                if not np.array_equal(np.sort(order),np.arange(18)):raise ValueError('All18 raw blocks required')
                expected_x=(features-center)/scale
                pipelines.append(dict(si=si,fi=fi,seed=record['seed'],family=record['family'],model=model,T=float(bundle['temperature']),blocks=blocks,order=order,x=expected_x,compiled=False))
                if record['family']=='HGB':
                    forest=shared.calibrated_forest(model,float(bundle['temperature']))
                    pipelines.append(dict(si=si,fi=5,seed=record['seed'],family='HGB_compiled',model=model,T=float(bundle['temperature']),blocks=blocks,order=order,x=expected_x,compiled=True,forest=forest))
                    with np.load(formal/f"seed{record['seed']}"/'timing_cohort_outcomes.npz',allow_pickle=False) as z:
                        if not np.array_equal(z['test_indices'],cases):raise AssertionError('Formal reference cohort changed')
                        frozen_full_reference.append(z['reference_probabilities'].copy())
                    official=read(formal/f"seed{record['seed']}"/'executed_results.json')
                    for r in official['timing_results']:
                        if r['name'] in ('native_full','compiled_full'):
                            formal_full_times.append(dict(seed=record['seed'],name=r['name'],complete_ms=r['complete_ms'],source='UNCHANGED_FORMAL_MAIN_MEASUREMENT'))
            if len(pipelines)!=18:raise AssertionError('Eighteen complete pipelines required')
            elapsed=np.empty((3,6,900,3),np.float64);p_store=np.empty((3,6,900,6),np.float64)
            max_error=np.zeros((3,6),np.float64);class_disagreements=np.zeros((3,6),np.int64)
            conformance_checked=np.zeros((3,6),np.int64)
            def execute(pipe,case):
                tick=time.perf_counter_ns()
                dtype=np.float64 if pipe['family'].startswith('HGB') else np.float32
                obs=np.zeros(153,dtype=dtype);known=np.zeros(18,np.bool_)
                for block in pipe['order']:pipe['blocks'].reveal(case,int(block),obs,known)
                if pipe['compiled']:prob=pipe['forest'].exact(obs)
                else:
                    scores=pipe['model']._raw_predict(obs[None]) if pipe['family']=='HGB' else np.log(np.clip(pipe['model'].predict_proba(obs[None]),1e-15,1.))
                    prob=softmax(scores/pipe['T'],axis=1)[0]
                end=time.perf_counter_ns()
                return prob,(end-tick)/1e6,obs
            def check(pipe,case,prob,obs):
                si,fi=pipe['si'],pipe['fi'];oldfi=0 if fi==5 else fi
                expected=old_probability[si,oldfi,case]
                tol=16*np.finfo(np.float32).eps if pipe['family']=='LR' else 2e-12
                error=float(np.max(np.abs(prob.astype(np.float64)-expected)))
                if not np.isfinite(prob).all() or np.any(prob<0) or abs(float(prob.sum())-1)>max(2e-12,tol):raise AssertionError('Native probability simplex failure')
                if error>tol:raise AssertionError(f"Prior probability mismatch {pipe['seed']} {pipe['family']} case{case}: {error} > {tol}")
                if not np.array_equal(obs.astype(np.float32),pipe['x'][case]):raise AssertionError('Real feature/scaler values differ from frozen cache')
                agreement=prob.argmax()==expected.argmax()
                if not agreement:raise AssertionError('Singleton prediction changes frozen class')
                return error
            # First common cohort window, independent of its label or prediction.
            for pipe in pipelines:
                prob,unused,obs=execute(pipe,int(cases[0]));check(pipe,int(cases[0]),prob,obs)
            rng=np.random.default_rng(TIMING_SEED);records=0
            with (out/'case_results.jsonl').open('x',encoding='utf-8') as stream:
                for ci,case in enumerate(cases):
                    for rep in range(REPETITIONS):
                        for index in rng.permutation(18):
                            pipe=pipelines[int(index)];si,fi=pipe['si'],pipe['fi']
                            prob,ms,obs=execute(pipe,int(case))
                            error=check(pipe,int(case),prob,obs)
                            if rep==0:p_store[si,fi,ci]=prob
                            elif not np.array_equal(p_store[si,fi,ci],prob):raise AssertionError('Repeated native outputs changed')
                            elapsed[si,fi,ci,rep]=ms;max_error[si,fi]=max(max_error[si,fi],error);conformance_checked[si,fi]+=1
                            stream.write(json.dumps(dict(seed=pipe['seed'],family=pipe['family'],case=int(case),cohort_position=ci,subject=int(subjects[case]),rep=rep,blocks=18,temperature=pipe['T'],complete_ms=ms,probability=np.asarray(prob).tolist(),prior_probability_error=error),allow_nan=False)+'\n')
                            records+=1
                    stream.flush()
                    if (ci+1)%100==0:print('REVIEW2_COMPLETE_RAW_TIMED',ci+1,'of900',flush=True)
            if records!=48600 or not np.all(conformance_checked==2700):raise AssertionError('Incomplete mixed timing coverage')
            risk=.5*np.square(p_store-target[None,None]).sum(-1)
            correct=p_store.argmax(-1)==labels[cases][None,None]
            per_model=[];pooled=[]
            for si,seed in enumerate(SEEDS):
                for fi,family in enumerate(FAMILIES):
                    per_model.append(dict(seed=seed,family=family,cases=900,repetitions=3,complete_ms=quantiles(np,elapsed[si,fi]),half_brier=float(risk[si,fi].mean()),accuracy=float(correct[si,fi].mean()),maximum_prior_probability_difference=float(max_error[si,fi]),class_disagreements=0))
                if np.max(np.abs(p_store[si,0]-frozen_full_reference[si]))>2e-12 or np.max(np.abs(p_store[si,5]-frozen_full_reference[si]))>2e-12:raise AssertionError('Formal complete probability mismatch')
            for fi,family in enumerate(FAMILIES):
                pooled.append(dict(family=family,systems=3,shared_cases_per_system=900,timed_observations=8100,complete_ms=quantiles(np,elapsed[:,fi]),half_brier=float(risk[:,fi].mean()),accuracy=float(correct[:,fi].mean()),systems_not_independent_cohorts=True))
            with (out/'outcomes.npz').open('xb') as f:
                np.savez_compressed(f,test_indices=cases,subjects=subjects[cases],labels=labels[cases],seeds=np.asarray(SEEDS),families=np.asarray(FAMILIES),probabilities=p_store,complete_ms=elapsed,half_brier=risk,correct=correct,maximum_prior_probability_error=max_error)
            result=dict(phase=protocol['phase'],status='COMPLETE',post_test_descriptive=True,model_fits=0,rows=records,elapsed_seconds=time.perf_counter()-started,per_model=per_model,pooled=pooled,formal_full_timings_unchanged=formal_full_times,threads=threadpool_info(),probability_checks=dict(maximum_errors=max_error.tolist(),all_repeated_outputs_identical=True,cached_feature_values_exact=True,reference_class_disagreements=0),main_results_replaced=False,primary_timing_not_rerun=True)
            if sha(lock_path)!=lock_sha:raise PermissionError('Root lock changed during diagnostic')
            for path,key in source_scripts.items():
                if sha(path)!=lock[key]:raise PermissionError('Frozen implementation changed during diagnostic')
            result['outputs']={name:dict(path=str(out/name),sha256=sha(out/name)) for name in ('outcomes.npz','case_results.jsonl','protocol_before_diagnostic.json')}
            dump(out/'executed_results.json',result)
            print('REVIEW2_FULL_MODEL_COMPLETE',str(out),json.dumps(pooled),flush=True)
    except BaseException as error:
        dump(out/'failure.json',dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started,policy='Preserve failure and partial records; no model, temperature or cohort retuning'))
        raise

if __name__=='__main__':main()
