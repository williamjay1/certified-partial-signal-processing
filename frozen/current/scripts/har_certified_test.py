"""Frozen held-out HAR certificate evaluation, requiring a root pre-test lock.

No training, model/preprocessing/order/schedule selection, ZIP opening, download
or statistical bootstrap is implemented. No test file is read by import/help.
"""
from __future__ import annotations
import os,sys,json,time,argparse,shutil,traceback
from pathlib import Path
for k in ['ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
sys.pycache_prefix=str(ROOT/'temp'/'pycache_har_certified_test')
for k in ['TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER']:os.environ[k]=str(ROOT/'temp'/'har_certified_test')
import har_development_benchmark as hd
import har_checkpoint_development as cp
import har_stopping_controls as controls
import numpy as np
import joblib
from threadpoolctl import threadpool_info,threadpool_limits

STAGE='FROZEN_BEFORE_TEST_ACCESS'
SELECTION=ROOT/'results'/'har_model_selection'/'20261002T052208Z_75415f24'/'selected_hgb.json'
SCHEDULE=ROOT/'results'/'har_checkpoint_development'/'schedule_development_v1'/'selected_schedule.json'
DEVELOPMENT=ROOT/'results'/'har_compute_development'/'corrected_execution'
COHORT_SEED=7302026
TIMING_REPETITIONS=3
CONTROL_METHODS=('score_separation_argmax','fixed_prefix_mean_completion','entropy_mean_completion')

def json_read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def on_d(path,existing=False):
    p=Path(path).resolve(strict=existing)
    if p.drive.upper()!='D:':raise ValueError('Test work/results require an absolute D-drive path')
    return p

def require_equal(lock,key,value):
    if key not in lock or lock[key]!=value:raise PermissionError('Frozen root lock mismatch: '+key)

def validate_lock(lock_path,control_state_path):
    # All these files are training/development code or records, never test data.
    lock=json_read(lock_path)
    require_equal(lock,'stage',STAGE)
    require_equal(lock,'har_method_frozen',True)
    require_equal(lock,'official_test_access_authorized',True)
    hashes={'selected_hgb_sha256':SELECTION,'train_cache_sha256':hd.CACHE,
        'har_compute_script_sha256':Path(hd.__file__),'checkpoint_script_sha256':Path(cp.__file__),
        'selected_schedule_sha256':SCHEDULE,'certificate_kernel_sha256':Path(hd.ck.__file__),
        'leaf_helper_sha256':Path(hd.lm.__file__),'har_operator_sha256':Path(hd.har.__file__),
        'certified_test_script_sha256':Path(__file__),
        'test_loader_sha256':ROOT/'research'/'prepare_har_test.py',
        'stopping_control_script_sha256':Path(controls.__file__),
        'stopping_control_state_sha256':control_state_path,
        'control_design_sha256':ROOT/'research'/'control_design.py'}
    for key,path in hashes.items():require_equal(lock,key,hd.digest(path))
    orders={str(s):hd.digest(DEVELOPMENT/f'seed{s}'/'frozen_order_and_cohort_before_development.json') for s in hd.SEEDS}
    require_equal(lock,'frozen_order_sha256',orders)
    schedule=json_read(SCHEDULE)
    if schedule['stage']!='DEVELOPMENT_SELECTED_COMMON_CHECKPOINT_SCHEDULE':raise PermissionError('Schedule selection is incomplete')
    require_equal(lock,'checkpoints',schedule['checkpoints'])
    require_equal(lock,'seeds',list(hd.SEEDS));require_equal(lock,'deltas',list(hd.DELTAS))
    require_equal(lock,'primary_delta',.025);require_equal(lock,'timing_cohort_seed',COHORT_SEED)
    require_equal(lock,'timing_max_windows_per_subject',100)
    require_equal(lock,'timing_repetitions',TIMING_REPETITIONS)
    require_equal(lock,'main_specifications',17)
    require_equal(lock,'additional_stopping_controls',list(CONTROL_METHODS))
    require_equal(lock,'uniform_A_primary_schedule_control',schedule['schedule']!='A')
    control_state=controls.load_frozen_control_state(control_state_path,lock['stopping_control_state_sha256'])
    if control_state['checkpoints']!=schedule['checkpoints']:raise PermissionError('Control/main checkpoint schedules disagree')
    selected=json_read(SELECTION)
    if set(selected)!=set(map(str,hd.SEEDS)):raise ValueError('All three model choices must be frozen')
    bundles={}
    for seed in hd.SEEDS:
        entry=selected[str(seed)]
        if entry['status']!='development_selected_frozen_hgb':raise PermissionError('Unfrozen selected model')
        p=on_d(entry['model_path'],True)
        if hd.digest(p)!=entry['model_sha256']:raise PermissionError('Frozen model bytes changed')
        bundles[seed]=joblib.load(p)
        if float(bundles[seed]['temperature'])!=float(entry['chosen_temperature']):raise ValueError('Model temperature disagrees with frozen choice')
    return lock,hd.digest(lock_path),selected,schedule,bundles,control_state

def cohort_from_subjects(subjects):
    """Outcome-free shared cohort; only IDs are accepted by the function."""
    rng=np.random.default_rng(COHORT_SEED);chosen=[]
    for subject in np.unique(subjects):
        indices=np.flatnonzero(subjects==subject)
        chosen.extend(rng.choice(indices,min(100,len(indices)),replace=False).tolist())
    return np.sort(np.asarray(chosen,np.int64))

def specifications(include_uniform):
    specs=[dict(name=name,delta=delta,canonical=True,kind='main') for name,delta in hd.specifications()]
    if include_uniform:specs.append(dict(name='uniform_A_tight_minimax',delta=.025,canonical=False,kind='uniform_schedule'))
    specs.extend(dict(name=name,delta=None,canonical=False,kind='stopping_control') for name in CONTROL_METHODS)
    return specs

def execute(mainpipe,uniformpipe,case,spec,trace=False,settings=None):
    name=spec['name'];delta=spec['delta']
    if name in hd.VARIANTS:return mainpipe.partial(case,delta,hd.VARIANTS[name],trace=trace)
    if name=='uniform_A_tight_minimax':return uniformpipe.partial(case,.025,0,trace=trace)
    if name in CONTROL_METHODS:return controls.execute_method(mainpipe,case,name,settings)
    return mainpipe.full(case,name=='compiled_full')

def check_excess(probability,reference,certificate):
    excess=.5*(np.sum(probability*probability,axis=-1)-np.sum(reference*reference,axis=-1))[...,None]-probability+reference
    if certificate is None:return excess,None
    violation=excess.max(axis=-1)-certificate
    if float(np.max(violation))>=2e-12:raise AssertionError('Every-label certificate conformance failure')
    return excess,violation

def outcomes(mainpipe,uniformpipe,cases,q,labels,specs,settings):
    probability=np.empty((len(specs),len(cases),q.shape[1]));certificate=np.empty((len(specs),len(cases)))
    blocks=np.empty_like(certificate,np.int64);fallback=np.empty_like(certificate,np.bool_);summaries=[]
    for j,spec in enumerate(specs):
        for i,case in enumerate(cases):
            r=execute(mainpipe,uniformpipe,int(case),spec,settings=settings)
            probability[j,i]=r['probability'];certificate[j,i]=np.nan if r['certificate'] is None else r['certificate'];blocks[j,i]=r['blocks'];fallback[j,i]=r['fallback']
        excess,violation=check_excess(probability[j],q[cases],None if spec['name'] in CONTROL_METHODS else certificate[j])
        class_disagreement=int(np.sum(probability[j].argmax(1)!=q[cases].argmax(1)))
        if spec['name']=='score_separation_argmax' and class_disagreement:raise AssertionError('Class-preserving control disagrees with frozen full reference')
        summaries.append(dict(**spec,cases=len(cases),**hd.metrics(probability[j],labels[cases]),
            mean_block_count=float(blocks[j].mean()),full_fallback_rate=float(fallback[j].mean()),
            mean_reference_loss_excess=float((hd.loss(probability[j],labels[cases])-hd.loss(q[cases],labels[cases])).mean()),
            maximum_every_label_excess=float(excess.max()),maximum_certificate_violation_signed=None if violation is None else float(violation.max()),
            reference_class_disagreements=class_disagreement,every_label_delta025_violations=int(np.sum(excess.max(1)>.025+2e-12))))
    return dict(probabilities=probability,certificates=certificate,block_counts=blocks,fallback=fallback),summaries

def save_outcomes(path,cases,subjects,labels,q,specs,arrays,kind):
    chosen=np.flatnonzero(np.asarray([spec['kind']==kind for spec in specs]))
    if len(chosen)==0:return
    selected=[specs[j] for j in chosen]
    with path.open('xb') as stream:
        np.savez_compressed(stream,test_indices=cases,subjects=subjects[cases],labels=labels[cases],
            method_names=np.asarray([spec['name'] for spec in selected]),
            deltas=np.asarray([np.nan if spec['delta'] is None else spec['delta'] for spec in selected]),
            reference_probabilities=q[cases],**{key:value[chosen] for key,value in arrays.items()})

def timed(mainpipe,uniformpipe,cases,subjects,q,labels,specs,seed,folder,settings):
    rng=np.random.default_rng(20268002+seed);rows=[]
    with (folder/'timing_case_results.jsonl').open('x',encoding='utf-8') as stream:
        for i,case in enumerate(cases):
            for rep in range(TIMING_REPETITIONS):
                for j in rng.permutation(len(specs)):
                    spec=specs[j];r=execute(mainpipe,uniformpipe,int(case),spec,settings=settings);p=r.pop('probability')
                    # No reference, label, test statistics or serialization enters the online policy or complete timer.
                    check_excess(p,q[case],r['certificate'])
                    if spec['name']=='score_separation_argmax' and np.argmax(p)!=np.argmax(q[case]):raise AssertionError('Class-preserving control conformance failed')
                    row=dict(seed=seed,case=int(case),test_index=int(case),subject=int(subjects[case]),rep=rep,
                        **spec,probability=p.tolist(),**r)
                    rows.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            if (i+1)%100==0:print('HAR_FROZEN_TEST_TIMED',seed,i+1,'of',len(cases),flush=True)
    times=[];cohort_outcomes=[];p_arrays=[];b_arrays=[];n_arrays=[];fb_arrays=[]
    for spec in specs:
        chosen=[r for r in rows if r['name']==spec['name'] and r['delta']==spec['delta']]
        first=[r for r in chosen if r['rep']==0];ids=np.asarray([r['case'] for r in first],int)
        if not np.array_equal(ids,cases):raise AssertionError('Methods did not use exactly the shared timing cohort')
        p=np.asarray([r['probability'] for r in first]);b=np.asarray([np.nan if r['certificate'] is None else r['certificate'] for r in first])
        excess,violation=check_excess(p,q[cases],None if spec['name'] in CONTROL_METHODS else b)
        p_arrays.append(p);b_arrays.append(b);n_arrays.append([r['blocks'] for r in first]);fb_arrays.append([r['fallback'] for r in first])
        times.append(dict(**spec,cases=len(cases),repetitions=TIMING_REPETITIONS,
            complete_ms=hd.timing_summary([r['complete_ms'] for r in chosen]),
            mean_block_count=float(np.mean([r['blocks'] for r in first])),full_fallback_rate=float(np.mean([r['fallback'] for r in first]))))
        cohort_outcomes.append(dict(**spec,cases=len(cases),**hd.metrics(p,labels[cases]),
            mean_reference_loss_excess=float((hd.loss(p,labels[cases])-hd.loss(q[cases],labels[cases])).mean()),
            maximum_every_label_excess=float(excess.max()),maximum_certificate_violation_signed=None if violation is None else float(violation.max()),
            reference_class_disagreements=int(np.sum(p.argmax(1)!=q[cases].argmax(1))),
            every_label_delta025_violations=int(np.sum(excess.max(1)>.025+2e-12))))
    faster=min([r for r in times if r['name'] in ('native_full','compiled_full')],key=lambda r:r['complete_ms']['mean'])
    for r in times:r.update(faster_full_comparator=faster['name'],complete_mean_time_reduction_vs_faster_full=1-r['complete_ms']['mean']/faster['complete_ms']['mean'])
    arrays=dict(probabilities=np.asarray(p_arrays),certificates=np.asarray(b_arrays),block_counts=np.asarray(n_arrays),fallback=np.asarray(fb_arrays))
    return times,cohort_outcomes,arrays

def main(args):
    started=time.perf_counter();lock_path=on_d(args.test_lock,True)
    control_state_path=on_d(args.stopping_control_state,True)
    lock,lock_sha,selected,schedule,bundles,control_state=validate_lock(lock_path,control_state_path)
    cache=on_d(args.test_cache,True);manifest_path=on_d(args.test_preparation_manifest,True);out=on_d(args.output)
    if out.exists():raise FileExistsError('Existing test execution directory is preserved; no silent rerun')
    if shutil.disk_usage('D:\\').free<2*1024**3:raise RuntimeError('Insufficient D space for immutable test outputs')
    manifest=json_read(manifest_path)
    if Path(manifest['cache']).resolve()!=cache or manifest['cache_sha256']!=hd.digest(cache):raise PermissionError('Test cache manifest mismatch')
    if manifest['test_lock_sha256']!=lock_sha or manifest['train_cache_sha256']!=hd.digest(hd.CACHE):raise PermissionError('Test cache was not prepared under this frozen lock/train cache')
    if manifest['test_loader_sha256']!=lock['test_loader_sha256'] or manifest['training_operator_script_sha256']!=lock['har_operator_sha256']:raise PermissionError('Test preparation operator mismatch')
    # Access boundary: all deployment/model/provenance checks above precede test arrays.
    with np.load(hd.CACHE,allow_pickle=False) as z:training={k:z[k] for k in z.files if k=='subjects' or k.startswith('scaler_') or k.startswith('fit_indices_') or k.startswith('development_indices_')}
    if hd.digest(lock_path)!=lock_sha:raise PermissionError('Root lock changed before test cache access')
    with np.load(cache,allow_pickle=False) as z:
        if str(z['test_lock_sha256'])!=lock_sha or str(z['train_cache_sha256'])!=lock['train_cache_sha256']:raise PermissionError('Cache-embedded freeze provenance mismatch')
        raw=z['raw_test'];features=z['features_test'];subjects=z['subjects'];labels=z['labels_test'];test_indices=z['original_test_row']
        if not np.array_equal(z['group_index'],hd.har.GROUP_INDEX) or not np.array_equal(z['feature_names'],np.asarray(hd.har.FEATURE_NAMES)):raise ValueError('Training/test operator coordinates differ')
    if raw.shape!=(2947,9,128) or features.shape!=(2947,153) or subjects.shape!=(2947,) or labels.shape!=(2947,):raise ValueError('Unexpected official held-out release shape')
    if not np.array_equal(test_indices,np.arange(2947)) or len(np.unique(subjects))!=9:raise ValueError('Official test window identifiers/subjects invalid')
    if np.intersect1d(training['subjects'],subjects).size:raise AssertionError('Official outer subject leakage')
    if not np.array_equal(np.sort(np.r_[np.unique(training['subjects']),np.unique(subjects)]),np.arange(1,31)):raise AssertionError('Outer split must cover all 30 distinct subjects')
    cohort=cohort_from_subjects(subjects)
    if len(cohort)!=900:raise ValueError('Locked 900-case test timing cohort unavailable; do not change sample rule')
    all_cases=np.arange(2947);include_uniform=schedule['schedule']!='A';specs=specifications(include_uniform)
    configuration=dict(phase='FROZEN_OFFICIAL_HELD_OUT_SUBJECT_TEST',new_fits=0,test_lock_sha256=lock_sha,
        test_cache_sha256=hd.digest(cache),test_preparation_manifest_sha256=hd.digest(manifest_path),
        scripts_frozen_before_access=True,seeds=list(hd.SEEDS),deltas=list(hd.DELTAS),primary_delta=.025,
        selected_schedule=schedule['schedule'],checkpoints=schedule['checkpoints'],main_specifications=17,
        auxiliary_uniform_A=include_uniform,randomized_timing_specifications=len(specs),
        additional_stopping_controls=list(CONTROL_METHODS),stopping_control_state_sha256=hd.digest(control_state_path),
        full_outcome_cases=2947,main_risk_time_same_cohort_cases=900,
        timing_cohort_seed=COHORT_SEED,timing_max_windows_per_subject=100,timing_repetitions=TIMING_REPETITIONS,
        official_test_subjects=np.unique(subjects).tolist(),threadpools=threadpool_info(),D_free_bytes=shutil.disk_usage('D:\\').free,
        timer_start='Released raw window in host memory; all selected block extraction/scaling/inference/certification/fallback included',
        timer_exclusions='Acquisition, UCI prefilter/window processing, archive/cache load, reference validation, serialization and separate profiles',
        no_test_model_order_tolerance_schedule_selection=True)
    hd.dump(out/'configuration_before_execution.json',configuration)
    hd.dump(out/'frozen_shared_timing_cohort_before_predictions.json',dict(test_indices=cohort.tolist(),seed=COHORT_SEED,
        selected_using_subjects_and_window_indices_only=True,labels_not_used_in_selection=True,
        subject_counts={str(s):int(np.sum(subjects[cohort]==s)) for s in np.unique(subjects)}))
    results=[]
    try:
        for seed in hd.SEEDS:
            obj=bundles[seed];model=obj['model'];T=float(obj['temperature']);folder=out/f'seed{seed}';folder.mkdir()
            center=training[f'scaler_center_seed{seed}'];scale=training[f'scaler_scale_seed{seed}']
            if not np.array_equal(center,obj['center']) or not np.array_equal(scale,obj['scale']):raise ValueError('Frozen fit scaler/model mismatch')
            if not np.array_equal(training[f'fit_indices_seed{seed}'],obj['fit_indices']) or not np.array_equal(training[f'development_indices_seed{seed}'],obj['development_indices']):raise ValueError('Frozen fitting/development ownership changed')
            prior=json_read(DEVELOPMENT/f'seed{seed}'/'frozen_order_and_cohort_before_development.json');order=prior['order']
            if prior['temperature']!=T or prior['model_sha256']!=selected[str(seed)]['model_sha256']:raise ValueError('Frozen order model/temperature mismatch')
            x=(features-center)/scale;forest=hd.calibrated_forest(model,T)
            rawblocks=hd.har.HARBlocks(raw,center,scale);cachedblocks=hd.CachedBlocks(x,hd.har.OFFSETS)
            settings=control_state['seeds'][str(seed)]
            if settings['model_sha256']!=selected[str(seed)]['model_sha256'] or settings['temperature']!=T or settings['order']!=order or settings['checkpoints']!=schedule['checkpoints']:raise PermissionError('Frozen controls refer to a different deployment pipeline')
            pipe=controls.StoppingPipeline(model,forest,rawblocks,order,18,T,schedule['checkpoints'])
            cached=controls.StoppingPipeline(model,forest,cachedblocks,order,18,T,schedule['checkpoints'])
            uniform=cp.CheckpointPipeline(model,forest,rawblocks,order,18,T,cp.SCHEDULES['A'])
            uniform_cached=cp.CheckpointPipeline(model,forest,cachedblocks,order,18,T,cp.SCHEDULES['A'])
            # Reference/labels remain outside every online policy input and primary complete timer.
            q=np.stack([forest.exact(np.asarray(x[case],np.float64)) for case in all_cases])
            native=hd.probabilities(model._raw_predict(np.asarray(x,np.float64))/T);diff=float(np.max(np.abs(q-native)))
            if diff>=2e-12:raise AssertionError('Native/compiled frozen full semantic conformance failure')
            check=hd.correctness(pipe,cached,x,q,all_cases[:24])
            if include_uniform:
                for case in all_cases[:24]:
                    a=uniform.partial(int(case),.025,0);b=uniform_cached.partial(int(case),.025,0)
                    if a['blocks']!=b['blocks'] or a['fallback']!=b['fallback'] or np.max(np.abs(a['probability']-b['probability']))>=2e-12:raise AssertionError('Auxiliary uniform schedule raw/cache conformance failure')
                    check_excess(a['probability'],q[case],a['certificate'])
            hd.dump(folder/'conformance_before_primary_timing.json',dict(native_compiled_all2947_max_difference=diff,raw_cache=check,all_possible_labels_checked=True))
            # The additional controls see only the same partial observer; check cache/raw paths before benchmarking.
            for name in CONTROL_METHODS:
                for case in all_cases[:24]:
                    a=controls.execute_method(pipe,int(case),name,settings);b=controls.execute_method(cached,int(case),name,settings)
                    if a['blocks']!=b['blocks'] or a['fallback']!=b['fallback'] or np.max(np.abs(a['probability']-b['probability']))>=2e-12:raise AssertionError('Stopping control raw/cache conformance failed')
            for spec in specs:execute(pipe,uniform,int(all_cases[0]),spec,settings=settings)
            arrays,full_outcomes=outcomes(cached,uniform_cached,all_cases,q,labels,specs,settings)
            save_outcomes(folder/'all_test_outcomes.npz',all_cases,subjects,labels,q,specs,arrays,'main')
            save_outcomes(folder/'auxiliary_uniform_schedule_all_test_outcomes.npz',all_cases,subjects,labels,q,specs,arrays,'uniform_schedule')
            save_outcomes(folder/'stopping_controls_all_test_outcomes.npz',all_cases,subjects,labels,q,specs,arrays,'stopping_control')
            hd.dump(folder/'all2947_outcome_results.json',full_outcomes)
            times,cohort_outcomes,cohort_arrays=timed(pipe,uniform,cohort,subjects,q,labels,specs,seed,folder,settings)
            save_outcomes(folder/'timing_cohort_outcomes.npz',cohort,subjects,labels,q,specs,cohort_arrays,'main')
            save_outcomes(folder/'auxiliary_uniform_schedule_timing_cohort_outcomes.npz',cohort,subjects,labels,q,specs,cohort_arrays,'uniform_schedule')
            save_outcomes(folder/'stopping_controls_timing_cohort_outcomes.npz',cohort,subjects,labels,q,specs,cohort_arrays,'stopping_control')
            profiles=[]
            for case in cohort[:12]:
                r=pipe.partial(int(case),.025,0,profile=True);r.pop('probability');r['case']=int(case);r['subject']=int(subjects[case]);profiles.append(r)
            hd.dump(folder/'separate_component_profiles.json',profiles)
            result=dict(seed=seed,phase='INDEPENDENT_OFFICIAL_TEST_WITH_PRIOR_DEVELOPMENT_SELECTION',temperature=T,
                model_candidate=selected[str(seed)]['candidate'],order=order,checkpoints=schedule['checkpoints'],
                full2947_reference=hd.metrics(q,labels),full2947_outcomes=full_outcomes,
                timing_cohort900_reference=hd.metrics(q[cohort],labels[cohort]),timing_cohort900_outcomes=cohort_outcomes,
                timing_results=times,conformance=check,native_compiled_max_difference=diff,
                full_outcomes_selected_cache_only_untimed=True,all_complete_timers_real_raw=True,
                primary_risk_and_time_same900_cohort=True)
            hd.dump(folder/'executed_results.json',result);results.append(result)
            print('HAR_FROZEN_TEST_SEED_COMPLETE',seed,json.dumps([r for r in times if r['delta']==.025 or r['name'] in ('native_full','compiled_full')]),flush=True)
        final=dict(configuration=configuration,seeds=results,total_seconds=time.perf_counter()-started,
            no_post_test_selection=True,all_three_seeds_retained=True)
        hd.dump(out/'executed_results.json',final)
        print('HAR_FROZEN_TEST_COMPLETE',json.dumps({'total_seconds':final['total_seconds'],'seeds':list(hd.SEEDS)}),flush=True)
    except BaseException as error:
        hd.dump(out/'technical_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            preserved_partial_results=True,elapsed_seconds=time.perf_counter()-started,no_silent_rerun=True))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-lock',type=Path,required=True)
    parser.add_argument('--test-cache',type=Path,required=True)
    parser.add_argument('--test-preparation-manifest',type=Path,required=True)
    parser.add_argument('--stopping-control-state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args)
