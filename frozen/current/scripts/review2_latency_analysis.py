"""Post-test descriptive latency analysis and a separately instrumented profile.

Never changes a frozen model, prediction policy, checkpoint, main timing record,
or manuscript. Profile selections depend only on the original timed-cohort case
IDs and subject IDs. New profile times do not replace the formal endpoint.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
for key in ('ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
sys.pycache_prefix=str(ROOT/'temp'/'pycache_review2_latency')
for key in ('TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER'):
    os.environ[key]=str(ROOT/'temp'/'review2_latency')
sys.path.insert(0,r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages')
sys.path.insert(0,r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_core')
import numpy as np

FORMAL=ROOT/'results'/'har_certified_test'/'locked_execution_v1'
OUTPUT=ROOT/'results'/'review2_latency_analysis'
SEEDS=(17,29,43)
PROFILE_SELECTION_SEED=9302026
PROFILE_CASES_PER_SUBJECT=4
PROFILE_REPETITIONS=3
PRIMARY_DELTA=.025

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False),encoding='utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def summary(x):
    a=np.asarray(x,float)
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),minimum=float(a.min()),maximum=float(a.max()))
def paired_summary(rows):
    return dict(cases=len(rows),systems=sorted(set(r['seed'] for r in rows)),
        subjects=sorted(set(r['subject'] for r in rows)),
        partial_ms=summary([r['partial_mean_ms'] for r in rows]),
        full_ms=summary([r['full_mean_ms'] for r in rows]),
        paired_increment_ms=summary([r['paired_increment_mean_ms'] for r in rows]),
        fraction_partial_slower=float(np.mean([r['paired_increment_mean_ms']>0 for r in rows])),
        mean_blocks=float(np.mean([r['blocks'] for r in rows])),
        fallback_fraction=float(np.mean([r['fallback'] for r in rows])),
        time_reduction=1.-float(np.mean([r['partial_mean_ms'] for r in rows]))/float(np.mean([r['full_mean_ms'] for r in rows])))

def analyze():
    out=OUTPUT/'saved_output_v1'
    if out.exists():raise FileExistsError('Preserve existing analysis; do not silently overwrite')
    if __import__('shutil').disk_usage('D:\\').free<1024**3:raise RuntimeError('Insufficient D space')
    out.mkdir(parents=True)
    case_rows=[];call_rows=[];schedule_rows=[];tail_rows=[];source_hashes={};cohort=None
    for seed in SEEDS:
        path=FORMAL/f'seed{seed}'/'timing_case_results.jsonl';source_hashes[str(path)]=digest(path)
        allrows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        selected={}
        for row in allrows:
            name=row['name']
            if (name in ('tight_minimax','uniform_A_tight_minimax') and row['delta']==PRIMARY_DELTA) or name=='compiled_full':
                k=(int(row['case']),int(row['rep']),name)
                if k in selected:raise AssertionError('Duplicate formal timing row')
                selected[k]=row
        cases=sorted(set(k[0] for k in selected));assert len(cases)==900
        if cohort is None:cohort=cases
        else:assert cases==cohort
        local_calls=[]
        for case in cases:
            primary=[selected[(case,rep,'tight_minimax')] for rep in range(3)]
            full=[selected[(case,rep,'compiled_full')] for rep in range(3)]
            uniform=[selected[(case,rep,'uniform_A_tight_minimax')] for rep in range(3)]
            for group in (primary,full,uniform):
                assert len({(int(r['subject']),int(r['blocks']),bool(r['fallback'])) for r in group})==1
            t=np.asarray([r['complete_ms'] for r in primary]);f=np.asarray([r['complete_ms'] for r in full]);u=np.asarray([r['complete_ms'] for r in uniform])
            row=dict(seed=seed,case=case,subject=int(primary[0]['subject']),blocks=int(primary[0]['blocks']),
                fallback=bool(primary[0]['fallback']),collapsed=bool(primary[0]['collapsed']),
                partial_mean_ms=float(t.mean()),full_mean_ms=float(f.mean()),paired_increment_mean_ms=float((t-f).mean()),
                partial_median_ms=float(np.median(t)),full_median_ms=float(np.median(f)),
                uniform_A_mean_ms=float(u.mean()),C_minus_A_ms=float((t-u).mean()),uniform_A_blocks=int(uniform[0]['blocks']))
            case_rows.append(row)
            schedule_rows.append(dict(seed=seed,case=case,subject=row['subject'],C_blocks=row['blocks'],A_blocks=row['uniform_A_blocks'],
                C_mean_ms=float(t.mean()),A_mean_ms=float(u.mean()),compiled_full_mean_ms=float(f.mean()),C_minus_A_ms=float((t-u).mean())))
            for rep in range(3):
                r=dict(seed=seed,case=case,subject=row['subject'],rep=rep,blocks=row['blocks'],fallback=row['fallback'],
                    partial_ms=float(t[rep]),full_ms=float(f[rep]),paired_increment_ms=float(t[rep]-f[rep]))
                call_rows.append(r);local_calls.append(r)
        for anchor in ('partial','full'):
            cutoff=float(np.quantile([r[anchor+'_ms'] for r in local_calls],.99))
            tail=[r for r in local_calls if r[anchor+'_ms']>=cutoff]
            tail_rows.append(dict(seed=seed,tail_anchor=anchor,quantile=.99,threshold_ms=cutoff,selected_calls=len(tail),
                distinct_cases=len({r['case'] for r in tail}),fallback_fraction=float(np.mean([r['fallback'] for r in tail])),
                mean_blocks=float(np.mean([r['blocks'] for r in tail])),
                partial_ms=summary([r['partial_ms'] for r in tail]),full_ms=summary([r['full_ms'] for r in tail]),
                paired_increment_ms=summary([r['paired_increment_ms'] for r in tail]),
                fraction_partial_slower=float(np.mean([r['paired_increment_ms']>0 for r in tail])),
                subject_call_counts={str(s):sum(r['subject']==s for r in tail) for s in sorted({r['subject'] for r in local_calls})}))
    groups=[]
    for key in ('seed','subject','blocks','fallback'):
        for value in sorted(set(r[key] for r in case_rows)):
            rows=[r for r in case_rows if r[key]==value]
            groups.append(dict(grouping=key,value=value,**paired_summary(rows)))
    for seed in SEEDS:
        for subject in sorted(set(r['subject'] for r in case_rows)):
            rows=[r for r in case_rows if r['seed']==seed and r['subject']==subject]
            groups.append(dict(grouping='system_subject',value={'seed':seed,'subject':subject},**paired_summary(rows)))
        for blocks in (12,16,18):
            rows=[r for r in case_rows if r['seed']==seed and r['blocks']==blocks]
            if rows:groups.append(dict(grouping='system_blocks',value={'seed':seed,'blocks':blocks},**paired_summary(rows)))
    schedule=[]
    for seed in (*SEEDS,None):
        rows=[r for r in schedule_rows if seed is None or r['seed']==seed]
        schedule.append(dict(seed=seed,system_case_evaluations=len(rows),A_mean_ms=float(np.mean([r['A_mean_ms'] for r in rows])),
            C_mean_ms=float(np.mean([r['C_mean_ms'] for r in rows])),full_mean_ms=float(np.mean([r['compiled_full_mean_ms'] for r in rows])),
            C_minus_A_ms=summary([r['C_minus_A_ms'] for r in rows]),
            A_blocks=float(np.mean([r['A_blocks'] for r in rows])),C_blocks=float(np.mean([r['C_blocks'] for r in rows])),
            C_faster_case_fraction=float(np.mean([r['C_minus_A_ms']<0 for r in rows]))))
    # Raw-call empirical CDF, with the original three repeats explicitly retained.
    cdf=[]
    for seed in SEEDS:
        rows=[r for r in call_rows if r['seed']==seed]
        for method in ('partial','full'):
            ordered=np.sort([r[method+'_ms'] for r in rows])
            for q in np.linspace(0,1,201):cdf.append(dict(seed=seed,method=method,quantile=float(q),time_ms=float(np.quantile(ordered,q))))
    save_csv(out/'paired_case_data.csv',case_rows);save_csv(out/'paired_call_data.csv',call_rows)
    save_csv(out/'checkpoint_A_C_case_data.csv',schedule_rows);save_csv(out/'empirical_cdf_quantiles.csv',cdf)
    result=dict(stage='POST_TEST_DESCRIPTIVE_LATENCY_ANALYSIS_FROM_IMMUTABLE_FORMAL_RECORDS',new_timing=False,
        new_fits=0,policy_changes=0,primary_delta=PRIMARY_DELTA,seeds=list(SEEDS),unique_timed_windows=900,
        shared_subjects=sorted(set(r['subject'] for r in case_rows)),system_case_outputs=len(case_rows),raw_call_pairs=len(call_rows),
        source_sha256=source_hashes,all_cases=paired_summary(case_rows),groups=groups,p99_tail_analysis=tail_rows,
        checkpoint_A_C=schedule,interpretation_limits=[
        'Group times use per-case means of the original three repeats; P99-tail rows select raw timing calls by each system and anchor.',
        'Stopping blocks and fallback are algorithm outcomes, not randomized treatments; descriptive differences do not identify their causal latency effects.',
        'Partial-tail and full-tail membership is selected after observing latency for diagnosis; it never changes policy or profile sampling.',
        'Case/repetition pairing shares the input but method order was randomized; jitter can affect either execution.',
        'Systems share nine subjects; case rows, repeats and 27 system-subject combinations are not independent people.'])
    write(out/'latency_analysis.json',result)
    write(out/'plot_specifications.json',dict(figures=[
        dict(name='complete_latency_cdf',data='empirical_cdf_quantiles.csv',x='time_ms',y='quantile',facet='seed',color='method',annotations=[.95,.99]),
        dict(name='paired_latency_by_processing_outcome',data='paired_case_data.csv',x='blocks',y='paired_increment_mean_ms',facet='seed',color='fallback',reference_line=0),
        dict(name='subject_fallback_and_paired_increment',data='paired_case_data.csv',aggregate_by=['seed','subject'],x='fallback fraction',y='paired increment mean',note='Descriptive association, not causal'),
        dict(name='same_case_checkpoint_A_C',data='checkpoint_A_C_case_data.csv',x='A_blocks',y='C_minus_A_ms',facet='seed',reference_line=0)]))
    print(json.dumps({'analysis':str(out),'all_cases':result['all_cases'],'tails':tail_rows},indent=2))

def robustness():
    """Posthoc timing-estimand sensitivity; retain every window and every run."""
    out=OUTPUT/'saved_output_v1'/'repeat_median_sensitivity.json'
    if out.exists():raise FileExistsError('Preserve existing diagnostic')
    rows=list(csv.DictReader((OUTPUT/'saved_output_v1'/'paired_case_data.csv').open(encoding='utf-8')))
    calls=list(csv.DictReader((OUTPUT/'saved_output_v1'/'paired_call_data.csv').open(encoding='utf-8')))
    groups=[]
    for seed in (*SEEDS,None):
        r=[x for x in rows if seed is None or int(x['seed'])==seed]
        p=float(np.mean([float(x['partial_median_ms']) for x in r]))
        f=float(np.mean([float(x['full_median_ms']) for x in r]))
        pmean=float(np.mean([float(x['partial_mean_ms']) for x in r]))
        fmean=float(np.mean([float(x['full_mean_ms']) for x in r]))
        groups.append(dict(seed=seed,system_case_outputs=len(r),mean_of_repeat_medians_partial_ms=p,
            mean_of_repeat_medians_full_ms=f,repeat_median_time_reduction=1-p/f,
            original_mean_partial_ms=pmean,original_mean_full_ms=fmean,original_mean_time_reduction=1-pmean/fmean))
    extremes={}
    for name in ('partial_ms','full_ms'):
        extremes[name]=sorted(calls,key=lambda x:float(x[name]),reverse=True)[:5]
    result=dict(stage='POSTHOC_REPEAT_MEDIAN_SENSITIVITY_TRIGGERED_BY_OBSERVED_LARGE_TIMING_CALL',
        main_endpoint_replaced=False,case_removal=False,repeated_runs_removed=False,
        observations='A compiled-full call of72.1447ms was observed in formal seed29. This diagnostic keeps every window/repetition and changes only the summary functional.',
        interpretation='Mean of three-repeat medians is a different execution-time estimand. It suppresses isolated repetition jitter; it does not prove whether a call was scheduling noise or model work. Do not replace or relabel the primary all-call mean.',
        groups=groups,largest_calls_diagnostic=extremes)
    write(out,result);print(json.dumps(result,indent=2))

def profile():
    # Heavy imports and JIT only enter this separately authorized diagnostic.
    sys.path.insert(0,str(ROOT/'research'))
    import har_development_benchmark as hd
    import har_checkpoint_development as cp
    import joblib
    from threadpoolctl import threadpool_limits,threadpool_info
    out=OUTPUT/'component_profile_v1'
    if out.exists():raise FileExistsError('Preserve profile; no silent retry/overwrite')
    out.mkdir(parents=True)
    lock=read(ROOT/'research'/'FROZEN_TEST_LOCK_20261002.json')
    for path,sha in lock['scripts'].items():
        if digest(path)!=sha:raise AssertionError('Frozen source changed: '+path)
    analysis=read(OUTPUT/'saved_output_v1'/'latency_analysis.json')
    rows=list(csv.DictReader((OUTPUT/'saved_output_v1'/'paired_case_data.csv').open(encoding='utf-8')))
    base={int(r['case']):int(r['subject']) for r in rows if int(r['seed'])==17}
    rng=np.random.default_rng(PROFILE_SELECTION_SEED);cases=[]
    for subject in sorted(set(base.values())):
        pool=np.asarray(sorted(c for c,s in base.items() if s==subject),int)
        cases.extend(rng.choice(pool,PROFILE_CASES_PER_SUBJECT,replace=False).tolist())
    cases=np.asarray(sorted(cases),int);assert len(cases)==36
    configuration=dict(stage='POST_TEST_INSTRUMENTED_COMPONENT_PROFILE_NOT_MAIN_BENCHMARK',primary_delta=PRIMARY_DELTA,
        profile_cases=cases.tolist(),case_subjects=[base[int(c)] for c in cases],selection_seed=PROFILE_SELECTION_SEED,
        selection_uses='Original same900 case IDs and subjects only; four random cases per subject, before reading profile outcomes',
        repetitions=PROFILE_REPETITIONS,seeds=list(SEEDS),frozen_root_lock_sha256=digest(ROOT/'research'/'FROZEN_TEST_LOCK_20261002.json'),
        analysis_sha256=digest(OUTPUT/'saved_output_v1'/'latency_analysis.json'),script_sha256=digest(Path(__file__)),
        separate_dispatch_note='Splitting the original joint compiled stage introduces extra Python/Numba dispatch and clock/bookkeeping costs; profile is descriptive and is never substituted for formal time.',
        exact_fallback_semantics='Frozen compiled exact full prediction, same outputs as the formal pipeline',
        D_free_bytes_before_compute=__import__('shutil').disk_usage('D:\\').free)
    write(out/'prospective_configuration.json',configuration)
    if configuration['D_free_bytes_before_compute']<1024**3:raise RuntimeError('Insufficient D space')
    with np.load(ROOT/'cache'/'har_test_frozen__20261002T060403Z_be087f1d.npz',allow_pickle=False) as z:
        raw=z['raw_test'];subjects=z['subjects']
    assert all(int(subjects[c])==base[int(c)] for c in cases)
    @hd.ck.njit(cache=True)
    def leaf_only(obs,masks,active,start,stop,feature,threshold,missing_left,node_tree,wrong_left,wrong_right,leaf_values,classes,baseline,root_abs):
        ok=hd.lm.update_masks(obs,active,start,stop,feature,threshold,missing_left,node_tree,wrong_left,wrong_right,masks)
        lo,hi,collapsed=hd.lm.mask_intervals(masks,leaf_values,classes,baseline,root_abs)
        return lo,hi,collapsed,ok
    names=('state_setup_ms','feature_ms','leaf_enclosure_ms','probability_interval_ms','output_certificate_ms','control_ms','exact_fallback_ms')
    def execute(pipe,case,partial):
        started=time.perf_counter_ns();cost={name:0. for name in names}
        stamp=time.perf_counter_ns();f=pipe.forest;leaf=pipe.leaves
        obs=np.zeros(pipe.dimension,np.float64);known=np.zeros(pipe.n_blocks,np.bool_)
        masks=leaf.initial.copy() if partial else None
        n=0;fallback=False;collapsed=False;stages=0
        cost['state_setup_ms']+=(time.perf_counter_ns()-stamp)/1e6
        if partial:
            for index,end in enumerate(pipe.ends):
                stamp=time.perf_counter_ns()
                while n<end:pipe.blocks.reveal(case,int(pipe.order[n]),obs,known);n+=1
                cost['feature_ms']+=(time.perf_counter_ns()-stamp)/1e6
                stamp=time.perf_counter_ns()
                lo,hi,collapsed,ok=leaf_only(obs,masks,pipe.active,pipe.offsets[index],pipe.offsets[index+1],f.feature,f.threshold,
                    f.missing_left,leaf.node_tree,leaf.wrong_if_left,leaf.wrong_if_right,leaf.leaf_values,f.classes,f.baseline,f.root_abs)
                cost['leaf_enclosure_ms']+=(time.perf_counter_ns()-stamp)/1e6
                if not ok:raise AssertionError('Empty reachable leaf mask')
                stamp=time.perf_counter_ns();l,u=hd.ck.probability_intervals(lo,hi)
                cost['probability_interval_ms']+=(time.perf_counter_ns()-stamp)/1e6
                stamp=time.perf_counter_ns()
                if collapsed:p=np.ones(len(l),np.float64)/len(l);b=1.
                else:p,b=hd.ck.choose_probability(l,u,0)
                cost['output_certificate_ms']+=(time.perf_counter_ns()-stamp)/1e6
                stamp=time.perf_counter_ns();need_exact=collapsed or n==len(pipe.order);stages+=1
                cost['control_ms']+=(time.perf_counter_ns()-stamp)/1e6
                if need_exact:
                    stamp=time.perf_counter_ns();p=f.exact(obs);b=0.
                    cost['exact_fallback_ms']+=(time.perf_counter_ns()-stamp)/1e6
                    fallback=n==len(pipe.order)
                stamp=time.perf_counter_ns();stop=b<=PRIMARY_DELTA
                cost['control_ms']+=(time.perf_counter_ns()-stamp)/1e6
                if stop:break
        else:
            stamp=time.perf_counter_ns()
            for block in pipe.order:pipe.blocks.reveal(case,int(block),obs,known);n+=1
            cost['feature_ms']+=(time.perf_counter_ns()-stamp)/1e6
            stamp=time.perf_counter_ns();p=f.exact(obs);b=0.;fallback=True
            cost['exact_fallback_ms']+=(time.perf_counter_ns()-stamp)/1e6
        stamp=time.perf_counter_ns()
        result=dict(probability=p,certificate=float(b),blocks=n,fallback=bool(fallback),collapsed=bool(collapsed),stages=stages)
        cost['control_ms']+=(time.perf_counter_ns()-stamp)/1e6
        result['instrumented_complete_ms']=(time.perf_counter_ns()-started)/1e6
        result.update(cost);result['unassigned_instrumentation_residual_ms']=result['instrumented_complete_ms']-sum(cost.values())
        return result
    allrows=[];conformance=[]
    selected=read(ROOT/'results'/'har_model_selection'/'20261002T052208Z_75415f24'/'selected_hgb.json')
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            entry=selected[str(seed)]
            if digest(entry['model_path'])!=entry['model_sha256']:raise AssertionError('Frozen model changed')
            bundle=joblib.load(entry['model_path']);model=bundle['model'];T=float(bundle['temperature'])
            block=hd.har.HARBlocks(raw,bundle['center'],bundle['scale']);forest=hd.calibrated_forest(model,T)
            pipe=cp.CheckpointPipeline(model,forest,block,lock['orders'][str(seed)]['order'],18,T,lock['checkpoints'])
            with np.load(FORMAL/f'seed{seed}'/'all_test_outcomes.npz',allow_pickle=False) as z:
                j=next(i for i,(name,d) in enumerate(zip(z['method_names'],z['deltas'])) if str(name)=='tight_minimax' and float(d)==PRIMARY_DELTA)
                expected_p=z['probabilities'][j,cases];expected_b=z['certificates'][j,cases]
                expected_n=z['block_counts'][j,cases];expected_f=z['fallback'][j,cases];reference=z['reference_probabilities'][cases]
            # First execution warms every separately dispatched kernel outside diagnostic timers used below.
            execute(pipe,int(cases[0]),True);execute(pipe,int(cases[0]),False)
            for i,case in enumerate(cases):
                r=execute(pipe,int(case),True);q=execute(pipe,int(case),False)
                pdiff=float(np.max(np.abs(r['probability']-expected_p[i])));qdiff=float(np.max(np.abs(q['probability']-reference[i])))
                assert pdiff<2e-12 and qdiff<2e-12 and abs(r['certificate']-float(expected_b[i]))<2e-12
                assert r['blocks']==int(expected_n[i]) and r['fallback']==bool(expected_f[i])
                conformance.append(dict(seed=seed,case=int(case),partial_probability_max_difference=pdiff,full_probability_max_difference=qdiff,
                    certificate_difference=float(r['certificate']-expected_b[i]),blocks=r['blocks'],fallback=r['fallback']))
            rr=np.random.default_rng(9312026+seed)
            for case in cases:
                for rep in range(PROFILE_REPETITIONS):
                    for partial in rr.permutation([True,False]):
                        r=execute(pipe,int(case),bool(partial));i=int(np.flatnonzero(cases==case)[0])
                        assert np.max(np.abs(r['probability']-(expected_p[i] if partial else reference[i])))<2e-12
                        r.pop('probability');allrows.append(dict(seed=seed,case=int(case),subject=int(subjects[case]),rep=rep,
                            method='instrumented_tight_primary' if partial else 'instrumented_compiled_full',**r))
            print('Profile completed seed',seed,flush=True)
    save_csv(out/'component_profile_case_results.csv',allrows)
    write(out/'profile_conformance.json',dict(status='PASS_PROFILE_PREDICTION_SEMANTICS',checks=conformance,
        cases_per_system=36,all_three_systems=True,no_new_fits=True,no_policy_change=True))
    aggregates=[]
    for seed in (*SEEDS,None):
        for method in ('instrumented_tight_primary','instrumented_compiled_full'):
            rs=[r for r in allrows if r['method']==method and (seed is None or r['seed']==seed)]
            aggregates.append(dict(seed=seed,method=method,raw_profile_calls=len(rs),distinct_system_cases=len({(r['seed'],r['case']) for r in rs}),
                complete_ms=summary([r['instrumented_complete_ms'] for r in rs]),
                component_mean_ms={name:float(np.mean([r[name] for r in rs])) for name in (*names,'unassigned_instrumentation_residual_ms')},
                mean_blocks=float(np.mean([r['blocks'] for r in rs])),fallback_fraction=float(np.mean([r['fallback'] for r in rs])),
                mean_stages=float(np.mean([r['stages'] for r in rs]))))
    counter=[]
    for _ in range(1000):
        start=time.perf_counter_ns();counter.append((time.perf_counter_ns()-start)/1e6)
    result=dict(stage='POST_TEST_SMALL_SAMPLE_COMPONENT_DIAGNOSTIC',primary_results_replaced=False,configuration=configuration,
        aggregate=aggregates,threadpools=threadpool_info(),counter_empty_span_ms=summary(counter),
        limitations=['Separate Python/Numba dispatches and clock/bookkeeping are charged to instrumented_complete_ms; these times do not reproduce an uninstrumented kernel.',
        'Component spans are nonoverlapping measured intervals; residual is the signed observed full-span minus their sum, never clipped or allocated by assumption.',
        'Small outcome-free sample is descriptive; no causal ablation or hardware/energy claim.',
        'Exact_fallback_ms is full prediction when all used features are known, not the cost of recomputing features.'])
    write(out/'component_profile.json',result)
    write(out/'plot_specifications.json',dict(name='diagnostic_component_stack',data='component_profile_case_results.csv',
        x='seed/method',components=[*names,'unassigned_instrumentation_residual_ms'],
        note='Instrumented post-test sample,36 windows/four per subject; different timer from main formal900. Residual shown separately.'))
    print(json.dumps({'profile':str(out),'aggregate':aggregates},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['analyze','robustness','profile'],required=True)
    args=parser.parse_args()
    if args.phase=='analyze':analyze()
    elif args.phase=='robustness':robustness()
    else:profile()
