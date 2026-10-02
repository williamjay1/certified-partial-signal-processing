"""Single-factor HAR development tuning of a common certificate schedule.

No test path, fitting, scaling fit, pricing or order selection is implemented.
"""
from __future__ import annotations
import os,sys,json,time,argparse,shutil,traceback
from pathlib import Path
for k in ['ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
sys.pycache_prefix=str(ROOT/'temp'/'pycache_har_checkpoint')
import har_development_benchmark as hd
import numpy as np
import joblib
from threadpoolctl import threadpool_info,threadpool_limits

SCHEDULES={'A':(4,8,12,16,18),'B':(4,12,16,18),'C':(12,16,18)}
PRIMARY_DELTA=.025
SPECIFICATIONS=[(schedule,variant) for schedule in SCHEDULES for variant in (0,1)]+[(None,'native_full'),(None,'compiled_full')]
SOURCE=ROOT/'results'/'har_compute_development'/'corrected_execution'
SELECTION=ROOT/'results'/'har_model_selection'/'20261002T052208Z_75415f24'/'selected_hgb.json'

class CheckpointPipeline(hd.HARPipeline):
    """The shared frozen pipeline with only explicit checkpoint counts changed."""
    def __init__(self,model,forest,blocks,order,n_blocks,temperature,ends):
        super().__init__(model,forest,blocks,order,n_blocks,temperature)
        ends=np.asarray(ends,np.int64)
        if ends.ndim!=1 or len(ends)<1 or np.any(ends<1) or np.any(np.diff(ends)<=0) or ends[-1]!=len(order):
            raise ValueError('Checkpoints must be increasing and end at all model-used blocks')
        self.ends=ends.copy()
        active=[];offsets=[0];previous=0
        for end in self.ends:
            batch=[]
            for block in self.order[previous:end]:
                batch.extend(self.leaves.active[self.leaves.offsets[block]:self.leaves.offsets[block+1]].tolist())
            active.extend(sorted(batch));offsets.append(len(active));previous=end
        self.active=np.asarray(active,np.int64);self.offsets=np.asarray(offsets,np.int64)

def spec_name(schedule,variant):
    return variant if schedule is None else schedule+('_tight_minimax' if variant==0 else '_coordinate_minimax')

def execute(pipelines,case,schedule,variant,trace=False):
    return pipelines['A'].full(case,variant=='compiled_full') if schedule is None else pipelines[schedule].partial(case,PRIMARY_DELTA,variant,trace=trace)

def check(rawpipes,cachepipes,x,q,cases):
    worst_bound=-1.;worst_interval=-1.;max_difference=0.;checks=0
    for case in cases:
        score=rawpipes['A'].model._raw_predict(np.asarray(x[case:case+1],np.float64))[0]/rawpipes['A'].temperature
        for schedule in ('B','C'):
            for variant in (0,1):
                r=execute(rawpipes,int(case),schedule,variant,True);c=execute(cachepipes,int(case),schedule,variant)
                max_difference=max(max_difference,float(np.max(np.abs(r['probability']-c['probability']))))
                assert r['blocks']==c['blocks'] and r['fallback']==c['fallback']
                for stage in r['trace']:
                    checks+=1;p=stage['probability']
                    assert np.all(stage['score_lower']<=score+2e-11) and np.all(score<=stage['score_upper']+2e-11)
                    worst_interval=max(worst_interval,float(np.max(stage['prob_lower']-q[case])),float(np.max(q[case]-stage['prob_upper'])))
                    excess=.5*(np.dot(p,p)-np.dot(q[case],q[case]))-p+q[case]
                    worst_bound=max(worst_bound,float(excess.max()-stage['certificate']))
    assert max_difference<2e-12 and worst_interval<2e-12 and worst_bound<2e-12
    return dict(status='PASS_NUMERICAL_NOT_FORMAL_MACHINE_PROOF',cases=len(cases),prefix_checks=checks,
        max_raw_cache_difference=max_difference,maximum_interval_violation_signed=worst_interval,maximum_every_label_certificate_violation_signed=worst_bound)

def main(record_tag):
    started=time.perf_counter();out=ROOT/'results'/'har_checkpoint_development'/record_tag
    if out.exists():raise FileExistsError('No silent overwrite/rerun of checkpoint experiment')
    if shutil.disk_usage('D:\\').free<1024**3:raise RuntimeError('Insufficient D drive space')
    selected=json.loads(SELECTION.read_text(encoding='utf-8'));assert set(selected)==set(map(str,hd.SEEDS))
    with np.load(hd.CACHE,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    raw=data['raw_train'];features=data['features_train'];labels=data['labels_train'];subjects=data['subjects']
    assert raw.shape==(7352,9,128) and features.shape==(7352,153)
    configuration=dict(phase='CHECKPOINT_ONLY_DEVELOPMENT_TUNING',test_accessed=False,new_fits=0,
        schedules={k:list(v) for k,v in SCHEDULES.items()},primary_delta=PRIMARY_DELTA,
        selection_rule='Equal-seed mean complete time of tight minimax; tie A then B then C; coordinate excluded',
        repetitions=3,cohort='Prior frozen 300 development windows per seed unchanged',
        script_sha256=hd.digest(Path(__file__)),shared_pipeline_sha256=hd.digest(Path(hd.__file__)),
        selection_sha256=hd.digest(SELECTION),train_cache_sha256=hd.digest(hd.CACHE),
        source_development_sha256=hd.digest(SOURCE/'executed_results.json'),threadpools=threadpool_info())
    hd.dump(out/'configuration_before_execution.json',configuration)
    results=[]
    try:
        for seed in hd.SEEDS:
            folder=out/f'seed{seed}';entry=selected[str(seed)];path=Path(entry['model_path'])
            assert hd.digest(path)==entry['model_sha256'];obj=joblib.load(path);model=obj['model'];T=float(obj['temperature'])
            frozen_path=SOURCE/f'seed{seed}'/'frozen_order_and_cohort_before_development.json'
            lock=json.loads(frozen_path.read_text(encoding='utf-8'));order=lock['order'];cohort=np.asarray(lock['timing_cohort_indices'],int)
            fit=data[f'fit_indices_seed{seed}'];dev=data[f'development_indices_seed{seed}']
            center=data[f'scaler_center_seed{seed}'];scale=data[f'scaler_scale_seed{seed}'];x=(features-center)/scale
            assert len(cohort)==300 and np.isin(cohort,dev).all() and len(order)==18
            assert lock['model_sha256']==entry['model_sha256'] and lock['temperature']==T
            assert np.array_equal(center,obj['center']) and np.array_equal(scale,obj['scale'])
            f=hd.calibrated_forest(model,T);assert np.array_equal(np.sort(order),np.sort(f.used))
            blocks=hd.har.HARBlocks(raw,center,scale);cacheblocks=hd.CachedBlocks(x,hd.har.OFFSETS)
            rawpipes={s:CheckpointPipeline(model,f,blocks,order,18,T,ends) for s,ends in SCHEDULES.items()}
            cachepipes={s:CheckpointPipeline(model,f,cacheblocks,order,18,T,ends) for s,ends in SCHEDULES.items()}
            q=np.zeros((len(raw),len(model.classes_)));q[dev]=np.stack([f.exact(np.asarray(x[i],np.float64)) for i in dev])
            native=hd.probabilities(model._raw_predict(np.asarray(x[dev],np.float64))/T)
            native_diff=float(np.max(np.abs(native-q[dev])));assert native_diff<2e-12
            hd.dump(folder/'frozen_inputs_before_comparison.json',dict(seed=seed,order=order,temperature=T,
                model_sha256=entry['model_sha256'],frozen_order_sha256=hd.digest(frozen_path),cohort=cohort.tolist(),
                development_subjects=np.unique(subjects[dev]).tolist(),schedules=configuration['schedules'],
                native_compiled_conformance_max_difference=native_diff,pricing_not_repeated=True))
            for schedule,variant in SPECIFICATIONS:execute(rawpipes,int(cohort[0]),schedule,variant)
            check_result=check(rawpipes,cachepipes,x,q,dev[:24]);hd.dump(folder/'correctness.json',check_result)
            rng=np.random.default_rng(20267002+seed);rows=[]
            for i,case in enumerate(cohort):
                for rep in range(3):
                    for j in rng.permutation(len(SPECIFICATIONS)):
                        schedule,variant=SPECIFICATIONS[j];r=execute(rawpipes,int(case),schedule,variant);p=r.pop('probability')
                        # Labels/reference checks and serialization are outside the complete timer.
                        excess=.5*(np.dot(p,p)-np.dot(q[case],q[case]))-p+q[case]
                        assert float(excess.max()-r['certificate'])<2e-12
                        rows.append(dict(seed=seed,case=int(case),subject=int(subjects[case]),rep=rep,name=spec_name(schedule,variant),
                            schedule=schedule,variant=variant,delta=None if schedule is None else PRIMARY_DELTA,probability=p.tolist(),**r))
                if (i+1)%50==0:print('HAR_CHECKPOINT_TIMED',seed,i+1,'of',len(cohort),flush=True)
            with (folder/'timing_case_results.jsonl').open('x',encoding='utf-8') as stream:
                for r in rows:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            summaries=[];arrays=[];certs=[];counts=[];fallback=[]
            for schedule,variant in SPECIFICATIONS:
                name=spec_name(schedule,variant);timed=[r for r in rows if r['name']==name];first=[r for r in timed if r['rep']==0]
                ids=np.asarray([r['case'] for r in first]);assert np.array_equal(ids,cohort)
                p=np.asarray([r['probability'] for r in first]);b=np.asarray([r['certificate'] for r in first]);ref=q[cohort]
                excess=.5*(np.sum(p*p,1)-np.sum(ref*ref,1))[:,None]-p+ref
                arrays.append(p);certs.append(b);counts.append([r['blocks'] for r in first]);fallback.append([r['fallback'] for r in first])
                summaries.append(dict(name=name,schedule=schedule,variant=variant,cases=300,repetitions=3,
                    complete_ms=hd.timing_summary([r['complete_ms'] for r in timed]),**hd.metrics(p,labels[cohort]),
                    mean_block_count=float(np.mean([r['blocks'] for r in first])),full_fallback_rate=float(np.mean([r['fallback'] for r in first])),
                    mean_reference_loss_excess=float((hd.loss(p,labels[cohort])-hd.loss(ref,labels[cohort])).mean()),
                    maximum_every_label_excess=float(excess.max()),maximum_certificate_violation_signed=float(np.max(excess.max(1)-b))))
            fastest=min([r for r in summaries if r['schedule'] is None],key=lambda r:r['complete_ms']['mean'])
            for r in summaries:r.update(faster_full_comparator=fastest['name'],complete_mean_time_reduction_vs_faster_full=1-r['complete_ms']['mean']/fastest['complete_ms']['mean'])
            np.savez_compressed(folder/'cohort_outcomes.npz',method_names=np.asarray([spec_name(s,v) for s,v in SPECIFICATIONS]),
                development_indices=cohort,subjects=subjects[cohort],labels=labels[cohort],reference_probabilities=q[cohort],
                probabilities=np.asarray(arrays),certificates=np.asarray(certs),block_counts=np.asarray(counts),fallback=np.asarray(fallback))
            result=dict(seed=seed,correctness=check_result,timing_cohort_reference=hd.metrics(q[cohort],labels[cohort]),summaries=summaries)
            hd.dump(folder/'executed_results.json',result);results.append(result)
            print('HAR_CHECKPOINT_SEED_COMPLETE',seed,json.dumps(summaries),flush=True)
        selection_scores={s:float(np.mean([next(r['complete_ms']['mean'] for r in seed['summaries'] if r['name']==s+'_tight_minimax') for seed in results])) for s in SCHEDULES}
        chosen=min(SCHEDULES,key=lambda s:(selection_scores[s],list(SCHEDULES).index(s)))
        final=dict(configuration=configuration,seeds=results,selection_scores_equal_seed_mean_ms=selection_scores,
            selected_schedule=chosen,selected_checkpoints=list(SCHEDULES[chosen]),selection_uses_coordinate=False,
            total_seconds=time.perf_counter()-started,test_accessed=False,
            inferential_scope='Development tuning only; all candidates retained, formal test still unopened.')
        hd.dump(out/'executed_results.json',final)
        hd.dump(out/'selected_schedule.json',dict(stage='DEVELOPMENT_SELECTED_COMMON_CHECKPOINT_SCHEDULE',
            schedule=chosen,checkpoints=list(SCHEDULES[chosen]),primary_delta=PRIMARY_DELTA,
            scores_equal_seed_mean_ms=selection_scores,source_results_sha256=hd.digest(out/'executed_results.json'),
            script_sha256=hd.digest(Path(__file__)),shared_pipeline_sha256=hd.digest(Path(hd.__file__))))
        print('HAR_CHECKPOINT_COMPLETE',json.dumps({k:final[k] for k in ('selected_schedule','selected_checkpoints','selection_scores_equal_seed_mean_ms','total_seconds')}),flush=True)
    except BaseException as error:
        hd.dump(out/'technical_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            preserved_intermediate_outputs=True,elapsed_seconds=time.perf_counter()-started))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record-tag',default='schedule_development_v1')
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.record_tag)
