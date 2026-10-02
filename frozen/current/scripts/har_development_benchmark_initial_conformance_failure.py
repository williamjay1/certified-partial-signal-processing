"""Train-only HAR compute development: generic leaf-mask partial processing.

No test path exists in this script. Frozen full-model choice is external, and
cost/order/timing-cohort locks precede compute development outcomes.
"""
from __future__ import annotations
import os,sys,json,time,shutil,hashlib,argparse
from pathlib import Path
for k in ['ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
sys.path.insert(0,str(ROOT/'research'))
import prepare_har_training as har
import numpy as np
import joblib
from threadpoolctl import threadpool_info,threadpool_limits
import certificate_kernel as ck
import leaf_mask_kernel as lm

CACHE=ROOT/'cache'/'har_train_only__20261002T051217Z_533d2ce1.npz'
SEEDS=(17,29,43)
DELTAS=(0.,.01,.025,.05,.1)
VARIANTS={'tight_minimax':0,'coordinate_minimax':1,'tight_midpoint':2}

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def probabilities(scores):
    a=np.asarray(scores,np.float64)
    v=np.exp(a-a.max(axis=-1,keepdims=True))
    return v/v.sum(axis=-1,keepdims=True)

def loss(p,y):
    p=np.asarray(p);y=np.asarray(y,int)
    return .5*(np.sum(p*p,axis=1)+1.-2.*p[np.arange(len(y)),y])

def metrics(p,y):
    ell=loss(p,y)
    return {'half_brier':float(ell.mean()),'accuracy':float(np.mean(np.argmax(p,axis=1)==y)),
            'class_counts':np.bincount(y,minlength=p.shape[1]).tolist(),
            'class_risk':[float(ell[y==c].mean()) if np.any(y==c) else None for c in range(p.shape[1])]}

def timing_summary(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)))

@ck.njit(cache=True)
def har_leaf_stage(obs,masks,active,start,stop,feature,threshold,missing_left,
                   node_tree,wrong_left,wrong_right,leaf_values,classes,baseline,root_abs,method):
    ok=lm.update_masks(obs,active,start,stop,feature,threshold,missing_left,node_tree,wrong_left,wrong_right,masks)
    lo,hi,collapsed=lm.mask_intervals(masks,leaf_values,classes,baseline,root_abs)
    l,u=ck.probability_intervals(lo,hi)
    if collapsed:p=np.ones(len(l),np.float64)/len(l);b=1.
    else:p,b=ck.choose_probability(l,u,method)
    return lo,hi,l,u,collapsed,p,b,ok

class CachedBlocks:
    """Selected-coordinate observer solely for untimed predictive outcomes."""
    def __init__(self,scaled,offsets):
        self.scaled=scaled;self.offsets=offsets;self.center=np.zeros(scaled.shape[1])
    def reveal(self,case,block,obs,known):
        if known[block]:raise AssertionError('Duplicate selected block')
        a,b=map(int,self.offsets[block:block+2]);obs[a:b]=self.scaled[case,a:b];known[block]=True

class HARPipeline:
    """Dimension/block-agnostic numeric multiclass forest pipeline."""
    def __init__(self,model,forest,blocks,order,n_blocks,temperature=1.,batch_size=4):
        self.model=model;self.forest=forest;self.blocks=blocks;self.order=np.asarray(order,np.int64)
        self.n_blocks=int(n_blocks);self.dimension=len(blocks.center);self.temperature=float(temperature)
        assert self.temperature>0 and len(self.order)>0
        self.ends=np.asarray(list(range(batch_size,len(order),batch_size))+[len(order)],np.int64)
        self.leaves=lm.FrozenLeafMasks(forest)
        active=[];offsets=[0];previous=0
        for end in self.ends:
            batch=[]
            for block in self.order[previous:end]:
                batch.extend(self.leaves.active[self.leaves.offsets[block]:self.leaves.offsets[block+1]].tolist())
            active.extend(sorted(batch));offsets.append(len(active));previous=end
        self.active=np.asarray(active,np.int64);self.offsets=np.asarray(offsets,np.int64)

    def full(self,case,compiled=True):
        started=time.perf_counter_ns()
        obs=np.zeros(self.dimension,np.float64);known=np.zeros(self.n_blocks,np.bool_)
        for block in self.order:self.blocks.reveal(case,int(block),obs,known)
        p=self.forest.exact(obs) if compiled else probabilities(self.model._raw_predict(obs[None,:])/self.temperature)[0]
        r={'probability':p,'certificate':0.,'blocks':len(self.order),'fallback':True,'collapsed':False}
        r['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return r

    def partial(self,case,delta,variant=0,trace=False,profile=False,class_only=False):
        started=time.perf_counter_ns();f=self.forest;leaf=self.leaves
        obs=np.zeros(self.dimension,np.float64);known=np.zeros(self.n_blocks,np.bool_);masks=leaf.initial.copy()
        stages=[] if trace else None
        components={'feature_ms':0.,'bound_certificate_ms':0.,'fallback_ms':0.}
        n=0;fallback=False;collapsed=False
        for index,end in enumerate(self.ends):
            t=time.perf_counter_ns() if profile else 0
            while n<end:self.blocks.reveal(case,int(self.order[n]),obs,known);n+=1
            if profile:components['feature_ms']+=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns() if profile else 0
            lo,hi,l,u,collapsed,p,b,ok=har_leaf_stage(obs,masks,self.active,self.offsets[index],self.offsets[index+1],
                f.feature,f.threshold,f.missing_left,leaf.node_tree,leaf.wrong_if_left,leaf.wrong_if_right,
                leaf.leaf_values,f.classes,f.baseline,f.root_abs,variant)
            if not ok:raise AssertionError('Observed split outcomes eliminated every leaf')
            if profile:components['bound_certificate_ms']+=(time.perf_counter_ns()-t)/1e6
            if collapsed or n==len(self.order):
                t=time.perf_counter_ns() if profile else 0
                p=f.exact(obs);b=0.;fallback=(n==len(self.order))
                if profile:components['fallback_ms']+=(time.perf_counter_ns()-t)/1e6
            if trace:stages.append(dict(n_blocks=n,score_lower=lo.copy(),score_upper=hi.copy(),prob_lower=l.copy(),prob_upper=u.copy(),probability=p.copy(),certificate=float(b)))
            if class_only:
                top=int(np.argmax(l));others=u.copy();others[top]=-np.inf
                if collapsed or n==len(self.order) or l[top]>np.max(others):
                    winner=int(np.argmax(p)) if collapsed or n==len(self.order) else top
                    p=np.eye(len(p))[winner];b=None;break
            elif b<=delta:break
        r={'probability':p,'certificate':None if class_only else float(b),'blocks':n,'fallback':bool(fallback),'collapsed':bool(collapsed)}
        if class_only:r['guarantee']='argmax_of_reference_only_not_probability_loss'
        if trace:r['trace']=stages
        if profile:r['profile']=components
        r['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return r

def calibrated_forest(model,temperature):
    assert np.array_equal(model.classes_,np.arange(len(model.classes_))), 'Labels must match probability-column indices'
    f=ck.FrozenForest(model,har.GROUP_INDEX,len(har.WIDTHS))
    if temperature!=1.:
        f.values/=temperature;f.baseline/=temperature
        f.static_lower/=temperature;f.static_upper/=temperature;f.root_abs/=temperature
    return f

# Shared spelling for controls/test driver; no constructor or behavior change.
LeafPipeline=HARPipeline

def profile_costs(blocks,fit,seed):
    cases=np.sort(np.random.default_rng(41317+seed).choice(fit,min(40,len(fit)),replace=False))
    obs=np.zeros(len(blocks.center),np.float64);known=np.zeros(len(har.WIDTHS),np.bool_)
    for b in range(len(har.WIDTHS)):known[:]=False;blocks.reveal(int(cases[0]),b,obs,known)
    rng=np.random.default_rng(61317+seed);records=[]
    for case in cases:
        for rep in range(3):
            for b in rng.permutation(len(har.WIDTHS)):
                known[:]=False;t=time.perf_counter_ns();blocks.reveal(int(case),int(b),obs,known)
                records.append(dict(case=int(case),rep=rep,block=int(b),milliseconds=(time.perf_counter_ns()-t)/1e6))
    costs=np.asarray([np.median([r['milliseconds'] for r in records if r['block']==b]) for b in range(len(har.WIDTHS))])
    assert np.all(np.isfinite(costs)) and np.all(costs>0)
    return costs,cases,records

def correctness(rawpipe,cachepipe,x,q,cases):
    f=rawpipe.forest;max_interval=-1.;max_excess=-1.;max_raw_cache=0.;checked=0
    for case in cases:
        native_scores=rawpipe.model._raw_predict(np.asarray(x[case:case+1],np.float64))[0]/rawpipe.temperature
        for name,v in VARIANTS.items():
            for delta in DELTAS:
                a=rawpipe.partial(int(case),delta,v,trace=True);c=cachepipe.partial(int(case),delta,v)
                max_raw_cache=max(max_raw_cache,float(np.max(np.abs(a['probability']-c['probability']))))
                assert a['blocks']==c['blocks'] and a['fallback']==c['fallback']
                for stage in a['trace']:
                    checked+=1
                    max_interval=max(max_interval,float(np.max(stage['prob_lower']-q[case])),float(np.max(q[case]-stage['prob_upper'])))
                    assert np.all(stage['score_lower']<=native_scores+2e-11) and np.all(native_scores<=stage['score_upper']+2e-11)
                    p=stage['probability'];excess=.5*(np.dot(p,p)-np.dot(q[case],q[case]))-p+q[case]
                    max_excess=max(max_excess,float(excess.max()-stage['certificate']))
    assert max_interval<2e-12 and max_excess<2e-12 and max_raw_cache<2e-12
    return dict(status='PASS_NUMERICAL_NOT_FORMAL_MACHINE_PROOF',cases=len(cases),prefix_checks=checked,
                max_interval_violation_signed=max_interval,max_every_label_violation_signed=max_excess,
                raw_vs_cache_probability_max_difference=max_raw_cache)

def specifications():return [(n,d) for n in VARIANTS for d in DELTAS]+[('native_full',None),('compiled_full',None)]

def evaluate(pipe,cases,q,labels):
    specs=specifications();p=np.empty((len(specs),len(cases),q.shape[1]),np.float64)
    b=np.empty((len(specs),len(cases)));counts=np.empty_like(b,np.int64);fallback=np.empty_like(b,np.bool_)
    summaries=[]
    for j,(name,delta) in enumerate(specs):
        for i,case in enumerate(cases):
            r=pipe.partial(int(case),delta,VARIANTS[name]) if name in VARIANTS else pipe.full(int(case),name=='compiled_full')
            p[j,i]=r['probability'];b[j,i]=r['certificate'];counts[j,i]=r['blocks'];fallback[j,i]=r['fallback']
        reference=q[cases];excess=.5*(np.sum(p[j]*p[j],1)-np.sum(reference*reference,1))[:,None]-p[j]+reference
        violation=float(np.max(excess.max(1)-b[j]));assert violation<2e-12
        summaries.append(dict(name=name,delta=delta,cases=len(cases),**metrics(p[j],labels[cases]),
            mean_block_count=float(counts[j].mean()),full_fallback_rate=float(fallback[j].mean()),
            mean_reference_loss_excess=float((loss(p[j],labels[cases])-loss(reference,labels[cases])).mean()),
            maximum_every_label_excess=float(excess.max()),maximum_certificate_violation_signed=violation))
    return dict(probabilities=p,certificates=b,block_counts=counts,fallback=fallback),summaries

def timed(pipe,cases,reps,seed):
    rng=np.random.default_rng(seed);specs=specifications();rows=[]
    for i,case in enumerate(cases):
        for rep in range(reps):
            for j in rng.permutation(len(specs)):
                name,delta=specs[j]
                r=pipe.partial(int(case),delta,VARIANTS[name]) if name in VARIANTS else pipe.full(int(case),name=='compiled_full')
                p=r.pop('probability')
                rows.append(dict(case=int(case),rep=rep,name=name,delta=delta,probability=p.tolist(),**r))
        if (i+1)%50==0:print('HAR_TIMED',seed,i+1,'of',len(cases),flush=True)
    summaries=[]
    for name,delta in specs:
        selected=[r for r in rows if r['name']==name and r['delta']==delta]
        summaries.append(dict(name=name,delta=delta,cases=len(cases),repetitions=reps,
            complete_ms=timing_summary([r['complete_ms'] for r in selected]),
            mean_block_count=float(np.mean([r['blocks'] for r in selected])),
            full_fallback_rate=float(np.mean([r['fallback'] for r in selected]))))
    fastest=min([r for r in summaries if r['name'] in ['native_full','compiled_full']],key=lambda r:r['complete_ms']['mean'])
    for r in summaries:
        r['faster_full_comparator']=fastest['name'];r['complete_mean_time_reduction_vs_faster_full']=1.-r['complete_ms']['mean']/fastest['complete_ms']['mean']
    return rows,summaries

def main(selection_path):
    started=time.perf_counter();free=shutil.disk_usage('D:\\').free
    if free<1024**3:raise RuntimeError('Insufficient D space; no substitution')
    out=ROOT/'results'/'har_compute_development';out.mkdir(parents=True,exist_ok=True)
    if (out/'executed_results.json').exists():raise RuntimeError('Completed results exist; no silent rerun')
    selected=json.loads(selection_path.read_text(encoding='utf-8'))
    assert set(selected)==set(map(str,SEEDS)), 'All three selected models must be completed before compute execution'
    for entry in selected.values():assert entry['status']=='development_selected_frozen_hgb'
    with np.load(CACHE,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    raw=data['raw_train'];features=data['features_train'];labels=data['labels_train'];subjects=data['subjects']
    assert raw.shape==(7352,9,128) and features.shape==(7352,153) and len(har.WIDTHS)==18
    overall=dict(phase='OFFICIAL_TRAIN_ONLY_COMPUTE_DEVELOPMENT',test_accessed=False,new_fits=0,
        cache_sha256=digest(CACHE),selection_sha256=digest(selection_path),selection_path=str(selection_path),
        script_sha256=digest(Path(__file__)),leaf_helper_sha256=digest(Path(lm.__file__)),
        certificate_kernel_sha256=digest(Path(ck.__file__)),D_free_bytes=free,threadpools=threadpool_info(),
        seeds=list(SEEDS),deltas=list(DELTAS),singlefactor_boundary='Frozen full models; training-only compute pricing/order; fixedbatch4')
    dump(out/'execution_configuration.json',overall);results=[]
    for seed in SEEDS:
        folder=out/f'seed{seed}';folder.mkdir(exist_ok=True)
        if (folder/'executed_results.json').exists():raise RuntimeError('Seed already completed; no silent rerun')
        entry=selected[str(seed)];checkpoint=Path(entry['model_path'])
        assert digest(checkpoint)==entry['model_sha256']
        obj=joblib.load(checkpoint);model=obj['model'];T=float(obj['temperature'])
        fit=data[f'fit_indices_seed{seed}'];dev=data[f'development_indices_seed{seed}']
        center=data[f'scaler_center_seed{seed}'];scale=data[f'scaler_scale_seed{seed}']
        assert np.array_equal(fit,obj['fit_indices']) and np.array_equal(dev,obj['development_indices'])
        assert np.array_equal(center,obj['center']) and np.array_equal(scale,obj['scale'])
        assert T==float(entry['chosen_temperature']) and T>0
        assert not np.intersect1d(subjects[fit],subjects[dev]).size
        x=(features-center)/scale;blocks=har.HARBlocks(raw,center,scale)
        forest=calibrated_forest(model,T)
        costs,costcases,costrecords=profile_costs(blocks,fit,seed)
        order=sorted(map(int,forest.used),key=lambda g:(-forest.gains[g]/costs[g],g))
        cohort=np.sort(np.random.default_rng(91217+seed).choice(dev,min(300,len(dev)),replace=False))
        lock=dict(seed=seed,temperature=T,model_candidate=entry['candidate'],model_sha256=entry['model_sha256'],
            fit_subjects=np.unique(subjects[fit]).tolist(),development_subjects=np.unique(subjects[dev]).tolist(),
            fit_windows=len(fit),development_windows=len(dev),cost_cases=costcases.tolist(),cost_repetitions=3,
            costs_ms=costs.tolist(),order=order,model_used_blocks=forest.used.tolist(),gain_proxy=forest.gains.tolist(),
            checkpoints=list(range(4,len(order),4))+[len(order)],timing_cohort_indices=cohort.tolist(),
            timing_cohort_subject_counts={str(s):int(np.sum(subjects[cohort]==s)) for s in np.unique(subjects[cohort])},
            deltas=list(DELTAS),primary_repetitions=3,raw_dimension=features.shape[1],raw_block_count=len(har.WIDTHS),
            no_development_repricing_or_order_selection=True,reference_includes_temperature=True)
        dump(folder/'frozen_order_and_cohort_before_development.json',lock);dump(folder/'fitting_cost_records.json',costrecords)
        pipe=HARPipeline(model,forest,blocks,order,len(har.WIDTHS),T)
        cached=HARPipeline(model,forest,CachedBlocks(x,har.OFFSETS),order,len(har.WIDTHS),T)
        # These full references never enter partial policy inputs.
        q=np.zeros((len(raw),len(model.classes_)),np.float64)
        q[dev]=np.stack([forest.exact(np.asarray(x[i],np.float64)) for i in dev])
        native=probabilities(model._raw_predict(np.asarray(x[dev],np.float64))/T)
        diff=float(np.max(np.abs(q[dev]-native)));assert diff<2e-12
        conformance=dict(native_vs_compiled_all_development_max_probability_difference=diff,
            reference_temperature=T,classes=model.classes_.tolist())
        small=dev[:24]
        for variant in VARIANTS.values():pipe.partial(int(small[0]),.025,variant)
        pipe.full(int(small[0]),True);pipe.full(int(small[0]),False)
        check=correctness(pipe,cached,x,q,small)
        dump(folder/'first24_correctness.json',check);dump(folder/'native_reference_conformance.json',conformance)
        smallrows,smalltimes=timed(pipe,small,1,20261002+seed)
        dump(folder/'first24_complete_times.json',smalltimes)
        print('HAR_FIRST24',seed,json.dumps([r for r in smalltimes if r['delta']==.025 or r['name'] in ['native_full','compiled_full']]),flush=True)
        # All-development prediction outcomes use selected cached slices only.
        arrays,devoutcomes=evaluate(cached,dev,q,labels)
        np.savez_compressed(folder/'full_development_outcomes.npz',development_indices=dev,subjects=subjects[dev],
            labels=labels[dev],reference_probabilities=q[dev],method_names=np.asarray([n for n,d in specifications()]),
            deltas=np.asarray([np.nan if d is None else d for n,d in specifications()]),**arrays)
        rows,times=timed(pipe,cohort,3,20261003+seed)
        # Predictive metrics on timed raw outputs, once per case outside timer.
        cohortoutcomes=[]
        for name,delta in specifications():
            selectedrows=[r for r in rows if r['name']==name and r['delta']==delta and r['rep']==0]
            ids=np.asarray([r['case'] for r in selectedrows],int);p=np.asarray([r['probability'] for r in selectedrows])
            ref=q[ids];b=np.asarray([r['certificate'] for r in selectedrows])
            excess=.5*(np.sum(p*p,1)-np.sum(ref*ref,1))[:,None]-p+ref
            violation=float(np.max(excess.max(1)-b));assert violation<2e-12
            cohortoutcomes.append(dict(name=name,delta=delta,cases=len(ids),**metrics(p,labels[ids]),
                mean_reference_loss_excess=float((loss(p,labels[ids])-loss(ref,labels[ids])).mean()),
                maximum_every_label_excess=float(excess.max()),maximum_certificate_violation_signed=violation))
        with (folder/'timing_case_results.jsonl').open('w',encoding='utf-8') as stream:
            for r in rows:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        profiles=[]
        for case in small[:12]:
            r=pipe.partial(int(case),.025,0,profile=True);r.pop('probability');r['case']=int(case);profiles.append(r)
        dump(folder/'separate_component_profiles.json',profiles)
        result=dict(seed=seed,phase='DEVELOPMENT_NOT_INDEPENDENT_TEST',lock=lock,correctness=check,conformance=conformance,
            full_development_reference=metrics(q[dev],labels[dev]),full_development_outcomes=devoutcomes,
            timing_cohort_reference=metrics(q[cohort],labels[cohort]),timing_cohort_outcomes=cohortoutcomes,
            timing_results=times,all_development_outcomes_use_selected_cache_only_not_timing=True,
            actual_raw_operators_in_all_complete_timers=True,test_accessed=False)
        dump(folder/'executed_results.json',result);results.append(result)
        print('HAR_SEED_COMPLETE',seed,json.dumps([r for r in times if r['delta']==.025 or r['name'] in ['native_full','compiled_full']]),flush=True)
    final=dict(configuration=overall,seeds=results,total_seconds=time.perf_counter()-started,
        scope='Three train-only subject-grouped development splits; selected model/calibration outcomes, not independent testing.')
    dump(out/'executed_results.json',final)
    print('HAR_COMPLETE',json.dumps({'total_seconds':final['total_seconds'],'seeds':list(SEEDS)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.selection)
