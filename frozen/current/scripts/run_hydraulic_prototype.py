"""One recorded, development-only deterministic processing prototype run."""
from __future__ import annotations
import os, sys, json, time, shutil, hashlib, argparse
from pathlib import Path
for name in ['ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
SOURCE=Path(r'D:\MLWork\ASC_ComputeValue_20261002')
sys.path.insert(0,str(SOURCE/'scripts'))
import raw_compute as rc
import numpy as np
import joblib
from threadpoolctl import threadpool_limits,threadpool_info
import certificate_kernel as ck

DELTAS=[0.,.01,.025,.05,.1]
VARIANTS={'tight_minimax':0,'coordinate_minimax':1,'tight_midpoint':2}

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def summary(values):
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'median':float(np.median(a)),
            'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99))}

class Pipeline:
    def __init__(self,model,forest,blocks,order,batch_size=1,fallback_backend='native'):
        self.model=model;self.forest=forest;self.blocks=blocks
        self.order=np.asarray(order,int);self.dimension=len(blocks.center)
        self.batch_size=batch_size
        self.fallback_backend=fallback_backend
        if batch_size>1:forest.prepare_batches(self.order,batch_size)

    def full(self,case,compiled=False):
        started=time.perf_counter_ns()
        obs=np.zeros(self.dimension,np.float64);mask=np.zeros(len(rc.WIDTHS),np.bool_)
        for block in self.order:
            self.blocks.reveal(case,int(block),obs,mask)
        if compiled:
            p=self.forest.exact(obs)
        else:
            p=self.model.predict_proba(obs[None,:])[0]
        result={'probability':p,'certificate':0.,'blocks':len(self.order),
                'fallback':True,'collapsed':False,'method':'compiled_full' if compiled else 'native_full'}
        result['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return result

    def partial(self,case,delta,variant=0,trace=False,profile=False):
        if self.batch_size>1:return self.partial_batch(case,delta,variant,trace,profile)
        started=time.perf_counter_ns()
        f=self.forest
        obs=np.zeros(self.dimension,np.float64);mask=np.zeros(len(rc.WIDTHS),np.bool_)
        lower=f.static_lower.copy();upper=f.static_upper.copy()
        stages=[] if trace else None
        times={'feature_ms':0.,'bounds_ms':0.,'certificate_ms':0.,'fallback_prediction_ms':0.}
        n=0;fallback=False;collapsed=False
        while True:
            t=time.perf_counter_ns() if profile else 0
            lo,hi,l,u,collapsed=f.intervals(lower,upper)
            if profile:times['bounds_ms']+=(time.perf_counter_ns()-t)/1e6
            if collapsed:
                t=time.perf_counter_ns() if profile else 0
                # All reachable leaf sums are identical: unknown coordinates do
                # not affect native output, so this is analytically exact B=0.
                p=self.forest.exact(obs) if self.fallback_backend=='compiled' else self.model.predict_proba(obs[None,:])[0];bound=0.
                if profile:times['fallback_prediction_ms']+=(time.perf_counter_ns()-t)/1e6
            else:
                t=time.perf_counter_ns() if profile else 0
                p,bound=ck.choose_probability(l,u,variant)
                if profile:times['certificate_ms']+=(time.perf_counter_ns()-t)/1e6
            if trace:
                stages.append({'n_blocks':n,'score_lower':lo.copy(),'score_upper':hi.copy(),
                               'prob_lower':l.copy(),'prob_upper':u.copy(),
                               'probability':p.copy(),'certificate':float(bound),
                               'collapsed':bool(collapsed)})
            if bound <= delta:
                fallback=(n==len(self.order))
                break
            if n == len(self.order):
                # Guard against numerical padding preventing delta-zero exit.
                t=time.perf_counter_ns() if profile else 0
                p=self.forest.exact(obs) if self.fallback_backend=='compiled' else self.model.predict_proba(obs[None,:])[0];bound=0.;fallback=True
                if profile:times['fallback_prediction_ms']+=(time.perf_counter_ns()-t)/1e6
                break
            block=int(self.order[n]);n+=1
            t=time.perf_counter_ns() if profile else 0
            self.blocks.reveal(case,block,obs,mask)
            if profile:times['feature_ms']+=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns() if profile else 0
            f.update(block,obs,mask,lower,upper)
            if profile:times['bounds_ms']+=(time.perf_counter_ns()-t)/1e6
        result={'probability':p,'certificate':float(bound),'blocks':n,
                'fallback':bool(fallback),'collapsed':bool(collapsed),'method':variant}
        if trace:result['trace']=stages
        if profile:result['profile']=times
        result['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return result

    def partial_batch(self,case,delta,variant=0,trace=False,profile=False):
        started=time.perf_counter_ns()
        f=self.forest
        obs=np.zeros(self.dimension,np.float64);mask=np.zeros(len(rc.WIDTHS),np.bool_)
        lower=f.static_lower.copy();upper=f.static_upper.copy()
        stages=[] if trace else None
        times={'feature_ms':0.,'joint_bound_certificate_ms':0.,'fallback_prediction_ms':0.}
        n=0;fallback=False;collapsed=False
        # No empty-input check: the fixed strategy starts with its first batch.
        # This does not assert that model-specific empty bounds equal the full
        # simplex. Waiting to check is always a valid conservative policy.
        for batch_index,end in enumerate(f.batch_ends):
            t=time.perf_counter_ns() if profile else 0
            while n<end:
                self.blocks.reveal(case,int(self.order[n]),obs,mask);n+=1
            if profile:times['feature_ms']+=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns() if profile else 0
            lo,hi,l,u,collapsed,p,bound=f.batch(batch_index,obs,mask,lower,upper,variant)
            if profile:times['joint_bound_certificate_ms']+=(time.perf_counter_ns()-t)/1e6
            if collapsed or n==len(self.order):
                t=time.perf_counter_ns() if profile else 0
                p=self.forest.exact(obs) if self.fallback_backend=='compiled' else self.model.predict_proba(obs[None,:])[0];bound=0.
                if profile:times['fallback_prediction_ms']+=(time.perf_counter_ns()-t)/1e6
                fallback=(n==len(self.order))
            if trace:stages.append({'n_blocks':n,'score_lower':lo.copy(),'score_upper':hi.copy(),
                'prob_lower':l.copy(),'prob_upper':u.copy(),'probability':p.copy(),
                'certificate':float(bound),'collapsed':bool(collapsed)})
            if bound<=delta:break
        result={'probability':p,'certificate':float(bound),'blocks':n,'fallback':bool(fallback),
                'collapsed':bool(collapsed),'method':variant}
        if trace:result['trace']=stages
        if profile:result['profile']=times
        result['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return result

def label_losses(p):
    return .5*(np.dot(p,p)+1.-2.*p)

def checks(pipe,x_va,q_va,indices):
    max_exact=0.;max_probability_violation=-1.;max_score_violation=-1.
    max_certificate_violation=-1.;tested=0
    # Fully observed, native-comparison and actual online prefix tests.
    for local in indices:
        fullp=pipe.forest.exact(np.asarray(x_va[local],np.float64))
        max_exact=max(max_exact,float(np.max(np.abs(fullp-q_va[local]))))
        result=pipe.partial(907+int(local),0.,0,trace=True)
        native_score=pipe.model._raw_predict(x_va[local:local+1])[0]
        for stage in result['trace']:
            tested+=1
            q=q_va[local]
            max_probability_violation=max(max_probability_violation,
                float(np.max(stage['prob_lower']-q)),float(np.max(q-stage['prob_upper'])))
            max_score_violation=max(max_score_violation,
                float(np.max(stage['score_lower']-native_score)),
                float(np.max(native_score-stage['score_upper'])))
            excess=label_losses(stage['probability'])-label_losses(q)
            max_certificate_violation=max(max_certificate_violation,
                float(np.max(excess)-stage['certificate']))
    assert max_exact<2e-12,(max_exact,'compiled/native mismatch')
    assert max_probability_violation<=2e-12,max_probability_violation
    assert max_score_violation<=2e-11,max_score_violation
    assert max_certificate_violation<=2e-12,max_certificate_violation
    # Additional label-independent random block prefixes, rather than only the
    # fixed-order online prefixes. No new feature/model selection is performed.
    rng=np.random.default_rng(20261002)
    for local in indices[:12]:
        obs=np.zeros(pipe.dimension,np.float64);known=np.zeros(28,np.bool_)
        low=pipe.forest.static_lower.copy();high=pipe.forest.static_upper.copy()
        for block in rng.permutation(pipe.order):
            pipe.blocks.reveal(907+int(local),int(block),obs,known)
            pipe.forest.update(int(block),obs,known,low,high)
            sl,su,l,u,collapsed=pipe.forest.intervals(low,high)
            q=q_va[local]
            assert np.all(l<=q+2e-12) and np.all(q<=u+2e-12)
            p,b=ck.choose_probability(l,u,0)
            assert np.max(label_losses(p)-label_losses(q)) <= b+2e-12
    # End-point algebra, including all-unknown simplex and arbitrary p.
    l=np.zeros(4);u=np.ones(4)
    p,b=ck.choose_probability(l,u,0)
    assert np.max(np.abs(p-.25))<1e-14 and abs(b-.375)<1e-9
    for q in [np.array([1.,0,0,0]),np.array([.1,.2,.3,.4])]:
        p,b=ck.choose_probability(q,q,0)
        assert np.max(np.abs(p-q))<1e-10 and 0.<=b<1e-9
    return {'fully_observed_native_max_probability_difference':max_exact,
            'prefixes_checked':tested,'random_prefix_cases':min(12,len(indices)),
            'maximum_probability_interval_violation_signed':max_probability_violation,
            'maximum_logit_interval_violation_signed':max_score_violation,
            'maximum_every_label_certificate_violation_signed':max_certificate_violation,
            'status':'PASS_NUMERICAL_CHECKS_NOT_FORMAL_FLOATING_POINT_PROOF'}

def timed_cases(pipe,indices,q_va,y_va,repetitions,seed,label):
    specs=[(name,delta) for name in VARIANTS for delta in DELTAS]+[('native_full',None),('compiled_full',None)]
    rng=np.random.default_rng(seed)
    rows=[]
    for number,local in enumerate(indices):
        for rep in range(repetitions):
            for specid in rng.permutation(len(specs)):
                name,delta=specs[specid]
                if name in VARIANTS:
                    result=pipe.partial(907+int(local),delta,VARIANTS[name])
                else:
                    result=pipe.full(907+int(local),name=='compiled_full')
                p=result.pop('probability')
                ref=q_va[local];y=int(y_va[local])
                excess=label_losses(p)-label_losses(ref)
                rows.append({'validation_position':int(local),'rep':rep,'name':name,'delta':delta,
                    'probability':p.tolist(),'half_brier':float(label_losses(p)[y]),
                    'accuracy':int(p.argmax()==y),'reference_loss_excess':float(excess[y]),
                    'max_every_label_excess':float(excess.max()),
                    'certificate_violation':float(excess.max()-result['certificate']),**result})
        if (number+1)%25==0:
            print('TIMED',label,number+1,'of',len(indices),flush=True)
    aggregate=[]
    for name,delta in specs:
        selected=[r for r in rows if r['name']==name and r['delta']==delta]
        unique=[r for r in selected if r['rep']==0]
        aggregate.append({'name':name,'delta':delta,'cases':len(indices),'timing_repetitions':repetitions,
            'complete_ms':summary([r['complete_ms'] for r in selected]),
            'half_brier':float(np.mean([r['half_brier'] for r in unique])),
            'accuracy':float(np.mean([r['accuracy'] for r in unique])),
            'mean_block_count':float(np.mean([r['blocks'] for r in unique])),
            'block_count_distribution':{str(k):sum(r['blocks']==k for r in unique) for k in sorted(set(r['blocks'] for r in unique))},
            'full_fallback_rate':float(np.mean([r['fallback'] for r in unique])),
            'exact_collapse_rate':float(np.mean([r['collapsed'] for r in unique])),
            'mean_reference_loss_excess':float(np.mean([r['reference_loss_excess'] for r in unique])),
            'maximum_every_label_excess':float(max(r['max_every_label_excess'] for r in unique)),
            'maximum_certificate_violation_signed':float(max(r['certificate_violation'] for r in unique))})
    faster=min([r for r in aggregate if r['name'] in ['native_full','compiled_full']],key=lambda r:r['complete_ms']['mean'])
    for r in aggregate:
        r['complete_mean_time_reduction_vs_faster_full']=1.-r['complete_ms']['mean']/faster['complete_ms']['mean']
        r['faster_full_comparator']=faster['name']
    return rows,aggregate

def main(batch_size=1,fallback_backend='native',record_name='compiled_fallback_repair'):
    started=time.perf_counter();free=shutil.disk_usage('D:\\').free
    if free<1024**3:raise RuntimeError('Insufficient D space; no storage substitution')
    out=ROOT/'results' if batch_size==1 else ROOT/'results'/'batch4_repair'
    if fallback_backend=='compiled':out=ROOT/'results'/record_name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'prototype_execution.json').exists():raise RuntimeError('Completed prototype exists; no silent rerun')
    modelpath=SOURCE/'results'/'redesign_diagnostic'/'full_hgb_seed17.joblib'
    model=joblib.load(modelpath)
    forest=ck.FrozenForest(model,rc.GROUP_INDEX)
    costdata=json.loads((SOURCE/'results'/'block_cost_estimates.json').read_text(encoding='utf-8'))
    costs=np.asarray(costdata['milliseconds'],float)
    order=sorted(map(int,forest.used),key=lambda g:(-forest.gains[g]/costs[g],g))
    manifest={'phase':'DEVELOPMENT_PROTOTYPE_NOT_INDEPENDENT_TEST','seed':17,'new_fits':0,
        'allowed_cases':{'predictor_fit':907,'internal_validation':223},'initial_blocks':[],
        'order':order,'model_used_blocks':forest.used.tolist(),'gain_proxy':forest.gains.tolist(),
        'fit_only_median_block_cost_ms':costs.tolist(),'deltas':DELTAS,
        'tree_count':len(forest.roots),'node_count':len(forest.feature),
        'nodes_affected_per_block':np.diff(forest.affected_offsets).tolist(),
        'single_cpu_thread_runtime':threadpool_info(),'D_free_bytes_before_compute':free,
        'model_sha256':rc.digest(modelpath),'kernel_sha256':rc.digest(Path(ck.__file__)),
        'driver_sha256':rc.digest(Path(__file__)),
        'no_gate_partitions_no_old_test_no_HAR':True,'primary_repetitions':3,
        'initial_small_case_count':24,'faster_full_controls':['native sklearn','same-backend compiled exact traversal'],
        'timer_inclusions':'resident raw to output: allocation, operators, scaling, bounds, certificate, control, output',
        'timer_exclusions':['acquisition','transmission','disk','offline fit','JIT','label/reference verification'],
        'float_guard_document':'research/PROTOTYPE_PLAN.md; engineering guards, not formal machine proof'}
    if batch_size>1:
        manifest['documented_development_repair']={
            'previous_results_retained':'results/prototype_execution.json',
            'source_of_change':'v1 overhead decomposition and root-authorized fixed checkpoint repair',
            'batch_size':batch_size,'checkpoints':list(range(batch_size,len(order),batch_size))+[len(order)],
            'changes':['fixed four-block checkpoints','union affected ancestors once per batch','joint compiled bounds/certificate stage','skip empty-input check'],
            'unchanged':['model','scaler','training gain/cost order','five deltas','three certificate variants','native fallback','native and compiled full controls'],
            'not_confirmatory':'All timing results are development diagnostics after an explicit implementation repair.'}
    if fallback_backend=='compiled':
        assert batch_size==4
        manifest['documented_development_repair']={
            'version':'V3_SAME_BACKEND_EXACT_FALLBACK_ONLY',
            'previous_results_retained':['results/prototype_execution.json','results/batch4_repair/prototype_execution.json'],
            'changes':['fallback/collapsed reference prediction uses same compiled exact frozen forest','reference q is the same-backend frozen full pipeline prediction'],
            'unchanged':['model','scaler','training gain/cost order','four-block checkpoints','five deltas','three certificate/output variants','native and compiled full controls'],
            'native_comparator':'Semantic conformance check of the same frozen model, not a different model.'}
        if record_name=='compiled_fallback_repair_corrected':
            manifest['technical_repair']={
                'failed_attempt_retained':'results/compiled_fallback_repair',
                'reason':'Batch4 fallback call site was not switched by the initial edit; incorrect implementation detected from first24 timing before completion.',
                'termination':'Ctrl-C after progress reported50/223; initial configuration/first24/checks preserved; no final223 output exists for that failed attempt.',
                'repair':'Apply frozen fallback_backend to batch4 collapsed/full branch and single-step exhausted-budget branch. No method or tolerance tuning.'}
    dump(out/'prototype_frozen_configuration_before_validation.json',manifest)
    # Only the two authorized partitions are indexed and then used. Other
    # development partitions are not inspected, evaluated or copied to work.
    with np.load(SOURCE/'cache'/'hydraulic_development_raw_v1.npz',allow_pickle=False) as z:
        tr=z['predictor_fit'];va=z['predictor_val'];selected=np.r_[tr,va]
        assert len(tr)==907 and len(va)==223
        x_va=z['X'][va];y_va=z['y'][va]
        raw=[z['raw_'+sensor][selected] for sensor in rc.SENSORS]
        center=z['center'];scale=z['scale'];original_rows=z['original_rows'][va]
        units=z['units'][selected]
        assert not np.intersect1d(units[:907],units[907:]).size
    pipe=Pipeline(model,forest,rc.RawBlocks(raw,center,scale),order,batch_size,fallback_backend)
    if batch_size>1:
        for v in VARIANTS.values():pipe.partial(0,.025,v)
        fit_profiles=[]
        for case in range(8):
            r=pipe.partial(case,.025,0,profile=True);r.pop('probability');r['fitting_position']=case
            fit_profiles.append(r)
        dump(out/'fitting_only_separate_profiles.json',fit_profiles)
    # Native probabilities for correctness/risk only: never enter online policy.
    native_q=model.predict_proba(x_va)
    q_va=np.stack([forest.exact(np.asarray(x,np.float64)) for x in x_va]) if fallback_backend=='compiled' else native_q
    conformance={'reference_backend':fallback_backend,
        'all223_native_vs_reference_max_probability_difference':float(np.max(np.abs(native_q-q_va))),
        'reference_is_same_frozen_model':True}
    dump(out/'reference_semantic_conformance.json',conformance)
    small=np.arange(24)
    # JIT/warm caches excluded equally from later complete per-episode timers.
    for variant in VARIANTS.values():pipe.partial(907,DELTAS[-1],variant)
    pipe.full(907,False);pipe.full(907,True)
    ck.distance_dual_lower(np.zeros(4),np.ones(4));ck.certificate_at(np.ones(4)*.5,np.ones(4)*.25)
    check=checks(pipe,x_va,q_va,small)
    dump(out/'prototype_small_correctness.json',check)
    rows,aggregate=timed_cases(pipe,small,q_va,y_va,1,20261002,'first24')
    dump(out/'prototype_first24_timings.json',{'correctness':check,'all_methods':aggregate})
    print('FIRST24_RESULT',json.dumps(aggregate,ensure_ascii=False),flush=True)
    # Every internal-validation row, all locked variants and tolerances, no
    # algorithm/tolerance/order changes following the first-24 diagnostic.
    rows,aggregate=timed_cases(pipe,np.arange(223),q_va,y_va,3,20261003,'all223')
    with (out/'prototype_case_results.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    # Stronger all-validation every-label check on emitted outputs, outside timer.
    assert max(r['certificate_violation'] for r in rows)<2e-12
    diagnostic_profiles=[]
    for local in small[:12]:
        for name,variant in VARIANTS.items():
            r=pipe.partial(907+int(local),.025,variant,profile=True)
            r.pop('probability');r['name']=name;r['validation_position']=int(local)
            diagnostic_profiles.append(r)
    dump(out/'prototype_separate_profiles.json',diagnostic_profiles)
    result={'phase':'DEVELOPMENT_PROTOTYPE','configuration':manifest,'correctness':check,
        'reference_conformance':conformance,
        'internal_validation_reference_half_brier':float(np.mean([label_losses(q_va[i])[int(y_va[i])] for i in range(223)])),
        'internal_validation_reference_accuracy':float(np.mean(q_va.argmax(1)==y_va)),
        'all_methods':aggregate,'total_execution_seconds':time.perf_counter()-started,
        'validation_original_rows':original_rows.tolist(),
        'component_profiles_are_separate_not_primary':True,
        'development_selection_history':'Hydraulic full HGB originated from an authorized predictor-repair diagnostic after the failed once-gate pilot. This is not a held-out confirmatory test.',
        'numerical_guarantee_limit':'Exact-arithmetic theorem; conservative engineering arithmetic guards and empirical checks, not a formal all-input float64 proof.'}
    dump(out/'prototype_execution.json',result)
    print('COMPLETE_RESULT',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,choices=[1,4],default=1)
    parser.add_argument('--fallback-backend',choices=['native','compiled'],default='native')
    parser.add_argument('--record-name',choices=['compiled_fallback_repair','compiled_fallback_repair_corrected'],default='compiled_fallback_repair')
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.batch_size,args.fallback_backend,args.record_name)
