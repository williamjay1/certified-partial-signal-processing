"""Recorded single-factor candidate-leaf-mask backend development pilot."""
from __future__ import annotations
import os,sys,json,time,shutil
from pathlib import Path
for name in ['ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
SOURCE=Path(r'D:\MLWork\ASC_ComputeValue_20261002')
sys.path.insert(0,str(ROOT/'research'))
import run_hydraulic_prototype as base
import certificate_kernel as ck
import leaf_mask_kernel as lm
import numpy as np
import joblib
from threadpoolctl import threadpool_limits,threadpool_info

@ck.njit(cache=True)
def leaf_batch_stage(observed,masks,active,start,stop,feature,threshold,missing_left,
                     node_tree,wrong_left,wrong_right,leaf_values,classes,baseline,root_abs,method):
    ok=lm.update_masks(observed,active,start,stop,feature,threshold,missing_left,
                       node_tree,wrong_left,wrong_right,masks)
    lo,hi,collapsed=lm.mask_intervals(masks,leaf_values,classes,baseline,root_abs)
    l,u=ck.probability_intervals(lo,hi)
    if collapsed:
        p=np.ones(len(l),np.float64)/len(l);bound=1.
    else:
        p,bound=ck.choose_probability(l,u,method)
    return lo,hi,l,u,collapsed,p,bound,ok

class LeafPipeline(base.Pipeline):
    def __init__(self,model,forest,blocks,order):
        super().__init__(model,forest,blocks,order,4,'compiled')
        self.leaves=lm.FrozenLeafMasks(forest)
        active=[];offsets=[0];previous=0
        for end in forest.batch_ends:
            batch=[]
            for block in self.order[previous:end]:
                batch.extend(self.leaves.active[self.leaves.offsets[block]:self.leaves.offsets[block+1]].tolist())
            active.extend(sorted(batch));offsets.append(len(active));previous=end
        self.batch_active=np.asarray(active,np.int64);self.batch_offsets=np.asarray(offsets,np.int64)

    def partial_batch(self,case,delta,variant=0,trace=False,profile=False):
        started=time.perf_counter_ns();f=self.forest;leaves=self.leaves
        obs=np.zeros(self.dimension,np.float64);known=np.zeros(len(base.rc.WIDTHS),np.bool_)
        candidates=leaves.initial.copy()
        stages=[] if trace else None
        times={'feature_ms':0.,'joint_leafmask_bound_certificate_ms':0.,'fallback_prediction_ms':0.}
        n=0;fallback=False;collapsed=False
        for index,end in enumerate(f.batch_ends):
            t=time.perf_counter_ns() if profile else 0
            while n<end:
                self.blocks.reveal(case,int(self.order[n]),obs,known);n+=1
            if profile:times['feature_ms']+=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns() if profile else 0
            lo,hi,l,u,collapsed,p,bound,ok=leaf_batch_stage(obs,candidates,self.batch_active,
                self.batch_offsets[index],self.batch_offsets[index+1],f.feature,f.threshold,f.missing_left,
                leaves.node_tree,leaves.wrong_if_left,leaves.wrong_if_right,leaves.leaf_values,
                f.classes,f.baseline,f.root_abs,variant)
            if not ok:raise AssertionError('Real observed values eliminated every candidate leaf')
            if profile:times['joint_leafmask_bound_certificate_ms']+=(time.perf_counter_ns()-t)/1e6
            if collapsed or n==len(self.order):
                t=time.perf_counter_ns() if profile else 0
                p=f.exact(obs);bound=0.;fallback=(n==len(self.order))
                if profile:times['fallback_prediction_ms']+=(time.perf_counter_ns()-t)/1e6
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

def write_report(result):
    lines=['# Candidate-leaf-mask backend development pilot','',
        'Date: 2026-10-02. This is a recorded single-factor backend optimization on Hydraulic development data, not an independent test. No classifier was fitted. Old gate and certificate outputs remain retained.', '',
        'The same frozen HGB, predictor-fitting scaler, fixed gain/cost block order, four-block checkpoints, five tolerances, output/certificate variants and compiled fallback are reused. Only reachable-leaf maintenance uses the root-supplied QuickScorer-inspired uint64 candidate mask. Native and compiled full controls share the same actual feature-processing operators. Complete times include all executed software after resident raw-window handoff.', '',
        f"Correctness: {result['correctness']['status']}. Native/compiled reference difference across all223 cases: {result['reference_conformance']['all223_native_vs_reference_max_probability_difference']:.3g}. All emitted outputs pass every-label certificate checks at numerical tolerance2e−12. Exact-arithmetic guarantees and conservative engineering guards do not constitute a formal machine-arithmetic proof.", '',
        '| Method | Delta | Mean ms | Median ms | P95 ms | P99 ms | Mean blocks | Fallback | Half-Brier | Accuracy | Time reduction vs faster full |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['all_methods']:
        d='full' if r['delta'] is None else f"{r['delta']:.3f}"
        lines.append(f"| {r['name']} | {d} | {r['complete_ms']['mean']:.6f} | {r['complete_ms']['median']:.6f} | {r['complete_ms']['p95']:.6f} | {r['complete_ms']['p99']:.6f} | {r['mean_block_count']:.3f} | {r['full_fallback_rate']:.2%} | {r['half_brier']:.6f} | {r['accuracy']:.2%} | {r['complete_mean_time_reduction_vs_faster_full']:.2%} |")
    tight=next(r for r in result['all_methods'] if r['name']=='tight_minimax' and r['delta']==.025)
    coord=next(r for r in result['all_methods'] if r['name']=='coordinate_minimax' and r['delta']==.025)
    full=next(r for r in result['all_methods'] if r['name']==tight['faster_full_comparator'])
    lines += ['',f"At the preselected .025 tolerance, tight minimax takes {tight['complete_ms']['mean']:.6f} ms versus {full['complete_ms']['mean']:.6f} ms for the faster full control ({tight['complete_mean_time_reduction_vs_faster_full']:.2%} time reduction). Coordinate minimax takes {coord['complete_ms']['mean']:.6f} ms and processes {coord['mean_block_count']:.3f} blocks versus {tight['mean_block_count']:.3f} for tight minimax. A large speed difference between runs cannot be attributed to the backend alone; this pilot has its own contemporaneous randomized full controls.", '',
        'The candidate-leaf backend is an implementation optimization. The mathematical tight certificate remains conditional on Q and its claimed contribution must be assessed against the nearly tied coordinate certificate; runtime feasibility alone does not establish independent scientific increment. All five tolerances, negative full-exit results and full fallback are retained.', '',
        f"Execution wall time: {result['total_execution_seconds']:.3f} s. Raw acquisition, transport, disk access, training/JIT compilation and outside-timer reference/label checks are excluded. P95/P99 are empirical software-time diagnostics, not hardware deadlines.", '',
        'Only907 predictor-fit and223 internal-validation rows were indexed for raw processing; no gate partitions, old435outer test or HAR outcomes were used. Separate profiles are diagnostic and do not replace complete timers. Records are under results/leaf_mask_backend_repair; the prospective plan is research/LEAF_MASK_PILOT_PLAN.md.']
    (ROOT/'research'/'LEAF_MASK_PILOT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    started=time.perf_counter();free=shutil.disk_usage('D:\\').free
    if free<1024**3:raise RuntimeError('D free space insufficient, no storage substitution')
    out=ROOT/'results'/'leaf_mask_backend_repair';out.mkdir(parents=True,exist_ok=True)
    if (out/'prototype_execution.json').exists():raise RuntimeError('Completed pilot exists, no silent rerun')
    v3=json.loads((ROOT/'results'/'compiled_fallback_repair_corrected'/'prototype_execution.json').read_text(encoding='utf-8'))
    order=v3['configuration']['order']
    modelpath=SOURCE/'results'/'redesign_diagnostic'/'full_hgb_seed17.joblib'
    model=joblib.load(modelpath);forest=ck.FrozenForest(model,base.rc.GROUP_INDEX)
    configuration={'phase':'SINGLE_FACTOR_DEVELOPMENT_BACKEND_REPAIR','new_fits':0,
        'allowed_cases':{'predictor_fit':907,'internal_validation':223},
        'order':order,'checkpoints':[4,8,12,16,20,24,28],'initial_input':'empty',
        'deltas':base.DELTAS,'variants':base.VARIANTS,'reference':'same compiled exact frozen full HGB',
        'backend':'uint64 reachable-leaf masks; cleared opposite subtrees for newly observed batch blocks',
        'single_factor_changes':['reachable-leaf score-bound maintenance backend only'],
        'single_thread_runtime':threadpool_info(),'D_free_bytes_before_compute':free,
        'tree_count':len(forest.roots),'node_count':len(forest.feature),
        'model_sha256':base.rc.digest(modelpath),'kernel_sha256':base.rc.digest(Path(ck.__file__)),
        'leaf_helper_sha256':base.rc.digest(Path(lm.__file__)),'driver_sha256':base.rc.digest(Path(__file__)),
        'parent_driver_sha256':base.rc.digest(Path(base.__file__)),
        'timing_coordination':'Only executed after HAR loader heavyparse/feature-cache construction completed; root/hydraulic_hooks notified.',
        'old_outputs_read_only':True,'native_and_compiled_full_controls':True,
        'repetitions':3,'no_gate_no_old_test_no_HAR':True}
    base.dump(out/'prototype_frozen_configuration_before_validation.json',configuration)
    with np.load(SOURCE/'cache'/'hydraulic_development_raw_v1.npz',allow_pickle=False) as z:
        tr=z['predictor_fit'];va=z['predictor_val'];indices=np.r_[tr,va]
        assert len(tr)==907 and len(va)==223
        raw=[z['raw_'+s][indices] for s in base.rc.SENSORS]
        blocks=base.rc.RawBlocks(raw,z['center'],z['scale'])
        x_va=z['X'][va];y_va=z['y'][va];original_rows=z['original_rows'][va]
    pipe=LeafPipeline(model,forest,blocks,order)
    q_va=np.stack([forest.exact(np.asarray(x,np.float64)) for x in x_va])
    native_q=model.predict_proba(x_va)
    conformance={'reference_backend':'compiled','all223_native_vs_reference_max_probability_difference':float(np.max(np.abs(q_va-native_q)))}
    assert conformance['all223_native_vs_reference_max_probability_difference']<2e-12
    base.dump(out/'reference_semantic_conformance.json',conformance)
    for v in base.VARIANTS.values():pipe.partial(907,.025,v)
    pipe.full(907,True);pipe.full(907,False)
    small=np.arange(24);check=base.checks(pipe,x_va,q_va,small)
    base.dump(out/'prototype_small_correctness.json',check)
    rows,aggregate=base.timed_cases(pipe,small,q_va,y_va,1,20261002,'leaf_first24')
    base.dump(out/'prototype_first24_timings.json',{'correctness':check,'all_methods':aggregate})
    print('LEAF_FIRST24_RESULT',json.dumps(aggregate,ensure_ascii=False),flush=True)
    rows,aggregate=base.timed_cases(pipe,np.arange(223),q_va,y_va,3,20261003,'leaf_all223')
    with (out/'prototype_case_results.jsonl').open('w',encoding='utf-8') as stream:
        for r in rows:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    assert max(r['certificate_violation'] for r in rows)<2e-12
    profiles=[]
    for local in small[:12]:
        for name,variant in base.VARIANTS.items():
            r=pipe.partial(907+int(local),.025,variant,profile=True);r.pop('probability')
            r['name']=name;r['validation_position']=int(local);profiles.append(r)
    base.dump(out/'prototype_separate_profiles.json',profiles)
    result={'phase':'DEVELOPMENT_BACKEND_REPAIR_NOT_INDEPENDENT_TEST','configuration':configuration,
        'correctness':check,'reference_conformance':conformance,'all_methods':aggregate,
        'total_execution_seconds':time.perf_counter()-started,'validation_original_rows':original_rows.tolist(),
        'component_profiles_are_separate_not_primary':True,
        'root_fit_correctness_record':'research/LEAF_MASK_CORRECTNESS.json',
        'floating_point_limit':'Exact-arithmetic theorem; engineering guards and numerical checks, not formal verified arithmetic.'}
    base.dump(out/'prototype_execution.json',result);write_report(result)
    print('LEAF_COMPLETE_RESULT',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
