"""Development-only, cached-feature order sensitivity without latency claims.

No test-cache access, raw processing, model fit, order selection, or timer in
the outcome policy. Every fixed order and frozen system is reported.
"""
from __future__ import annotations
import os,sys,json,csv,shutil,traceback
from pathlib import Path
for k in ('ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
sys.pycache_prefix=str(ROOT/'temp'/'pycache_review2_order')
for k in ('TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER'):os.environ[k]=str(ROOT/'temp'/'review2_order')
sys.path.insert(0,r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages')
sys.path.insert(0,r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_core')
import har_development_benchmark as hd
from har_checkpoint_development import CheckpointPipeline
import numpy as np
import joblib
from threadpoolctl import threadpool_limits,threadpool_info

OUT=ROOT/'results'/'review2_order_sensitivity'/'development_cached_v1'
SELECTION=ROOT/'results'/'har_model_selection'/'20261002T052208Z_75415f24'/'selected_hgb.json'
ORDER_SOURCE=ROOT/'results'/'har_compute_development'/'corrected_execution'
ORDERS=('locked_gain_per_time','gain_only','reverse_locked','fixed_random')
DELTA=.025
ENDS=(12,16,18)

def save_csv(path,rows):
    with path.open('x',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def cached_policy(pipe,case,reference_score=None,reference_probability=None):
    """Same frozen semantic stages; only selected cached coordinates revealed."""
    f=pipe.forest;leaf=pipe.leaves
    obs=np.zeros(pipe.dimension,np.float64);known=np.zeros(pipe.n_blocks,np.bool_);masks=leaf.initial.copy()
    n=0;fallback=False;first=None;worst_score=-1.;worst_interval=-1.;worst_bound=-1.
    for index,end in enumerate(pipe.ends):
        while n<end:pipe.blocks.reveal(case,int(pipe.order[n]),obs,known);n+=1
        lo,hi,l,u,collapsed,p,b,ok=hd.har_leaf_stage(obs,masks,pipe.active,pipe.offsets[index],pipe.offsets[index+1],
            f.feature,f.threshold,f.missing_left,leaf.node_tree,leaf.wrong_if_left,leaf.wrong_if_right,
            leaf.leaf_values,f.classes,f.baseline,f.root_abs,0)
        if not ok:raise AssertionError('Observed splits eliminated every leaf')
        if first is None:
            widths=np.maximum(0.,u-l)
            first=dict(first_checkpoint_sum_probability_width=float(widths.sum()),
                       first_checkpoint_diameter_upper_proxy=float(np.sqrt(min(2.,float(widths@widths)))),
                       first_checkpoint_certificate_before_exact=float(b))
        if collapsed or n==len(pipe.order):
            p=f.exact(obs);b=0.;fallback=(n==len(pipe.order))
        if reference_score is not None:
            worst_score=max(worst_score,float(np.max(lo-reference_score)),float(np.max(reference_score-hi)))
            worst_interval=max(worst_interval,float(np.max(l-reference_probability)),float(np.max(reference_probability-u)))
            excess=.5*(p@p-reference_probability@reference_probability)-p+reference_probability
            worst_bound=max(worst_bound,float(excess.max()-b))
        if b<=DELTA:break
    return dict(probability=p,certificate=float(b),blocks=n,fallback=bool(fallback),collapsed=bool(collapsed),
                worst_score_violation_signed=worst_score,worst_interval_violation_signed=worst_interval,
                worst_certificate_violation_signed=worst_bound,**first)

def aggregate(rows,p,q,y):
    ell=hd.loss(p,y);reference=hd.loss(q,y)
    every=.5*(np.sum(p*p,axis=1)-np.sum(q*q,axis=1))[:,None]-p+q
    return dict(cases=len(rows),mean_blocks=float(np.mean([r['blocks'] for r in rows])),
        fallback_fraction=float(np.mean([r['fallback'] for r in rows])),
        block_count_counts={str(n):sum(r['blocks']==n for r in rows) for n in ENDS},
        partial_half_brier=float(ell.mean()),full_half_brier=float(reference.mean()),
        mean_realized_reference_loss_excess=float(np.mean(ell-reference)),
        maximum_realized_reference_loss_excess=float(np.max(ell-reference)),
        maximum_every_label_excess=float(every.max()),
        maximum_terminal_certificate=float(max(r['certificate'] for r in rows)),
        maximum_prefix_certificate_violation_signed=float(max(r['worst_certificate_violation_signed'] for r in rows)),
        maximum_prefix_probability_interval_violation_signed=float(max(r['worst_interval_violation_signed'] for r in rows)),
        mean_first_checkpoint_sum_probability_width=float(np.mean([r['first_checkpoint_sum_probability_width'] for r in rows])),
        mean_first_checkpoint_diameter_upper_proxy=float(np.mean([r['first_checkpoint_diameter_upper_proxy'] for r in rows])))

def main():
    if OUT.exists():raise FileExistsError('Existing diagnostic is immutable; no silent repeat')
    free=shutil.disk_usage('D:\\').free
    if free<1024**3:raise RuntimeError('Insufficient D space')
    OUT.mkdir(parents=True)
    source_files=[Path(__file__),Path(hd.__file__),ROOT/'scripts'/'har_checkpoint_development.py',
                  ROOT/'scripts'/'certificate_kernel.py',ROOT/'research'/'leaf_mask_kernel.py']
    cfg=dict(stage='POST_TEST_REVISION_DEVELOPMENT_ONLY_ORDER_SENSITIVITY',test_accessed=False,
        new_fits=0,new_latency_estimates=0,raw_windows_accessed=False,policy_selected=False,
        seeds=list(hd.SEEDS),orders=list(ORDERS),delta=DELTA,checkpoints=list(ENDS),
        random_order_seed='9302026 + frozen system seed',gain_only_tie='Increasing block ID',
        development_cohort='All original four development subjects per system, no subsampling',
        cache_path=str(hd.CACHE),D_free_bytes_before_compute=free,threadpools=threadpool_info(),
        source_sha256={str(p):hd.digest(p) for p in source_files},selection_sha256=hd.digest(SELECTION),
        probability_proxy_note='Outer interval-simplex bound only, not exact diameter or correlation-gap estimate')
    hd.dump(OUT/'prospective_configuration.json',cfg)
    selected=json.loads(SELECTION.read_text(encoding='utf-8'));allrows=[];systemresults=[];subjectrows=[]
    try:
        # Explicit keys ensure raw_train is not decompressed/read.
        with np.load(hd.CACHE,allow_pickle=False) as z:
            features=z['features_train'];labels=z['labels_train'];subjects=z['subjects']
            ids={seed:z[f'development_indices_seed{seed}'] for seed in hd.SEEDS}
            centers={seed:z[f'scaler_center_seed{seed}'] for seed in hd.SEEDS}
            scales={seed:z[f'scaler_scale_seed{seed}'] for seed in hd.SEEDS}
        for seed in hd.SEEDS:
            folder=OUT/f'seed{seed}';entry=selected[str(seed)];path=Path(entry['model_path'])
            if hd.digest(path)!=entry['model_sha256']:raise AssertionError('Frozen model digest mismatch')
            obj=joblib.load(path);model=obj['model'];T=float(obj['temperature']);dev=np.asarray(ids[seed],int)
            center=centers[seed];scale=scales[seed]
            assert np.array_equal(dev,obj['development_indices']) and len(np.unique(subjects[dev]))==4
            assert np.array_equal(center,obj['center']) and np.array_equal(scale,obj['scale'])
            # Preserve the original float32 scaling before conversion to float64.
            x=(features[dev]-center)/scale
            f=hd.calibrated_forest(model,T)
            source=ORDER_SOURCE/f'seed{seed}'/'frozen_order_and_cohort_before_development.json'
            lock=json.loads(source.read_text(encoding='utf-8'));locked=np.asarray(lock['order'],np.int64)
            assert lock['model_sha256']==entry['model_sha256'] and lock['temperature']==T
            orders={ORDERS[0]:locked,ORDERS[1]:np.asarray(sorted(f.used,key=lambda b:(-f.gains[b],int(b))),np.int64),
                    ORDERS[2]:locked[::-1].copy(),ORDERS[3]:np.random.default_rng(9302026+seed).permutation(locked)}
            frozen=dict(seed=seed,model_path=str(path),model_sha256=entry['model_sha256'],temperature=T,
                development_indices=dev.tolist(),development_subjects=np.unique(subjects[dev]).tolist(),
                development_cases=len(dev),orders={k:v.tolist() for k,v in orders.items()},
                first_checkpoint_feature_dimensions={k:int(hd.har.WIDTHS[v[:12]].sum()) for k,v in orders.items()},
                training_gains=f.gains.tolist(),original_order_lock_sha256=hd.digest(source),
                checkpoints=list(ENDS),delta=DELTA,fit_only_scaler_unchanged=True)
            hd.dump(folder/'orders_before_outcomes.json',frozen)
            observer=hd.CachedBlocks(x,hd.har.OFFSETS)
            q=np.stack([f.exact(np.asarray(v,np.float64)) for v in x])
            scores=model._raw_predict(np.asarray(x,np.float64))/T
            native=hd.probabilities(scores);native_diff=float(np.max(np.abs(q-native)));assert native_diff<2e-12
            pp=[];bb=[];counts=[];fallbacks=[];summaries=[];semantic=[]
            for name,order in orders.items():
                pipe=CheckpointPipeline(model,f,observer,order,18,T,ENDS)
                outcomes=[];probs=[]
                for i,case in enumerate(dev):
                    r=cached_policy(pipe,i,scores[i],q[i]);p=r.pop('probability')
                    assert r['worst_score_violation_signed']<2e-11
                    assert r['worst_interval_violation_signed']<2e-12
                    assert r['worst_certificate_violation_signed']<2e-12
                    if i<8:
                        existing=pipe.partial(i,DELTA,0)
                        diff=float(np.max(np.abs(existing['probability']-p)))
                        assert diff<2e-12 and existing['certificate']==r['certificate']
                        assert existing['blocks']==r['blocks'] and existing['fallback']==r['fallback']
                        semantic.append(dict(order=name,case=int(case),maximum_probability_difference=diff))
                    row=dict(seed=seed,order=name,case=int(case),subject=int(subjects[case]),label=int(labels[case]),**r)
                    every=.5*(p@p-q[i]@q[i])-p+q[i]
                    row['realized_reference_loss_excess']=float(every[labels[case]])
                    row['maximum_every_label_excess']=float(every.max())
                    outcomes.append(row);probs.append(p);allrows.append(row)
                p=np.asarray(probs);summaries.append(dict(order=name,**aggregate(outcomes,p,q,labels[dev])))
                for subject in np.unique(subjects[dev]):
                    mask=subjects[dev]==subject;sub=[r for r in outcomes if r['subject']==subject]
                    subjectrows.append(dict(seed=seed,order=name,subject=int(subject),**aggregate(sub,p[mask],q[mask],labels[dev][mask])))
                pp.append(p);bb.append([r['certificate'] for r in outcomes]);counts.append([r['blocks'] for r in outcomes]);fallbacks.append([r['fallback'] for r in outcomes])
                print('ORDER_SENSITIVITY',seed,name,json.dumps(summaries[-1]),flush=True)
            np.savez_compressed(folder/'development_outcomes.npz',method_names=np.asarray(ORDERS),
                development_indices=dev,subjects=subjects[dev],labels=labels[dev],reference_probabilities=q,
                probabilities=np.asarray(pp),certificates=np.asarray(bb),block_counts=np.asarray(counts),fallback=np.asarray(fallbacks))
            result=dict(seed=seed,development_cases=len(dev),subjects=frozen['development_subjects'],temperature=T,
                native_compiled_probability_max_difference=native_diff,summaries=summaries,semantic_checks=semantic)
            hd.dump(folder/'executed_results.json',result);systemresults.append(result)
        save_csv(OUT/'case_outcomes.csv',allrows);hd.dump(OUT/'subject_summaries.json',subjectrows)
        equal=[]
        for name in ORDERS:
            r=[next(v for v in s['summaries'] if v['order']==name) for s in systemresults]
            equal.append(dict(order=name,systems=3,total_system_development_windows=sum(v['cases'] for v in r),
                equal_system_mean_blocks=float(np.mean([v['mean_blocks'] for v in r])),
                equal_system_mean_fallback_fraction=float(np.mean([v['fallback_fraction'] for v in r])),
                equal_system_mean_realized_reference_loss_excess=float(np.mean([v['mean_realized_reference_loss_excess'] for v in r])),
                maximum_every_label_excess=float(max(v['maximum_every_label_excess'] for v in r)),
                maximum_prefix_certificate_violation_signed=float(max(v['maximum_prefix_certificate_violation_signed'] for v in r)),
                equal_system_mean_first_checkpoint_probability_width=float(np.mean([v['mean_first_checkpoint_sum_probability_width'] for v in r])),
                equal_system_mean_first_checkpoint_diameter_upper_proxy=float(np.mean([v['mean_first_checkpoint_diameter_upper_proxy'] for v in r]))))
        hd.dump(OUT/'executed_results.json',dict(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATORY',
            configuration=cfg,systems=systemresults,equal_system_summaries=equal,new_order_selected=False,
            main_policy_or_test_results_changed=False,latency_or_energy_measured=False,
            interpretation='Descriptive dependence of processing count on fixed orders; reused development subjects/models, not new held-out confirmation'))
        hd.dump(OUT/'plot_specifications.json',dict(figures=[
            dict(name='development_block_order_sensitivity',source='executed_results.json',x='order',y=['mean_blocks','fallback_fraction'],facet='seed'),
            dict(name='first_checkpoint_outer_uncertainty',source='executed_results.json',x='order',y=['mean_first_checkpoint_sum_probability_width','mean_first_checkpoint_diameter_upper_proxy'],facet='seed',note='Outer-set proxies; not exact tree correlation gap')]))
        print('ORDER_SENSITIVITY_COMPLETE',json.dumps(equal),flush=True)
    except BaseException as error:
        hd.dump(OUT/'technical_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),retained=True))
        raise

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
