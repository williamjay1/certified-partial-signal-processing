"""Paired complete-time / risk summaries, resampling shared subjects.

Read executed outputs only. No fitting, dataset selection or policy changes.
"""
from pathlib import Path
import sys, json, argparse, hashlib
sys.path.insert(0, r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages')
import numpy as np

SEEDS=(17,29,43)
BOOTSTRAPS=10000
RNG_SEED=7302026

def loss(p,y):
    return .5*(np.sum(p*p,axis=-1)+1.-2.*np.take_along_axis(p,y[...,None],axis=-1)[...,0])

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False),encoding='utf-8')

def interval(a):return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]

def key(name,delta):return str(name),None if delta is None or np.isnan(delta) else float(delta)

def main(root,output):
    if output.exists():raise FileExistsError('Analysis output must be fresh')
    output.mkdir(parents=True)
    records=[];arrays=[];ids=None;shared_cohort=None;subject_ids=None;methods=None
    for seed in SEEDS:
        folder=root/f'seed{seed}'
        z=dict(np.load(folder/'all_test_outcomes.npz',allow_pickle=False))
        for filename in ('auxiliary_uniform_schedule_all_test_outcomes.npz','stopping_controls_all_test_outcomes.npz'):
            extra=dict(np.load(folder/filename,allow_pickle=False))
            for shared in ('subjects','labels','test_indices','reference_probabilities'):
                assert np.array_equal(z[shared],extra[shared]),'Auxiliary case mismatch'
            for field in ('method_names','deltas','probabilities','certificates','block_counts','fallback'):
                z[field]=np.concatenate((z[field],extra[field]),axis=0)
        y=z['labels'].astype(int);subjects=z['subjects'].astype(int)
        p=z['probabilities'];q=z['reference_probabilities'];b=z['certificates']
        assert len(y)==2947 and p.shape[1]==2947 and np.isfinite(p).all()
        keys=[key(n,d) for n,d in zip(z['method_names'],z['deltas'])]
        if methods is None:methods=keys;subject_ids=np.unique(subjects);assert len(subject_ids)==9
        else:assert keys==methods and np.array_equal(subject_ids,np.unique(subjects))
        ell=loss(p,np.broadcast_to(y,p.shape[:-1]));ref=loss(q,y)
        differences=ell-ref[None,:]
        every=.5*(np.sum(p*p,-1)-np.sum(q*q,-1)[None,:])[...,None]-p+q[None,:,:]
        violations=np.max(every,axis=-1)-b
        certified=[j for j,(n,d) in enumerate(keys) if n in ('tight_minimax','coordinate_minimax','tight_midpoint','uniform_A_tight_minimax')]
        assert float(violations[certified].max())<2e-12
        rows=[json.loads(line) for line in (folder/'timing_case_results.jsonl').read_text(encoding='utf-8').splitlines()]
        cohort=np.sort(np.unique([r['case'] for r in rows])).astype(int)
        if shared_cohort is None:shared_cohort=cohort
        else:assert np.array_equal(cohort,shared_cohort)
        positions={int(c):i for i,c in enumerate(cohort)};method_positions={k:i for i,k in enumerate(keys)}
        t=np.full((len(keys),len(cohort),3),np.nan)
        for r in rows:
            k=key(r['name'],r['delta'])
            if k not in method_positions:continue # separately reported additional controls
            j=method_positions[k];i=positions[int(r['case'])];rep=int(r['rep'])
            assert np.isnan(t[j,i,rep]),'Repeated timing row'
            t[j,i,rep]=float(r['complete_ms'])
            assert np.max(np.abs(np.asarray(r['probability'])-p[j,int(r['case'])]))<2e-12
        assert np.isfinite(t).all()
        averaged=t.mean(axis=2)
        full_indices=[keys.index((n,None)) for n in ('native_full','compiled_full')]
        f=min(full_indices,key=lambda j:averaged[j].mean())
        system=[]
        for j,(name,delta) in enumerate(keys):
            vals=t[j].ravel()
            system.append(dict(name=name,delta=delta,complete_mean_ms=float(vals.mean()),
                complete_median_ms=float(np.median(vals)),p95_ms=float(np.quantile(vals,.95)),p99_ms=float(np.quantile(vals,.99)),
                full_test_risk=float(ell[j].mean()),timed_cohort_risk=float(ell[j,cohort].mean()),
                full_test_accuracy=float(np.mean(p[j].argmax(-1)==y)),
                timed_cohort_accuracy=float(np.mean(p[j,cohort].argmax(-1)==y[cohort])),
                full_test_risk_excess=float(differences[j].mean()),timed_cohort_risk_excess=float(differences[j,cohort].mean()),
                maximum_every_label_excess=float(every[j].max()),maximum_certificate_violation_signed=float(violations[j].max()) if np.isfinite(violations[j]).all() else None,
                delta025_every_label_violations=int(np.sum(np.max(every[j],-1)>.025+2e-12)),
                reference_class_disagreements=int(np.sum(p[j].argmax(-1)!=q.argmax(-1))),
                tolerance_violations= int(np.sum(np.max(every[j],-1)>delta+2e-12)) if delta is not None else None,
                full_test_blocks=float(z['block_counts'][j].mean()),timed_cohort_blocks=float(z['block_counts'][j,cohort].mean()),
                full_test_fallback=float(z['fallback'][j].mean()),timed_cohort_fallback=float(z['fallback'][j,cohort].mean()),
                time_reduction=float(1.-vals.mean()/averaged[f].mean()),reference_backend=keys[f][0]))
        full_n=np.asarray([np.sum(subjects==s) for s in subject_ids])
        time_n=np.asarray([np.sum(subjects[cohort]==s) for s in subject_ids])
        sums=dict(time=np.stack([averaged[:,subjects[cohort]==s].sum(1) for s in subject_ids],axis=1),
            timed_loss=np.stack([ell[:,cohort][:,subjects[cohort]==s].sum(1) for s in subject_ids],axis=1),
            timed_difference=np.stack([differences[:,cohort][:,subjects[cohort]==s].sum(1) for s in subject_ids],axis=1),
            full_loss=np.stack([ell[:,subjects==s].sum(1) for s in subject_ids],axis=1),
            full_difference=np.stack([differences[:,subjects==s].sum(1) for s in subject_ids],axis=1),
            full_n=full_n,time_n=time_n,reference=f)
        arrays.append(sums)
        records.append(dict(seed=seed,methods=system,reference_backend=keys[f][0],subject_counts=full_n.tolist(),
                            timed_subject_counts=time_n.tolist()))
    rng=np.random.default_rng(RNG_SEED)
    sampled=rng.integers(0,len(subject_ids),size=(BOOTSTRAPS,len(subject_ids)))
    multiplicity=np.stack([(sampled==s).sum(1) for s in range(len(subject_ids))],axis=1)
    averaged_boot={}
    for metric in ('time','timed_loss','timed_difference','full_loss','full_difference'):
        denom='time_n' if metric in ('time','timed_loss','timed_difference') else 'full_n'
        averaged_boot[metric]=np.mean([a[metric]@multiplicity.T/(a[denom]@multiplicity.T)[None,:] for a in arrays],axis=0)
    full_time_boot=np.mean([(a['time'][a['reference']]@multiplicity.T)/(a['time_n']@multiplicity.T) for a in arrays],axis=0)
    pooled=[]
    for j,(name,delta) in enumerate(methods):
        system=[r['methods'][j] for r in records]
        mr=float(np.mean([r['complete_mean_ms'] for r in system]))
        fr=float(np.mean([r['methods'][a['reference']]['complete_mean_ms'] for r,a in zip(records,arrays)]))
        pooled.append(dict(name=name,delta=delta,complete_mean_ms=mr,time_reduction=1.-mr/fr,
            time_reduction_95ci=interval(1.-averaged_boot['time'][j]/full_time_boot),
            full_test_risk=float(np.mean([r['full_test_risk'] for r in system])),
            full_test_risk_95ci=interval(averaged_boot['full_loss'][j]),
            full_test_risk_excess=float(np.mean([r['full_test_risk_excess'] for r in system])),
            full_test_risk_excess_95ci=interval(averaged_boot['full_difference'][j]),
            timed_cohort_risk=float(np.mean([r['timed_cohort_risk'] for r in system])),
            timed_cohort_risk_excess=float(np.mean([r['timed_cohort_risk_excess'] for r in system])),
            timed_cohort_risk_excess_95ci=interval(averaged_boot['timed_difference'][j]),
            full_test_accuracy=float(np.mean([r['full_test_accuracy'] for r in system])),
            timed_cohort_accuracy=float(np.mean([r['timed_cohort_accuracy'] for r in system])),
            full_test_blocks=float(np.mean([r['full_test_blocks'] for r in system])),
            full_test_fallback=float(np.mean([r['full_test_fallback'] for r in system])),
            maximum_certificate_violation_signed=max((r['maximum_certificate_violation_signed'] for r in system if r['maximum_certificate_violation_signed'] is not None),default=None),
            delta025_every_label_violations=sum(r['delta025_every_label_violations'] for r in system),
            reference_class_disagreements=sum(r['reference_class_disagreements'] for r in system),
            tolerance_violations=sum(r['tolerance_violations'] for r in system) if delta is not None else None))
    subjects=[]
    primary=methods.index(('tight_minimax',.025))
    for i,s in enumerate(subject_ids):
        subjects.append(dict(subject=int(s),full_cases=int(arrays[0]['full_n'][i]),timed_cases=int(arrays[0]['time_n'][i]),
            mean_time_reduction=1.-np.mean([a['time'][primary,i]/a['time_n'][i] for a in arrays])/
                np.mean([a['time'][a['reference'],i]/a['time_n'][i] for a in arrays]),
            full_risk_excess=float(np.mean([a['full_difference'][primary,i]/a['full_n'][i] for a in arrays]))))
    result=dict(stage='ANALYSIS_OF_EXECUTED_LOCKED_TEST',subjects=subject_ids.tolist(),
        windows=2947,timed_windows=len(shared_cohort),seeds=list(SEEDS),bootstrap_repetitions=BOOTSTRAPS,
        bootstrap_seed=RNG_SEED,bootstrap_unit='same nine subjects shared across all three frozen systems',
        systems=records,pooled=pooled,subject_primary=subjects)
    dump(output/'heldout_analysis.json',result)
    np.savez_compressed(output/'bootstrap_arrays.npz',**averaged_boot,full_time=full_time_boot,subject_ids=subject_ids)
    print(json.dumps(next(r for r in pooled if r['name']=='tight_minimax' and r['delta']==.025),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--test-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.test_dir,args.output)
