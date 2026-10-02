"""Separate, fitting-only operator profile; never a complete-time endpoint."""
from __future__ import annotations
import os,sys,time,json
from pathlib import Path
for k in ['ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
SOURCE=Path(r'D:\MLWork\ASC_ComputeValue_20261002')
sys.path.insert(0,str(SOURCE/'scripts'))
import raw_compute as rc
import certificate_kernel as ck
import numpy as np
import joblib
from threadpoolctl import threadpool_limits

def main():
    out=ROOT/'results'/'batch4_repair'/'fitting_only_detailed_components.json'
    if out.exists():raise RuntimeError('Existing completed profile; no silent rerun')
    model=joblib.load(SOURCE/'results'/'redesign_diagnostic'/'full_hgb_seed17.joblib')
    f=ck.FrozenForest(model,rc.GROUP_INDEX)
    manifest=json.loads((ROOT/'results'/'batch4_repair'/'prototype_frozen_configuration_before_validation.json').read_text(encoding='utf-8'))
    order=manifest['order'];f.prepare_batches(np.asarray(order),4)
    with np.load(SOURCE/'cache'/'hydraulic_development_raw_v1.npz',allow_pickle=False) as z:
        idx=z['predictor_fit'][:8]
        raw=[z['raw_'+s][idx] for s in rc.SENSORS]
        blocks=rc.RawBlocks(raw,z['center'],z['scale'])
    rows=[]
    for case in range(8):
        obs=np.zeros(238,np.float64);known=np.zeros(28,np.bool_)
        lo=f.static_lower.copy();hi=f.static_upper.copy()
        for block in order:
            t=time.perf_counter_ns();blocks.reveal(case,block,obs,known);feature=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns();f.update(block,obs,known,lo,hi);update=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns();sl,su,col=ck.score_intervals(f.roots,f.classes,f.baseline,lo,hi,f.root_abs);score=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns();l,u=ck.probability_intervals(sl,su);prob=(time.perf_counter_ns()-t)/1e6
            t=time.perf_counter_ns();p,b=ck.choose_probability(l,u,0);cert=(time.perf_counter_ns()-t)/1e6
            # Native and compiled full-prediction cost diagnostics at the same
            # observed feature vector; outputs not used for policy selection.
            rows.append({'fitting_position':case,'block':block,'feature_ms':feature,
                'affected_node_update_ms':update,'score_interval_ms':score,
                'probability_interval_ms':prob,'tight_certificate_ms':cert})
    # Per-row first kernels may include dispatch/cache-load cost; report both
    # all records and warm rows excluding first fitting case, without replacing
    # or re-running the earlier complete-time result.
    fields=['feature_ms','affected_node_update_ms','score_interval_ms','probability_interval_ms','tight_certificate_ms']
    result={'type':'SEPARATE_INSTRUMENTED_FITTING_ONLY_COMPONENT_PROFILE',
        'cases':8,'no_outcomes_consumed':True,'not_primary_complete_time':True,
        'warm_case_component_mean_ms':{k:float(np.mean([r[k] for r in rows if r['fitting_position']>0])) for k in fields},
        'warm_full28_component_total_mean_ms':{k:float(np.mean([sum(r[k] for r in rows if r['fitting_position']==c) for c in range(1,8)])) for k in fields},
        'raw_records':rows}
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_records'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
