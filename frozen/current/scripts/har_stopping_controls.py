"""Fixed HAR stopping controls; development selection and frozen reuse API.

No official test access, fitting, scaler fitting, temperature or order selection.
Development mean-risk controls do not establish individual loss guarantees.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
for key in ('ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
TEMP=ROOT/'temp'/'har_stopping_controls'
TEMP.mkdir(parents=True,exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER'):os.environ[key]=str(TEMP)
sys.pycache_prefix=str(ROOT/'temp'/'pycache_har_stopping_controls')
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'research'))
import har_checkpoint_development as hc
import har_development_benchmark as hd
import control_design as design
import numpy as np
import joblib
from threadpoolctl import threadpool_limits,threadpool_info

ck=hd.ck; lm=hd.lm
SEEDS=(17,29,43)
DELTA=.025
SCHEDULE=ROOT/'results'/'har_checkpoint_development'/'schedule_development_v1'/'selected_schedule.json'
SOURCE=ROOT/'results'/'har_compute_development'/'corrected_execution'
SELECTION=ROOT/'results'/'har_model_selection'/'20261002T052208Z_75415f24'/'selected_hgb.json'
METHODS=('score_separation_argmax','fixed_prefix_mean_completion','entropy_mean_completion',
         'tight_minimax','native_full','compiled_full')


def utc():return datetime.now(timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


@ck.njit(cache=True)
def entropy_kernel(p):
    value=0.
    for i in range(len(p)):
        if p[i]>0.:value-=p[i]*np.log(p[i])
    return min(1.,max(0.,value/np.log(len(p))))


@ck.njit(cache=True)
def separation_kernel(lo,hi):
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return -1
    for i in range(len(lo)):
        if lo[i]>hi[i]:return -1
    winner=np.argmax(lo)
    for i in range(len(lo)):
        if i!=winner and not lo[winner]>hi[i]:return -1
    return winner


@ck.njit(cache=True)
def argmax_stage(obs,masks,active,start,stop,feature,threshold,missing_left,
                 node_tree,wrong_left,wrong_right,leaf_values,classes,baseline,root_abs):
    ok=lm.update_masks(obs,active,start,stop,feature,threshold,missing_left,
                       node_tree,wrong_left,wrong_right,masks)
    lo,hi,collapsed=lm.mask_intervals(masks,leaf_values,classes,baseline,root_abs)
    p=np.zeros(len(lo),np.float64)
    if not ok or collapsed:return p,False,collapsed,ok,-1
    winner=separation_kernel(lo,hi)
    if winner<0:return p,False,False,ok,winner
    lower,upper=ck.probability_intervals(lo,hi)
    p,unused_certificate=ck.choose_probability(lower,upper,0)
    return p,np.argmax(p)==winner,False,ok,winner


class StoppingPipeline(hc.CheckpointPipeline):
    """Frozen controls with the same raw observer/order/model/shared checkpoints."""
    def argmax(self,case):
        started=time.perf_counter_ns();f=self.forest;leaf=self.leaves
        obs=np.zeros(self.dimension,np.float64);known=np.zeros(self.n_blocks,np.bool_);masks=leaf.initial.copy()
        n=0;fallback=False;collapsed=False;winner=-1
        for index,end in enumerate(self.ends):
            while n<end:self.blocks.reveal(case,int(self.order[n]),obs,known);n+=1
            p,stop,collapsed,ok,winner=argmax_stage(obs,masks,self.active,self.offsets[index],self.offsets[index+1],
                f.feature,f.threshold,f.missing_left,leaf.node_tree,leaf.wrong_if_left,leaf.wrong_if_right,
                leaf.leaf_values,f.classes,f.baseline,f.root_abs)
            if not ok:raise AssertionError('Observed split outcomes eliminated every leaf')
            if collapsed or n==len(self.order):
                p=f.exact(obs);fallback=n==len(self.order);winner=int(np.argmax(p));break
            if stop:break
        r=dict(probability=p,certificate=None,blocks=n,fallback=bool(fallback),collapsed=bool(collapsed),
               guarantee='argmax_of_frozen_reference_only',certified_class=int(winner))
        r['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return r

    def fixed_prefix(self,case,prefix_blocks):
        started=time.perf_counter_ns()
        if not isinstance(prefix_blocks,(int,np.integer)) or int(prefix_blocks) not in self.ends:
            raise ValueError('Fixed prefix must equal a frozen actual checkpoint')
        obs=np.zeros(self.dimension,np.float64);known=np.zeros(self.n_blocks,np.bool_)
        for index in range(int(prefix_blocks)):
            self.blocks.reveal(case,int(self.order[index]),obs,known)
        p=self.forest.exact(obs)
        r=dict(probability=p,certificate=None,blocks=int(prefix_blocks),fallback=int(prefix_blocks)==len(self.order),
               collapsed=False,guarantee='empirical_development_mean_risk_only')
        r['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return r

    def entropy(self,case,threshold):
        started=time.perf_counter_ns()
        if not np.isfinite(threshold) or not 0.<=threshold<=1.:
            raise ValueError('Entropy threshold must be finite and lie in [0,1]')
        obs=np.zeros(self.dimension,np.float64);known=np.zeros(self.n_blocks,np.bool_);n=0;value=0.
        for end in self.ends:
            while n<end:self.blocks.reveal(case,int(self.order[n]),obs,known);n+=1
            p=self.forest.exact(obs)
            value=entropy_kernel(p)
            if n==len(self.order) or value<=threshold:break
        r=dict(probability=p,certificate=None,blocks=n,fallback=n==len(self.order),collapsed=False,
               normalized_entropy=float(value),guarantee='empirical_development_mean_risk_only')
        r['complete_ms']=(time.perf_counter_ns()-started)/1e6
        return r

    def mean_completion_path(self,case):
        """Untimed candidate sweep; each p sees only its processed prefix."""
        obs=np.zeros(self.dimension,np.float64);known=np.zeros(self.n_blocks,np.bool_);n=0;rows=[]
        for end in self.ends:
            while n<end:self.blocks.reveal(case,int(self.order[n]),obs,known);n+=1
            p=self.forest.exact(obs)
            rows.append((p,float(entropy_kernel(p))))
        return rows


def load_frozen_control_state(path,expected_sha256=None):
    """Pure metadata validation for the authorized root formal-test driver."""
    path=Path(path)
    if expected_sha256 is not None and digest(path)!=str(expected_sha256).lower():
        raise ValueError('Frozen stopping control state changed')
    state=json.loads(path.read_text(encoding='utf-8-sig'))
    if state.get('stage')!='DEVELOPMENT_SELECTED_STOPPING_CONTROLS_FROZEN' or set(state.get('seeds',{}))!=set(map(str,SEEDS)):
        raise ValueError('Incomplete or unfrozen stopping-control state')
    if state.get('primary_delta')!=DELTA:
        raise ValueError('Frozen stopping-control primary tolerance changed')
    if digest(SCHEDULE)!=state['selected_schedule_sha256'] or digest(SELECTION)!=state['selected_hgb_sha256']:
        raise ValueError('Frozen schedule/full model selection changed')
    if digest(__file__)!=state['stopping_script_sha256']:
        raise ValueError('Stopping-control script changed since development freeze')
    for module,key in ((hd,'pipeline_sha256'),(hc,'checkpoint_pipeline_sha256'),
                       (design,'control_design_sha256'),(ck,'certificate_kernel_sha256'),(lm,'leaf_kernel_sha256')):
        if digest(module.__file__)!=state[key]:
            raise ValueError('Frozen shared stopping-control dependency changed: '+key)
    return state


def execute_method(pipe,case,name,settings):
    if name=='score_separation_argmax':return pipe.argmax(case)
    if name=='fixed_prefix_mean_completion':return pipe.fixed_prefix(case,int(settings['fixed_prefix_blocks']))
    if name=='entropy_mean_completion':
        # An ineligible entire entropy grid fails closed to exact full completion.
        if settings.get('entropy_force_full',False):return pipe.fixed_prefix(case,len(pipe.order))
        return pipe.entropy(case,float(settings['entropy_threshold']))
    if name=='tight_minimax':return pipe.partial(case,DELTA,0)
    return pipe.full(case,name=='compiled_full')


def all_label_excess(p,q):
    return .5*(np.sum(p*p,axis=-1)-np.sum(q*q,axis=-1))[...,None]-p+q


def outcome_summary(p,q,y,counts,fallback,costs,order,subjects):
    excess=all_label_excess(p,q);worst=excess.max(1)
    observed=hd.loss(p,y)-hd.loss(q,y)
    prefix_cost=np.r_[0.,np.cumsum(np.asarray(costs)[np.asarray(order)])]
    logloss=-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(np.float64).eps,1.))
    summary=dict(**hd.metrics(p,y),log_loss=float(logloss.mean()),
        mean_reference_loss_excess=float(observed.mean()),mean_max_label_excess=float(worst.mean()),
        maximum_every_label_excess=float(worst.max()),fraction_cases_every_label_excess_above_delta=float(np.mean(worst>DELTA+2e-12)),
        fraction_cases_observed_label_excess_above_delta=float(np.mean(observed>DELTA+2e-12)),
        mean_block_count=float(np.mean(counts)),full_fallback_rate=float(np.mean(fallback)),
        mean_fit_priced_cost_ms=float(np.mean(prefix_cost[counts])),violation_numerical_tolerance=2e-12,
        subject_outcomes=[])
    for subject in np.unique(subjects):
        rows=subjects==subject
        summary['subject_outcomes'].append(dict(subject=int(subject),windows=int(rows.sum()),
            half_brier=float(hd.loss(p[rows],y[rows]).mean()),accuracy=float(np.mean(np.argmax(p[rows],1)==y[rows])),
            log_loss=float(logloss[rows].mean()),mean_reference_loss_excess=float(observed[rows].mean()),
            mean_max_label_excess=float(worst[rows].mean()),maximum_every_label_excess=float(worst[rows].max()),
            fraction_cases_every_label_excess_above_delta=float(np.mean(worst[rows]>DELTA+2e-12))))
    return summary,excess,observed


def select_candidates(pipe,dev,q,y,costs,order,subjects,folder):
    prefix=np.asarray(pipe.ends,np.int64);n=len(dev);k=q.shape[1]
    path_probability=np.empty((n,len(prefix),k));path_entropy=np.empty((n,len(prefix)))
    for i,case in enumerate(dev):
        for j,(p,h) in enumerate(pipe.mean_completion_path(int(case))):
            path_probability[i,j]=p;path_entropy[i,j]=h
    candidates=[];outputs=[];block_counts=[]
    for j,end in enumerate(prefix):
        candidates.append(dict(setting_id=f'prefix_{int(end):02d}',family='fixed_prefix',prefix_blocks=int(end)))
        outputs.append(path_probability[:,j]);block_counts.append(np.full(n,end,np.int64))
    for threshold in design.NORMALIZED_ENTROPY_GRID:
        eligible=path_entropy<=threshold;eligible[:,-1]=True
        stage=eligible.argmax(1)
        candidates.append(dict(setting_id=f'entropy_{threshold:.3f}',family='entropy',entropy_threshold=float(threshold)))
        outputs.append(path_probability[np.arange(n),stage]);block_counts.append(prefix[stage])
    expected=[f'prefix_{int(end):02d}' for end in prefix]+[f'entropy_{t:.3f}' for t in design.NORMALIZED_ENTROPY_GRID]
    assert [c['setting_id'] for c in candidates]==expected
    records=[];labelwise=[];observed=[]
    for c,p,counts in zip(candidates,outputs,block_counts):
        scores,excess,difference=outcome_summary(p,q,y,counts,counts==len(order),costs,order,subjects)
        labelwise.append(excess);observed.append(difference)
        records.append(dict(**c,**scores,eligible=scores['mean_reference_loss_excess']<=DELTA,
                            criterion='observed-label development sample mean relative half-Brier <= .025'))
    selected={}
    for family in ('fixed_prefix','entropy'):
        eligible=[r for r in records if r['family']==family and r['eligible']]
        chosen=min(eligible,key=lambda r:(r['mean_fit_priced_cost_ms'],r['mean_reference_loss_excess'],r['setting_id'])) if eligible else None
        if chosen is None:
            # Full completion must be safe; preserve rather than invent a favorable setting.
            chosen=dict(setting_id='full_completion_fallback',family=family,
                        prefix_blocks=len(order),entropy_threshold=0.,entropy_force_full=True,fallback_selection=True)
        selected[family]=chosen
    dump(folder/'all_candidate_summaries.json',records)
    dump(folder/'candidate_selection.json',selected)
    np.savez_compressed(folder/'all_candidate_outcomes.npz',development_indices=dev,subjects=subjects,labels=y,
        reference_probabilities=q,setting_ids=np.asarray(expected),probabilities=np.asarray(outputs),
        block_counts=np.asarray(block_counts),every_label_loss_excess=np.asarray(labelwise),
        observed_label_loss_excess=np.asarray(observed),every_label_delta_violation=np.asarray(labelwise).max(2)>DELTA+2e-12,
        path_checkpoints=prefix,path_probabilities=path_probability,path_entropies=path_entropy)
    return dict(fixed_prefix_blocks=int(selected['fixed_prefix'].get('prefix_blocks',len(order))),
                entropy_threshold=float(selected['entropy'].get('entropy_threshold',0.)),
                entropy_force_full=bool(selected['entropy'].get('entropy_force_full',False)),
                fixed_setting_id=selected['fixed_prefix']['setting_id'],entropy_setting_id=selected['entropy']['setting_id'],
                fixed_selection_summary=selected['fixed_prefix'],entropy_selection_summary=selected['entropy'])


def evaluate_cache(pipe,cases,q,y,costs,order,subjects,settings):
    n=len(cases);p=np.empty((len(METHODS),n,q.shape[1]));counts=np.empty((len(METHODS),n),np.int64)
    fallback=np.empty_like(counts,np.bool_);certificates=np.full(counts.shape,np.nan);summaries=[]
    labelwise=np.empty_like(p);observed=np.empty(counts.shape)
    for j,name in enumerate(METHODS):
        for i,case in enumerate(cases):
            r=execute_method(pipe,int(case),name,settings);p[j,i]=r['probability'];counts[j,i]=r['blocks'];fallback[j,i]=r['fallback']
            if r['certificate'] is not None:certificates[j,i]=r['certificate']
        s,e,d=outcome_summary(p[j],q,y,counts[j],fallback[j],costs,order,subjects)
        labelwise[j]=e;observed[j]=d
        if name=='score_separation_argmax':
            agreement=float(np.mean(np.argmax(p[j],1)==np.argmax(q,1)));assert agreement==1.
            s['reference_argmax_agreement']=agreement
        if name in ('tight_minimax','native_full','compiled_full'):
            violation=float(np.max(e.max(1)-certificates[j]));assert violation<2e-12
            s['maximum_certificate_violation_signed']=violation
        summaries.append(dict(name=name,guarantee='argmax_only' if name=='score_separation_argmax' else
            ('every_case_every_label_relative_half_brier' if name in ('tight_minimax','native_full','compiled_full') else 'development_sample_mean_risk_only'),**s))
    return dict(probabilities=p,block_counts=counts,fallback=fallback,certificates=certificates,
                every_label_loss_excess=labelwise,observed_label_loss_excess=observed),summaries


def timing_pass(rawpipe,cachepipe,cohort,q,y,subjects,costs,order,settings,seed,folder):
    # Warm kernels on a fixed case, then verify cache/raw equality outside timers.
    for name in METHODS:execute_method(rawpipe,int(cohort[0]),name,settings)
    max_difference=0.;checks=0
    for case in cohort[:24]:
        for name in METHODS:
            a=execute_method(rawpipe,int(case),name,settings);b=execute_method(cachepipe,int(case),name,settings)
            difference=float(np.max(np.abs(a['probability']-b['probability'])));max_difference=max(max_difference,difference)
            assert difference<2e-12 and a['blocks']==b['blocks'] and a['fallback']==b['fallback']
            if name=='score_separation_argmax':assert int(np.argmax(a['probability']))==int(np.argmax(q[int(case)]))
            checks+=1
    dump(folder/'raw_cache_first24_correctness.json',dict(status='PASS_NUMERICAL_NOT_FORMAL_MACHINE_PROOF',
         method_case_checks=checks,max_probability_difference=max_difference))
    rows=[];rng=np.random.default_rng(20269002+seed)
    for i,case in enumerate(cohort):
        for rep in range(3):
            for j in rng.permutation(len(METHODS)):
                name=METHODS[j];r=execute_method(rawpipe,int(case),name,settings);p=r.pop('probability')
                excess=.5*(np.dot(p,p)-np.dot(q[case],q[case]))-p+q[case]
                if name=='score_separation_argmax':assert np.argmax(p)==np.argmax(q[case])
                if r['certificate'] is not None:assert float(excess.max()-r['certificate'])<2e-12
                rows.append(dict(seed=seed,case=int(case),subject=int(subjects[case]),rep=rep,name=name,
                    probability=p.tolist(),every_label_loss_excess=excess.tolist(),
                    observed_label_loss_excess=float(excess[y[case]]),**r))
        if (i+1)%50==0:print('HAR_STOPPING_TIMED',seed,i+1,'of',len(cohort),flush=True)
    with (folder/'timing_case_results.jsonl').open('x',encoding='utf-8') as stream:
        for r in rows:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    summaries=[];outputs=[];counts_all=[];fallback_all=[];excess_all=[]
    for name in METHODS:
        timed=[r for r in rows if r['name']==name];first=[r for r in timed if r['rep']==0]
        ids=np.asarray([r['case'] for r in first],int);assert np.array_equal(ids,cohort)
        p=np.asarray([r['probability'] for r in first]);counts=np.asarray([r['blocks'] for r in first],int)
        fallback=np.asarray([r['fallback'] for r in first],bool)
        s,e,d=outcome_summary(p,q[cohort],y[cohort],counts,fallback,costs,order,subjects[cohort])
        if name=='score_separation_argmax':s['reference_argmax_agreement']=float(np.mean(p.argmax(1)==q[cohort].argmax(1)))
        summaries.append(dict(name=name,cases=300,repetitions=3,complete_ms=hd.timing_summary([r['complete_ms'] for r in timed]),**s))
        outputs.append(p);counts_all.append(counts);fallback_all.append(fallback);excess_all.append(e)
    best=min([r for r in summaries if r['name'] in ('native_full','compiled_full')],key=lambda r:r['complete_ms']['mean'])
    for row in summaries:row.update(faster_full_comparator=best['name'],complete_mean_time_reduction_vs_faster_full=
                                   1.-row['complete_ms']['mean']/best['complete_ms']['mean'])
    np.savez_compressed(folder/'timed_raw_cohort_outcomes.npz',development_indices=cohort,subjects=subjects[cohort],
        labels=y[cohort],reference_probabilities=q[cohort],method_names=np.asarray(METHODS),probabilities=np.asarray(outputs),
        block_counts=np.asarray(counts_all),fallback=np.asarray(fallback_all),every_label_loss_excess=np.asarray(excess_all))
    dump(folder/'timing_summaries.json',summaries)
    return summaries


def main(record_tag=None):
    if shutil.disk_usage('D:\\').free<2_000_000_000:raise RuntimeError('Insufficient D space; no output relocation')
    schedule=json.loads(SCHEDULE.read_text());ends=schedule['checkpoints']
    if schedule['schedule']!='C' or ends!=[12,16,18]:raise ValueError('This prospective control plan locked common schedule C=[12,16,18]')
    selected=json.loads(SELECTION.read_text());assert set(selected)==set(map(str,SEEDS))
    tag=record_tag or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'results'/'har_stopping_controls_development'/tag
    if out.exists():raise FileExistsError('Use a fresh explicitly recorded output directory; no overwrite/rerun')
    out.mkdir(parents=True,exist_ok=False)
    configuration=dict(created_utc=utc(),phase='TRAIN_ONLY_STOPPING_CONTROL_DEVELOPMENT',test_accessed=False,new_fits=0,
        plan_sha256=digest(ROOT/'research'/'HAR_STOPPING_CONTROL_PLAN.md'),script_sha256=digest(__file__),
        selected_hgb_sha256=digest(SELECTION),selected_schedule_sha256=digest(SCHEDULE),
        pipeline_sha256=digest(hd.__file__),checkpoint_pipeline_sha256=digest(hc.__file__),
        control_design_sha256=digest(design.__file__),certificate_kernel_sha256=digest(ck.__file__),leaf_kernel_sha256=digest(lm.__file__),
        train_cache_sha256=digest(hd.CACHE),seeds=list(SEEDS),checkpoints=ends,prefix_candidates=ends,
        entropy_candidates=list(design.NORMALIZED_ENTROPY_GRID),primary_delta=DELTA,methods=list(METHODS),
        control_selection='observed-label dev sample mean loss excess <= .025, minimum mean sum original fit block costs; tie risk then setting ID',
        development_labels_used_for_selection=True,test_labels_used_for_selection=False,
        runtime_used_for_control_selection=False,threadpools=threadpool_info(),d_free_gib=shutil.disk_usage('D:\\').free/2**30)
    dump(out/'configuration_before_evaluation.json',configuration)
    started=time.perf_counter();state=dict(stage='DEVELOPMENT_SELECTED_STOPPING_CONTROLS_FROZEN',
        created_utc=utc(),primary_delta=DELTA,checkpoints=ends,seeds={},selected_hgb_sha256=digest(SELECTION),
        selected_schedule_sha256=digest(SCHEDULE),stopping_script_sha256=digest(__file__),
        pipeline_sha256=digest(hd.__file__),checkpoint_pipeline_sha256=digest(hc.__file__),
        control_design_sha256=digest(design.__file__),certificate_kernel_sha256=digest(ck.__file__),
        leaf_kernel_sha256=digest(lm.__file__),selection_uses_observed_development_labels=True,
        individual_risk_guarantee=False,independent_calibration_set=False,test_accessed=False)
    try:
        with np.load(hd.CACHE,allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        raw=data['raw_train'];features=data['features_train'];y=data['labels_train'];subjects=data['subjects']
        assert raw.shape==(7352,9,128) and features.shape==(7352,153)
        prepared={}
        # Freeze every seed's controls before any selected-control runtime is observed.
        for seed in SEEDS:
            folder=out/f'seed{seed}';folder.mkdir(exist_ok=False)
            entry=selected[str(seed)];assert digest(entry['model_path'])==entry['model_sha256']
            obj=joblib.load(entry['model_path']);model=obj['model'];temperature=float(obj['temperature'])
            lock_path=SOURCE/f'seed{seed}'/'frozen_order_and_cohort_before_development.json'
            lock=json.loads(lock_path.read_text());order=lock['order'];costs=np.asarray(lock['costs_ms'])
            cohort=np.asarray(lock['timing_cohort_indices'],int);dev=data[f'development_indices_seed{seed}']
            fit=data[f'fit_indices_seed{seed}'];center=data[f'scaler_center_seed{seed}'];scale=data[f'scaler_scale_seed{seed}']
            assert lock['model_sha256']==entry['model_sha256'] and lock['temperature']==temperature and temperature==entry['chosen_temperature']
            assert len(cohort)==300 and np.isin(cohort,dev).all() and not np.intersect1d(subjects[fit],subjects[dev]).size
            assert np.array_equal(dev,obj['development_indices']) and np.array_equal(fit,obj['fit_indices'])
            assert np.array_equal(center,obj['center']) and np.array_equal(scale,obj['scale'])
            x=(features-center)/scale;forest=hd.calibrated_forest(model,temperature)
            cached=StoppingPipeline(model,forest,hd.CachedBlocks(x,hd.har.OFFSETS),order,18,temperature,ends)
            rawpipe=StoppingPipeline(model,forest,hd.har.HARBlocks(raw,center,scale),order,18,temperature,ends)
            q=np.zeros((len(raw),6));q[dev]=np.stack([forest.exact(np.asarray(x[i],np.float64)) for i in dev])
            native=hd.probabilities(model._raw_predict(np.asarray(x[dev],np.float64))/temperature)
            assert np.max(np.abs(native-q[dev]))<2e-12
            settings=select_candidates(cached,dev,q[dev],y[dev],costs,order,subjects[dev],folder)
            state['seeds'][str(seed)]=dict(seed=seed,**settings,model_path=entry['model_path'],model_sha256=entry['model_sha256'],
                temperature=temperature,order=order,checkpoints=ends,costs_ms=costs.tolist(),
                original_order_cohort_lock_sha256=digest(lock_path),fit_subject_ids=np.unique(subjects[fit]).tolist(),
                development_subject_ids=np.unique(subjects[dev]).tolist(),development_windows=len(dev),
                scaler_center_sha256=hashlib.sha256(center.tobytes()).hexdigest(),scaler_scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest())
            prepared[seed]=dict(folder=folder,cached=cached,rawpipe=rawpipe,q=q,dev=dev,cohort=cohort,order=order,costs=costs,settings=settings)
            print('HAR_STOPPING_SELECTED',seed,json.dumps({k:settings[k] for k in ('fixed_prefix_blocks','entropy_threshold','fixed_setting_id','entropy_setting_id')}),flush=True)
        state['frozen_before_runtime_utc']=utc();dump(out/'frozen_control_state.json',state)
        frozen_sha=digest(out/'frozen_control_state.json');results=[]
        for seed in SEEDS:
            p=prepared[seed];dev=p['dev'];folder=p['folder']
            arrays,summaries=evaluate_cache(p['cached'],dev,p['q'][dev],y[dev],p['costs'],p['order'],subjects[dev],p['settings'])
            np.savez_compressed(folder/'all_development_selected_outcomes.npz',development_indices=dev,subjects=subjects[dev],labels=y[dev],
                reference_probabilities=p['q'][dev],method_names=np.asarray(METHODS),**arrays)
            dump(folder/'all_development_selected_summaries.json',summaries)
            times=timing_pass(p['rawpipe'],p['cached'],p['cohort'],p['q'],y,subjects,p['costs'],p['order'],p['settings'],seed,folder)
            result=dict(seed=seed,settings=p['settings'],phase='DEVELOPMENT_NOT_INDEPENDENT_TEST',
                all_development_reference=hd.metrics(p['q'][dev],y[dev]),all_development_outcomes=summaries,
                timing_cohort_reference=hd.metrics(p['q'][p['cohort']],y[p['cohort']]),timing_summaries=times,test_accessed=False)
            dump(folder/'executed_results.json',result);results.append(result)
            print('HAR_STOPPING_SEED_COMPLETE',seed,json.dumps(times),flush=True)
        assert digest(out/'frozen_control_state.json')==frozen_sha
        final=dict(configuration=configuration,created_utc=utc(),status='complete',seeds=results,
                   frozen_control_state=str(out/'frozen_control_state.json'),frozen_control_state_sha256=frozen_sha,
                   elapsed_seconds=time.perf_counter()-started,test_accessed=False,new_fits=0,
                   scope='Development sample mean risk selection; controls have no pointwise half-Brier guarantee, class control has argmax-only guarantee.')
        dump(out/'executed_results.json',final)
        print('HAR_STOPPING_COMPLETE',json.dumps({k:final[k] for k in ('status','frozen_control_state','frozen_control_state_sha256','elapsed_seconds')}),flush=True)
    except BaseException as error:
        dump(out/'technical_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
             preserved_intermediate_outputs=True,elapsed_seconds=time.perf_counter()-started,test_accessed=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',action='store_true',required=True,
                        help='Explicitly run authorized train-only control development; no test mode exists')
    parser.add_argument('--record-tag',default=None)
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.record_tag)
