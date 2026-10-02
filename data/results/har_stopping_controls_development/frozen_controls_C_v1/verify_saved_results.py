"""Read-only, lightweight verification of this completed development run."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
sys.pycache_prefix=str(ROOT/'temp'/'pycache_har_control_audit')
sys.path[:0]=[r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_core',
             r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages',r'D:\ptenv\Lib\site-packages']
import numpy as np

OUT=Path(__file__).parent
results=json.loads((OUT/'executed_results.json').read_text())
state_path=OUT/'frozen_control_state.json'
state=json.loads(state_path.read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert results['status']=='complete' and results['test_accessed'] is False and results['new_fits']==0
assert sha(state_path)==results['frozen_control_state_sha256']
assert sha(ROOT/'scripts'/'har_stopping_controls.py')==state['stopping_script_sha256']
tree=ast.parse((ROOT/'scripts'/'har_stopping_controls.py').read_text())
assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and
               node.func.attr in ('fit','fit_transform','partial_fit') for node in ast.walk(tree))
method_names=results['configuration']['methods']
audit=[]
for record in results['seeds']:
    seed=record['seed'];folder=OUT/f'seed{seed}'
    with np.load(folder/'all_candidate_outcomes.npz',allow_pickle=False) as z:
        c={k:z[k] for k in z.files}
    with np.load(folder/'all_development_selected_outcomes.npz',allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    with np.load(folder/'timed_raw_cohort_outcomes.npz',allow_pickle=False) as z:
        t={k:z[k] for k in z.files}
    y=d['labels'];q=d['reference_probabilities'];p=d['probabilities'];n=len(y)
    loss=lambda p:.5*(np.sum(p*p,axis=-1)+1.-2.*p[...,np.arange(n),y])
    lossq=.5*(np.sum(q*q,1)+1.-2.*q[np.arange(n),y])
    losses=loss(p);excess=.5*(np.sum(p*p,-1)-np.sum(q*q,-1))[...,None]-p+q
    assert np.max(np.abs(excess-d['every_label_loss_excess']))<2e-15
    assert np.max(np.abs(losses-lossq-d['observed_label_loss_excess']))<2e-15
    assert np.max(np.abs(np.sum(p,-1)-1.))<1e-10 and p.min()>=-1e-15
    assert len(c['setting_ids'])==14 and len(set(c['setting_ids']))==14
    cp=c['probabilities'];ce=.5*(np.sum(cp*cp,-1)-np.sum(q*q,-1))[...,None]-cp+q
    assert np.max(np.abs(ce-c['every_label_loss_excess']))<2e-15
    assert np.array_equal(ce.max(2)>.025+2e-12,c['every_label_delta_violation'])
    candidate_records=json.loads((folder/'all_candidate_summaries.json').read_text())
    for i,candidate in enumerate(candidate_records):
        actual=float((loss(cp[i])-lossq).mean())
        assert abs(candidate['mean_reference_loss_excess']-actual)<2e-15
        assert candidate['eligible']==(actual<=.025)
    for family,key in [('fixed_prefix','fixed_setting_id'),('entropy','entropy_setting_id')]:
        eligible=[r for r in candidate_records if r['family']==family and r['eligible']]
        chosen=min(eligible,key=lambda r:(r['mean_fit_priced_cost_ms'],r['mean_reference_loss_excess'],r['setting_id']))
        assert chosen['setting_id']==state['seeds'][str(seed)][key]
    for j,summary in enumerate(record['all_development_outcomes']):
        assert summary['name']==method_names[j]
        assert abs(summary['half_brier']-float(losses[j].mean()))<2e-15
        assert abs(summary['maximum_every_label_excess']-float(excess[j].max()))<2e-15
    assert np.array_equal(p[0].argmax(1),q.argmax(1))
    assert np.max(excess[3])<=.025+2e-12
    assert np.array_equal(p[1],p[2])
    positions=np.searchsorted(d['development_indices'],t['development_indices'])
    assert np.array_equal(d['development_indices'][positions],t['development_indices'])
    cache_raw_difference=float(np.max(np.abs(p[:,positions]-t['probabilities'])))
    assert cache_raw_difference<2e-12
    rows=[json.loads(line) for line in (folder/'timing_case_results.jsonl').read_text().splitlines()]
    assert len(rows)==6*300*3
    repeat_max=0.
    for j,name in enumerate(method_names):
        timed=[r for r in rows if r['name']==name]
        assert len(timed)==900 and all(r['complete_ms']>0. for r in timed)
        bycase={int(case):t['probabilities'][j,i] for i,case in enumerate(t['development_indices'])}
        for row in timed:
            repeat_max=max(repeat_max,float(np.max(np.abs(bycase[row['case']]-row['probability']))))
        reported=record['timing_summaries'][j]['complete_ms']
        times=np.asarray([r['complete_ms'] for r in timed])
        assert abs(float(times.mean())-reported['mean'])<1e-12
        assert abs(float(np.quantile(times,.95))-reported['p95'])<1e-12
        assert abs(float(np.quantile(times,.99))-reported['p99'])<1e-12
    assert repeat_max<2e-12
    audit.append(dict(seed=seed,development_windows=n,candidate_settings=14,selected_methods=6,
                      raw_timed_method_case_repetitions=len(rows),reference_argmax_mismatches=0,
                      maximum_cache_vs_raw_probability_difference=cache_raw_difference,
                      maximum_repeat_probability_difference=repeat_max,pointwise_main_delta_violations=0))
verification=dict(status='PASS',official_test_accessed=False,new_model_fits=0,
                  state_sha256=sha(state_path),stopping_script_sha256=state['stopping_script_sha256'],
                  findings=audit,scope='Recomputed saved outcomes and timing aggregates; no rerun, fitting, or raw/test data access.')
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
lines=['# HAR stopping controls: development evidence',
       '',f"Completed all three frozen seeds in {results['elapsed_seconds']:.2f} seconds. This is development evidence; the four development subjects per seed were previously used for full-model and temperature selection. No official test was accessed and no new model was fitted.",
       '', 'All seeds select fixed prefix 12 and normalized entropy threshold 1.0; entropy_force_full=false. Their predictions are identical. This degeneration is retained, rather than expanding the candidate grid after observing the results.',
       '', 'Selection used observed-label sample mean relative half-Brier <=0.025 and then minimum original fit-priced block cost. It did not use measured development runtime. All 14 candidates per seed remain available in all_candidate_summaries.json and all_candidate_outcomes.npz.',
       '', '|Seed|Method|All-dev half-Brier|Observed mean excess|Every-label violation fraction|Maximum every-label excess|Mean blocks|',
       '|---:|---|---:|---:|---:|---:|---:|']
short={'score_separation_argmax':'Score-separated argmax','fixed_prefix_mean_completion':'Fixed prefix 12',
       'entropy_mean_completion':'Entropy 1.0','tight_minimax':'Tight minimax .025','native_full':'Native full','compiled_full':'Compiled full'}
for r in results['seeds']:
    for s in r['all_development_outcomes']:
        lines.append(f"|{r['seed']}|{short[s['name']]}|{s['half_brier']:.6f}|{s['mean_reference_loss_excess']:.6f}|{100*s['fraction_cases_every_label_excess_above_delta']:.2f}%|{s['maximum_every_label_excess']:.6f}|{s['mean_block_count']:.3f}|")
lines+=['', 'Every-label violations mean at least one possible label has loss excess >0.025, with 2e-12 numerical tolerance. The mean-completion controls pass the observed mean-risk target but violate the individual target in 16.64–26.78% of development cases. They therefore establish a useful cheap cost/risk comparison, not a pointwise or CRC guarantee.',
        '', 'Score-separated argmax agrees with the full reference on every development window. It does not constrain probability loss; seed 29 observed mean excess is 0.034405, above the primary tolerance. It is an explicit score-interval reference control, not a reproduction of native LazyGBM.',
        '', 'The tight minimax method has zero development every-label violations; this check is numerical evidence under the frozen software arithmetic, rather than a machine-level formal proof.',
        '', '|Seed|Method|Mean ms|P95 ms|P99 ms|Saving vs faster full|',
        '|---:|---|---:|---:|---:|---:|']
for r in results['seeds']:
    for s in r['timing_summaries']:
        ms=s['complete_ms']
        lines.append(f"|{r['seed']}|{short[s['name']]}|{ms['mean']:.6f}|{ms['p95']:.6f}|{ms['p99']:.6f}|{100*s['complete_mean_time_reduction_vs_faster_full']:.2f}%|")
lines+=['', 'Timings use the original 300-case development cohort for each seed, three repetitions, randomized method order, and actual raw statistics/FFT plus fit-only scaling. All six methods were measured contemporaneously; compiled full was faster than native full in every seed. Per-case probabilities, block/fallback flags, every-label and observed-label excesses, and raw timings are preserved in timing_case_results.jsonl.',
        '', 'Fixed/entropy controls save roughly 25–30% on this cohort. Tight minimax saves 10.26%, 0.06%, and 7.06% in seeds 17, 29, and 43. The near-zero seed 29 benefit is retained; no independent test or uncertainty claim is drawn from this development comparison.',
        '', 'Saved outcomes, all candidate gates, deterministic selection, probability validity, every-case argmax preservation, main certificate tolerance, cache/raw selected outputs, repeat outputs, and mean/P95/P99 timings were independently recomputed from saved arrays/rows; verification.json records PASS. No kernels, predictors, or settings were changed after freezing.',
        '', f"Frozen state SHA256: `{sha(state_path)}`. Public script SHA256: `{state['stopping_script_sha256']}`. Root formal-test integration must load this state and use the frozen dispatcher; official test access still requires the immutable root lock."]
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(verification))
