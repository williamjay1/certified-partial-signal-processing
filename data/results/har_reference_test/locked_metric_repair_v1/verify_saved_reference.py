"""Lightweight saved-array verification; no fitting or predictor execution."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
sys.pycache_prefix=str(ROOT/'temp'/'pycache_har_reference_audit')
sys.path[:0]=[r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_core',
             r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages',r'D:\ptenv\Lib\site-packages']
import numpy as np

OUT=Path(__file__).parent
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
run=json.loads((OUT/'evaluation_manifest.json').read_text())
metrics=json.loads((OUT/'full_model_metrics.json').read_text())
old=json.loads((OUT.parent/'locked_execution_v1'/'full_model_metrics.json').read_text())
lock=json.loads((ROOT/'research'/'FROZEN_TEST_LOCK_20261002.json').read_text())
assert run['status']=='complete' and run['successful_models']==15 and run['failed_models']==0
assert len(metrics['models'])==15 and not metrics['failures']
assert len(old['models'])==12 and len(old['failures'])==3
assert run['root_lock_sha256']==sha(ROOT/'research'/'FROZEN_TEST_LOCK_20261002.json')
assert lock['reference_script_sha256']==sha(ROOT/'scripts'/'har_reference_test.py')
assert run['fitting_operations']==0 and run['test_model_selection'] is False
assert run['raw_test_array_loaded'] is False and run['bootstrap_executed'] is False
original=ast.parse((ROOT/'scripts'/'har_reference_test.py').read_text())
repaired=ast.parse((ROOT/'scripts'/'har_reference_test_metric_repair.py').read_text())
def principal_nodes(tree):
    nodes=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign):continue
        names={sub.id for target in node.targets for sub in ast.walk(target) if isinstance(sub,ast.Name)}
        if names.intersection({'p','logits','xs','risk','correct','probabilities','target'}):nodes.append(ast.dump(node))
    return nodes
assert principal_nodes(original)==principal_nodes(repaired)
assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and
               node.func.attr in ('fit','fit_transform','partial_fit') for node in ast.walk(repaired))
with np.load(OUT/'reference_outcomes.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
with np.load(OUT/'shared_subject_bootstrap_inputs.npz',allow_pickle=False) as z:bs={k:z[k] for k in z.files}
p=data['probabilities'];y=data['labels'];subjects=data['subjects'];q=data['half_brier']
assert p.shape==(3,5,2947,6) and np.array_equal(data['case_id'],np.arange(2947))
recomputed=.5*np.sum((p-np.eye(6)[y])**2,axis=3)
assert np.max(np.abs(recomputed-q))<2e-15
assert np.array_equal(p.argmax(3)==y,data['correct'])
metricp=p/p.sum(3,keepdims=True)
negative=-np.log(np.clip(np.take_along_axis(metricp,np.broadcast_to(y,(3,5,2947))[...,None],3)[...,0],np.finfo(np.float64).eps,1.))
assert np.max(np.abs(negative-data['negative_log_probability']))<2e-14
assert np.max(np.abs(q[:,0,None,:]-q[:,1:,:]-data['hgb_minus_baseline_half_brier']))<2e-15
conformance=[]
for si,seed in enumerate(data['seeds']):
    with np.load(ROOT/'results'/'har_certified_test'/'locked_execution_v1'/f'seed{seed}'/'all_test_outcomes.npz',allow_pickle=False) as z:
        assert np.array_equal(z['subjects'],subjects) and np.array_equal(z['labels'],y)
        difference=float(np.max(np.abs(p[si,0]-z['reference_probabilities'])))
    assert difference<2e-12
    conformance.append(dict(seed=int(seed),maximum_hgb_vs_formal21_reference_probability_difference=difference))
    for fi,family in enumerate(data['families']):
        m=next(m for m in metrics['models'] if m['seed']==seed and m['family']==family)
        assert abs(m['half_brier']-float(q[si,fi].mean()))<2e-15
        assert abs(m['accuracy']-float(data['correct'][si,fi].mean()))<2e-15
        assert abs(m['log_loss']-float(negative[si,fi].mean()))<1e-12
        prior=[m for m in old['models'] if m['seed']==seed and m['family']==family]
        if prior:
            assert prior[0]['half_brier']==m['half_brier'] and prior[0]['accuracy']==m['accuracy']
for sj,subject in enumerate(bs['subject_ids']):
    rows=subjects==subject
    assert int(rows.sum())==bs['subject_case_counts'][sj]
    assert np.max(np.abs(q[:,:,rows].sum(2)-bs['subject_half_brier_sums'][:,:,sj]))<2e-12
    assert np.array_equal(data['correct'][:,:,rows].sum(2),bs['subject_correct_counts'][:,:,sj])
verification=dict(status='PASS',recovery_stage=run['recovery_stage'],frozen_root_lock_unchanged=True,
                  original_reference_script_unchanged=True,new_model_fits=0,models=15,test_subjects=9,
                  principal_prediction_assignments_unchanged=True,prior12_half_brier_accuracy_identical=True,
                  per_case_risk_accuracy_and_metric_logloss_recomputed=True,paired_subject_bootstrap_inputs_recomputed=True,
                  hgb_formal21_conformance=conformance,bootstrap_executed=False,
                  maximum_metric_only_normalization_difference=max(m['maximum_metric_only_normalization_difference'] for m in metrics['models']))
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
competence=json.loads((OUT/'reference_competence.json').read_text())
lines=['# Frozen complete-reference test competence',
       '', 'All15 frozen HGB/RF/ET/LR/SVC family/system combinations evaluated all2947 official test windows from nine held-out subjects. No model, temperature, scaler, or partial policy was fitted or reselected. Full-model prediction time is diagnostic only and is not a complete raw processing-time comparison.',
       '', '|System|Family|Frozen T|Half-Brier|Accuracy|Secondary log loss|',
       '|---:|---|---:|---:|---:|---:|']
for m in metrics['models']:
    lines.append(f"|{m['seed']}|{m['family']}|{m['temperature']:g}|{m['half_brier']:.6f}|{100*m['accuracy']:.2f}%|{m['log_loss']:.6f}|")
lines+=['', '|Family|Mean system half-Brier|Mean system accuracy|Mean secondary log loss|',
        '|---|---:|---:|---:|']
for family,m in competence['seed_averaged_window_weighted_metrics'].items():
    lines.append(f"|{family}|{m['half_brier']:.6f}|{100*m['accuracy']:.2f}%|{m['log_loss']:.6f}|")
lines+=['', 'SVC has lower observed test half-Brier than frozen HGB in every system; HGB exceeds SVC by0.006541,0.014170 and0.007403. The descriptive three-system means are HGB0.061363 and SVC0.051992, a difference0.009371. HGB beats RF and LR on this representation in every system, and beats ET in systems17/43 while ET is slightly better in system29. The reference is therefore a competent tree predictor, with an observable gap to SVC; the study cannot claim the strongest available HAR accuracy or probability predictor.',
        '', 'These comparisons diagnose the frozen reference. They do not change it. The method certificate currently supports numerical trees, so replacing the reference with SVC based on test outcomes would both violate the freeze and change the evaluated capability.',
        '', 'All15 case probability/loss arrays,135 subject-model records, paired HGB-minus-baseline case/subject differences and shared-nine-subject sums are preserved. Root owns the common subject bootstrap; this run has no confidence interval or independent27-cluster claim.',
        '', 'The first attempt retained12 successful records and three LR metric-conformance failures in locked_execution_v1. Root authorized a separate post-lock technical recovery. Only secondary log loss uses a separately row-normalized float64 copy; original probabilities, half-Brier and accuracy retain the frozen calculation. The original locked script and root lock are byte-identical; the recovery manifest records POSTLOCK_SECONDARY_METRIC_CONFORMANCE_REPAIR and the new script hash. All12 original successful half-Brier/accuracy results are exactly identical in the recovery.',
        '', 'Saved-case risk/accuracy/log-loss, paired subject sums, recovery prediction AST, unchanged locked hashes and complete-reference agreement with the formal21 driver passed verification. The maximum HGB probability discrepancy is below2e-12 (see verification.json). No retraining or runtime rerun occurred for this audit.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(verification))
print(json.dumps(competence['seed_averaged_window_weighted_metrics']))
