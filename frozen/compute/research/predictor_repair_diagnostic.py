"""Authorized redesign feasibility diagnostic, strictly predictor fit/val only.

No old pilot result, tolerance, checkpoint or code is modified. Exactly two
fixed HGB classifiers are fitted, without a grid or gate/audit evaluation.
"""
from __future__ import annotations
import os,sys,json,time,shutil,hashlib
from pathlib import Path
for name in ['ASC_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
ROOT=Path(r'D:\MLWork\ASC_ComputeValue_20261002')
OUT=ROOT/'results'/'redesign_diagnostic'
sys.path.insert(0,str(ROOT/'scripts'))
import raw_compute as rc
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_info,threadpool_limits
import joblib

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def metrics(prob,y):
    losses=rc.legacy.brier(prob,y)
    pred=prob.argmax(1)
    return {'half_brier':float(losses.mean()),'accuracy':float((pred==y).mean()),
            'class_counts':np.bincount(y,minlength=4).tolist(),
            'class_risk':[float(losses[y==c].mean()) for c in range(4)],
            'predicted_class_counts':np.bincount(pred,minlength=4).tolist()}

def mask_encode(x,mask):
    return np.concatenate([x*mask[:,rc.GROUP_INDEX],mask],axis=1).astype(np.float32)

def partition_description(x,y,units):
    return {'n_cases':len(y),'n_units':len(np.unique(units)),
            'class_counts':np.bincount(y,minlength=4).tolist(),
            'class_units':[len(np.unique(units[y==c])) for c in range(4)],
            'unit_size_range':[int(np.min(np.unique(units,return_counts=True)[1])),
                               int(np.max(np.unique(units,return_counts=True)[1]))],
            'max_absolute_standardized_feature':float(np.max(np.abs(x)))}

def main():
    free=shutil.disk_usage('D:\\').free
    if free<1024**3:raise RuntimeError('D space insufficient; no storage substitution')
    if (OUT/'executed_results.json').exists():raise RuntimeError('Existing completed diagnostic; no silent rerun')
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    source=ROOT/'cache'/'hydraulic_development_raw_v1.npz'
    z=np.load(source,allow_pickle=False)
    tr=z['predictor_fit'];va=z['predictor_val']
    x_tr=z['X'][tr];x_va=z['X'][va]
    y_tr=z['y'][tr];y_va=z['y'][va]
    u_tr=z['units'][tr];u_va=z['units'][va]
    assert len(tr)==907 and len(va)==223 and not np.intersect1d(u_tr,u_va).size
    # No gate fitting/calibration/audit index arrays, outcomes or rows consumed.
    policy=json.loads((ROOT/'checkpoints'/'static_policy.json').read_text(encoding='utf-8'))
    initial=int(policy['initial_stats_block']);assert initial==12
    mask_tr=np.zeros((len(tr),28),np.float32);mask_tr[:,initial]=1
    mask_va=np.zeros((len(va),28),np.float32);mask_va[:,initial]=1
    full_tr=np.ones_like(mask_tr);full_va=np.ones_like(mask_va)
    params={'learning_rate':.1,'max_leaf_nodes':31,'min_samples_leaf':20,
            'l2_regularization':1.,'max_bins':255,'early_stopping':False,'random_state':17}
    configuration={'phase':'EXPLORATORY_REDESIGN_CLASSIFIER_FEASIBILITY',
        'allowed_data':'907 predictor-fitting cases/76 proxy units;223 internal-validation cases/25 units ONLY',
        'forbidden_evaluation':['gate_fit','gate_cal','gate_audit','old outer test','HAR'],
        'free_D_bytes_before_fit':free,'feature_scaling':'Existing predictor-fit-only standardization, unchanged',
        'full_model':{'class':'HistGradientBoostingClassifier','max_iter':150,**params},
        'masked_model':{'class':'HistGradientBoostingClassifier','max_iter':100,**params,
                        'mask_copies_per_fit_row':10,'mask_definition':'One full and one EPS1-statistics-only copy; eight independent Bernoulli whole-block masks with per-copy density Uniform(0,1), seed17. Both observed-feature values and all28mask bits supplied.'},
        'no_grid':True,'model_count':2,'seed':17,'predeclared_outcome':'representation/partial-input classifier feasibility, not an old pilot pass or new confirmatory result',
        'threads':1,'runtime_threadpools':threadpool_info(),'script_sha256':rc.digest(Path(__file__)),
        'previous_selection_sha256':rc.digest(ROOT/'results'/'selection_lock_before_audit.json'),
        'previous_pilot_summary_sha256':rc.digest(ROOT/'results'/'pilot_summary.json'),
        'checkpoint_sha256':rc.digest(ROOT/'checkpoints'/'raw_masked_predictor_seed17.pt')}
    dump(OUT/'execution_configuration.json',configuration)

    # Read-only fitting inspection and exact validation-checkpoint consistency.
    saved=rc.legacy.torch.load(ROOT/'checkpoints'/'raw_masked_predictor_seed17.pt',map_location='cpu',weights_only=False)
    rc.legacy.seed_all(17);pred=rc.legacy.Predictor(238,rc.GROUP_INDEX,4,seed=17)
    pred.net.load_state_dict(saved['weights']);pred.net.eval();pred.temperature=saved['temperature']
    rng=np.random.default_rng(17);vv=np.repeat(x_va,4,axis=0);vy=np.repeat(y_va,4)
    vm=(rng.random((len(vv),28))<np.tile(np.array([.15,.35,.65,1.]),len(x_va))[:,None]).astype(np.float32)
    with rc.legacy.torch.no_grad():
        logits=pred.net(rc.legacy.torch.tensor(mask_encode(vv,vm)))
        checkpoint_ce=float(rc.legacy.nn.functional.cross_entropy(logits,rc.legacy.torch.tensor(vy)))
    inspect={'saved_stopping_epoch':saved['fit_info']['epochs'],'saved_best_validation_ce':saved['fit_info']['validation_ce'],
        'reconstructed_checkpoint_validation_ce':checkpoint_ce,
        'ce_absolute_difference':abs(checkpoint_ce-saved['fit_info']['validation_ce']),
        'best_epoch_recorded':False,'temperature':saved['temperature'],
        'code_findings':[
            'Network is seeded before construction in the new raw-pilot wrapper.',
            'Fixed validation masks are generated before training with densities .15/.35/.65/1.',
            'A copied best validation-CE state_dict is restored before temperature calibration and serialization.',
            'Training batches independently sample rows without replacement within each batch; they do not form a once-per-epoch permutation.',
            'With stale>=15 the fit must still reach epoch31 before stopping; stop_epoch does not identify best_epoch.',
            'No obvious best-checkpoint restoration bug found; original best epoch and training trajectory were not saved and cannot be inferred from the stopping count.'
        ]}
    if inspect['ce_absolute_difference']>1e-6:inspect['code_findings'].append('WARNING: saved checkpoint CE did not reconstruct within1e-6')

    out={'frozen_masked_mlp':{'fit':{'full':metrics(pred.predict(x_tr,full_tr),y_tr),'initial_EPS1':metrics(pred.predict(x_tr,mask_tr),y_tr)},
        'internal_val':{'full':metrics(pred.predict(x_va,full_va),y_va),'initial_EPS1':metrics(pred.predict(x_va,mask_va),y_va)}}}
    with threadpool_limits(limits=1):
        full=HistGradientBoostingClassifier(max_iter=150,**params)
        t=time.perf_counter();full.fit(x_tr,y_tr);full_seconds=time.perf_counter()-t
        joblib.dump(full,OUT/'full_hgb_seed17.joblib')
        out['full_only_hgb']={'fit':{'full':metrics(full.predict_proba(x_tr),y_tr)},
                            'internal_val':{'full':metrics(full.predict_proba(x_va),y_va)},
                            'fit_seconds':full_seconds,'iterations':int(full.n_iter_),
                            'initial_input_not_evaluated':'This full-only predictor is not trained on masked inputs. Zero-filled initial-only prediction would be out of support and is not a fair partial-input baseline.'}
        print('FULL_HGB',json.dumps(out['full_only_hgb']['internal_val']),full_seconds,flush=True)

        rng=np.random.default_rng(17)
        repeated=np.repeat(x_tr,10,axis=0);aug_mask=np.zeros((len(tr),10,28),np.float32)
        aug_mask[:,0,:]=1;aug_mask[:,1,initial]=1
        density=rng.uniform(0.,1.,(len(tr),8,1))
        aug_mask[:,2:,:]=(rng.random((len(tr),8,28))<density).astype(np.float32)
        augmented=mask_encode(repeated,aug_mask.reshape(-1,28));yy=np.repeat(y_tr,10)
        partial=HistGradientBoostingClassifier(max_iter=100,**params)
        t=time.perf_counter();partial.fit(augmented,yy);partial_seconds=time.perf_counter()-t
        joblib.dump(partial,OUT/'joint_masked_hgb_seed17.joblib')
        out['joint_masked_hgb']={'fit':{
            'full':metrics(partial.predict_proba(mask_encode(x_tr,full_tr)),y_tr),
            'initial_EPS1':metrics(partial.predict_proba(mask_encode(x_tr,mask_tr)),y_tr)},
            'internal_val':{'full':metrics(partial.predict_proba(mask_encode(x_va,full_va)),y_va),
                            'initial_EPS1':metrics(partial.predict_proba(mask_encode(x_va,mask_va)),y_va)},
            'fit_seconds':partial_seconds,'iterations':int(partial.n_iter_),
            'augmented_training_rows':len(yy),'original_training_cases':len(tr),
            'input_dimensions':266,'augmentation_bytes':augmented.nbytes}
        print('MASKED_HGB',json.dumps(out['joint_masked_hgb']['internal_val']),partial_seconds,flush=True)

    shifts=[]
    for c in range(4):
        xc=x_tr[y_tr==c];xv=x_va[y_va==c]
        shift=xv.mean(0)-xc.mean(0)
        shifts.append({'class':c,'fit_n':len(xc),'val_n':len(xv),
            'rms_class_mean_feature_shift':float(np.sqrt(np.mean(shift**2))),
            'mean_abs_class_mean_feature_shift':float(np.mean(np.abs(shift))),
            'val_cells_outside_fitting_class_range':float(np.mean((xv<xc.min(0))|(xv>xc.max(0))))})
    val_full_mlp=out['frozen_masked_mlp']['internal_val']['full']['half_brier']
    val_initial_mlp=out['frozen_masked_mlp']['internal_val']['initial_EPS1']['half_brier']
    differences={'full_hgb_minus_frozen_mlp_full_risk':out['full_only_hgb']['internal_val']['full']['half_brier']-val_full_mlp,
        'joint_hgb_minus_frozen_mlp_full_risk':out['joint_masked_hgb']['internal_val']['full']['half_brier']-val_full_mlp,
        'joint_hgb_minus_frozen_mlp_initial_risk':out['joint_masked_hgb']['internal_val']['initial_EPS1']['half_brier']-val_initial_mlp}
    result={'phase':configuration['phase'],'execution_complete':True,'new_models_fitted':2,
        'old_pilot_decision_unchanged':'STOP_NO_EXPANSION','prior_gate_risk_ceiling_unchanged':True,
        'old_policy_or_checkpoint_modified':False,'gate_or_outer_test_evaluated':False,
        'predictor_fit':partition_description(x_tr,y_tr,u_tr),'internal_val':partition_description(x_va,y_va,u_va),
        'group_intersection':0,'class_conditional_feature_diagnostics':shifts,
        'fitting_code_inspection':inspect,'models':out,'risk_differences':differences,
        'total_seconds':time.perf_counter()-started,
        'conclusion_boundary':'Fixed internal-validation model feasibility diagnostic. It can reveal represented signal and a plausible prediction-model repair; it does not independently validate planning/router utility, causally locate an optimization failure, reopen the old pilot or establish generalization.'}
    assert rc.digest(ROOT/'results'/'pilot_summary.json')==configuration['previous_pilot_summary_sha256']
    assert rc.digest(ROOT/'results'/'selection_lock_before_audit.json')==configuration['previous_selection_sha256']
    assert rc.digest(ROOT/'checkpoints'/'raw_masked_predictor_seed17.pt')==configuration['checkpoint_sha256']
    dump(OUT/'executed_results.json',result)
    print('COMPLETE',json.dumps({'seconds':result['total_seconds'],'differences':differences,'old_pilot_unchanged':True}),flush=True)

if __name__=='__main__':main()
