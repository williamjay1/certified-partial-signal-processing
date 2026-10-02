"""Root-controlled freeze creation. Run after all training/dev scripts finish.

Hashes opaque bytes, never decodes a held-out outcome or raw test window.
Refuses replacing a prior freeze. Does not itself execute the tests.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json

ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
OLD=Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
COMP=Path(r'D:\MLWork\ASC_ComputeValue_20261002')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def entry(path):return dict(path=str(path),sha256=digest(path))

def main():
    lockpath=ROOT/'research'/'FROZEN_TEST_LOCK_20261002.json'
    if lockpath.exists():raise FileExistsError('A freeze exists; it will not be rewritten')
    selection=ROOT/'results'/'har_model_selection'/'20261002T052208Z_75415f24'/'selected_hgb.json'
    chosen=json.loads(selection.read_text(encoding='utf-8'))
    schedulepath=ROOT/'results'/'har_checkpoint_development'/'schedule_development_v1'/'selected_schedule.json'
    schedule=json.loads(schedulepath.read_text(encoding='utf-8'))
    rawmanifest=ROOT/'datasets'/'raw_source_manifest_20261002T051056Z.json'
    source=json.loads(rawmanifest.read_text(encoding='utf-8'))
    v4=ROOT/'results'/'leaf_mask_backend_repair'/'prototype_frozen_configuration_before_validation.json'
    config=json.loads(v4.read_text(encoding='utf-8'))
    hydropath=Path(r'F:\AcademicData\ASC_SelectiveReplanning_20260930\raw\hydraulic_source__20260930T122053Z_b6069f27.zip')
    hscripts=[ROOT/'scripts'/'hydraulic_certified_test.py',ROOT/'scripts'/'run_leaf_mask_pilot.py',
       ROOT/'scripts'/'run_hydraulic_prototype.py',ROOT/'scripts'/'certificate_kernel.py',
       ROOT/'research'/'leaf_mask_kernel.py',COMP/'scripts'/'raw_compute.py',OLD/'scripts'/'methods.py']
    scripts=[ROOT/'scripts'/'har_certified_test.py',ROOT/'scripts'/'har_reference_test.py',
        ROOT/'scripts'/'har_stopping_controls.py',ROOT/'scripts'/'har_development_benchmark.py',
        ROOT/'scripts'/'har_checkpoint_development.py',
        ROOT/'scripts'/'certificate_kernel.py',ROOT/'research'/'leaf_mask_kernel.py',
        ROOT/'research'/'prepare_har_training.py',ROOT/'research'/'prepare_har_test.py',
        ROOT/'scripts'/'analyze_heldout.py',ROOT/'research'/'control_design.py']
    orders={str(s):json.loads((ROOT/'results'/'har_compute_development'/'corrected_execution'/f'seed{s}'/'frozen_order_and_cohort_before_development.json').read_text(encoding='utf-8')) for s in (17,29,43)}
    # Control-state contract is finalized by its owner before this script runs.
    controls=ROOT/'results'/'har_stopping_controls_development'/'frozen_controls_C_v1'/'frozen_control_state.json'
    lock=dict(stage='FROZEN_BEFORE_TEST_ACCESS',created_utc=datetime.now(timezone.utc).isoformat(),
        official_test_access_authorized=True,har_method_frozen=True,primary_delta=.025,
        primary_endpoint='paired complete software-time reduction at locked relative half-Brier tolerance; same timed-case risk reported',
        seeds=[17,29,43],deltas=[0.,.01,.025,.05,.1],repetitions=3,
        source_sha256=source['sha256'],source_manifest=entry(rawmanifest),
        train_cache_sha256=digest(ROOT/'cache'/'har_train_only__20261002T051217Z_533d2ce1.npz'),
        selected_hgb_sha256=digest(selection),selection=entry(selection),
        selected_schedule_sha256=digest(schedulepath),selected_schedule=schedule,
        orders=orders,models={s:entry(Path(v['model_path'])) for s,v in chosen.items()},
        control_state=entry(controls),test_loader_sha256=digest(ROOT/'research'/'prepare_har_test.py'),
        har_compute_script_sha256=digest(ROOT/'scripts'/'har_development_benchmark.py'),
        checkpoint_script_sha256=digest(ROOT/'scripts'/'har_checkpoint_development.py'),
        certificate_kernel_sha256=digest(ROOT/'scripts'/'certificate_kernel.py'),
        leaf_helper_sha256=digest(ROOT/'research'/'leaf_mask_kernel.py'),
        har_operator_sha256=digest(ROOT/'research'/'prepare_har_training.py'),
        certified_test_script_sha256=digest(ROOT/'scripts'/'har_certified_test.py'),
        frozen_order_sha256={s:digest(ROOT/'results'/'har_compute_development'/'corrected_execution'/f'seed{s}'/'frozen_order_and_cohort_before_development.json') for s in ('17','29','43')},
        checkpoints=schedule['checkpoints'],timing_cohort_seed=7302026,timing_max_windows_per_subject=100,
        timing_repetitions=3,main_specifications=17,uniform_A_primary_schedule_control=schedule['schedule']!='A',
        stopping_control_script_sha256=digest(ROOT/'scripts'/'har_stopping_controls.py'),
        stopping_control_state_sha256=digest(controls),control_design_sha256=digest(ROOT/'research'/'control_design.py'),
        additional_stopping_controls=['score_separation_argmax','fixed_prefix_mean_completion','entropy_mean_completion'],
        reference_script_sha256=digest(ROOT/'scripts'/'har_reference_test.py'),
        model_selection_records_sha256=digest(selection.parent/'candidate_records.json'),
        scripts={str(p):digest(p) for p in scripts},
        timing_cohort=dict(rule='uniform at most100 windows per test subject, same cohort all systems',seed=7302026),
        bootstrap=dict(unit='same nine subjects shared across systems',repetitions=10000,seed=7302026),
        hydraulic=dict(kind='HYDRAULIC_EXTERNAL_CONFIRMATION',seed=17,batch_size=4,primary_delta=.025,
            deltas=[0.,.01,.025,.05,.1],variants={'tight_minimax':0,'coordinate_minimax':1,'tight_midpoint':2},
            repetitions=3,timing_seed=20261011,order=config['order'],
            model=entry(COMP/'results'/'redesign_diagnostic'/'full_hgb_seed17.joblib'),
            development_cache=entry(COMP/'cache'/'hydraulic_development_raw_v1.npz'),
            development_manifest=entry(COMP/'datasets'/'hydraulic_development_manifest.json'),
            prepared_split=entry(OLD/'datasets'/'hydraulic_valve.npz'),raw_source=entry(hydropath),
            v4_configuration=entry(v4),scripts={str(p):digest(p) for p in hscripts},controls=[]),
        disclosures=['Checkpoint and mean-risk stopping settings selected on development data',
           'Hydraulic outer prepared-feature test previously used by earlier methods; certified raw pipeline frozen before confirmation',
           'No official HAR test outcomes or signal members decoded before this freeze'])
    with lockpath.open('x',encoding='utf-8') as f:json.dump(lock,f,indent=2,ensure_ascii=False)
    print(json.dumps({'lock_path':str(lockpath),'sha256':digest(lockpath),'created_utc':lock['created_utc']}))

if __name__=='__main__':main()
