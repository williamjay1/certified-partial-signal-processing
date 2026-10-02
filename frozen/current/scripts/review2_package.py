"""Build a local audit/reproducibility snapshot; never upload or alter sources."""
from pathlib import Path
import argparse, ast, datetime as dt, hashlib, json, shutil, sys, zipfile

ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
COMPUTE=Path(r'D:\MLWork\ASC_ComputeValue_20261002')
LEGACY=Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
LOCK=ROOT/'research'/'FROZEN_TEST_LOCK_20261002.json'
RUN=ROOT/'results'/'har_model_selection'/'20261002T052208Z_75415f24'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def write_new(p,body):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(body)

VALIDATOR=r'''"""Read-only portable manifest/file validator; no external paths are read."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for entry in m['files']:
    path=(root/entry['relative_path']).resolve()
    if root not in path.parents:raise ValueError('Unsafe manifest path')
    if not path.is_file() or path.stat().st_size!=entry['bytes']:raise ValueError(entry['relative_path'])
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    if h.hexdigest()!=entry['sha256']:raise ValueError('Hash mismatch: '+entry['relative_path'])
print(json.dumps({'status':'PASS','files':len(m['files']),'uploaded':False,'full_pipeline_portability_validated':False}))
'''

SAVED_AUDIT=r'''"""Portable read-only numerical replay of saved primary arrays and timings.
Requires NumPy, but imports no scientific driver/model and writes no result file.
It does not rerun raw feature processing, prediction, certification or timing.
"""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent/'files'/'current'
formal=root/'results'/'har_certified_test'/'locked_execution_v1'
part_means=[];full_means=[];risks=[];full_risks=[];per_system=[]
for seed in (17,29,43):
    folder=formal/f'seed{seed}'
    with np.load(folder/'timing_cohort_outcomes.npz',allow_pickle=False) as z:
        names=z['method_names'];deltas=z['deltas'];p=z['probabilities'];q=z['reference_probabilities'];y=z['labels'];cases=z['test_indices']
        i=int(np.flatnonzero((names=='tight_minimax') & (deltas==.025))[0])
        loss=.5*np.square(p[i]-np.eye(6)[y]).sum(-1);base=.5*np.square(q-np.eye(6)[y]).sum(-1)
        every=.5*(np.square(p[i]).sum(-1)-np.square(q).sum(-1))[:,None]-p[i]+q
        assert len(cases)==900 and np.max(every)<=.025+2e-12
    t=[];native=[];compiled=[]
    for line in (folder/'timing_case_results.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        if r['name']=='tight_minimax' and r['delta']==.025:t.append(r['complete_ms'])
        elif r['name']=='native_full':native.append(r['complete_ms'])
        elif r['name']=='compiled_full':compiled.append(r['complete_ms'])
    assert len(t)==len(native)==len(compiled)==2700
    full=min(float(np.mean(native)),float(np.mean(compiled)));partial=float(np.mean(t))
    part_means.append(partial);full_means.append(full);risks.append(float(loss.mean()));full_risks.append(float(base.mean()))
    per_system.append({'seed':seed,'partial_ms':partial,'faster_full_ms':full,'time_reduction':1-partial/full,'half_brier':float(loss.mean()),'reference_half_brier':float(base.mean())})
hydro=root/'results'/'hydraulic_external_confirmation'/'20261002T061642Z_3208a13f'
with np.load(hydro/'outcomes.npz',allow_pickle=False) as z:
    i=int(np.flatnonzero((z['method_names']=='tight_minimax') & (z['deltas']==.025))[0]);j=int(np.flatnonzero(z['method_names']=='compiled_full')[0])
    p=z['probabilities'][i];q=z['reference_probability'];y=z['labels'];t=z['complete_ms'];units=z['proxy_segments']
    loss=.5*np.square(p-np.eye(4)[y]).sum(-1);base=.5*np.square(q-np.eye(4)[y]).sum(-1)
    every=.5*(np.square(p).sum(-1)-np.square(q).sum(-1))[:,None]-p+q
    assert len(y)==435 and len(np.unique(units))==48 and np.max(every)<=.025+2e-12
    hydro_result={'cases':435,'proxy_segments':48,'partial_ms':float(t[i].mean()),'compiled_full_ms':float(t[j].mean()),'time_reduction':float(1-t[i].mean()/t[j].mean()),'half_brier':float(loss.mean()),'reference_half_brier':float(base.mean())}
print(json.dumps({'status':'PASS_SAVED_NUMERICAL_REPLAY','models_fitted':0,'predictions_rerun':0,'timing_rerun':0,'HAR_systems':per_system,'HAR_pooled':{'time_reduction':float(1-np.mean(part_means)/np.mean(full_means)),'half_brier':float(np.mean(risks)),'reference_half_brier':float(np.mean(full_risks))},'Hydraulic':hydro_result},indent=2))
'''

README='''# Local reproducibility and audit snapshot

This is a local package of executed research artifacts. It has not been uploaded,
assigned a DOI, or independently executed as a full pipeline on another machine.
The original source files, models, raw archives and previous results were not
modified, moved or deleted to create this package.

The package supports portable manifest/hash verification and read-only numerical
replay of the saved primary outcomes. It contains frozen scientific code, sixteen
deployed model artifacts (fifteen HAR models plus one Hydraulic HGB), model and
partition/temperature/checkpoint records, processed D-drive signal/feature caches,
primary per-case predictions and real timing records, shared-subject analysis and
the added post-test diagnostics. It does not contain the original F-drive raw ZIPs
or any large legacy Python environment. The immutable raw URLs/hashes are retained
in manifests. Working signal caches are included; they are derived study inputs,
not a replacement claim about original raw-archive custody.

## Two read-only checks

From this directory or an extracted ZIP, run `python validate_manifest.py` to
verify every bundled file using only the Python standard library. Then run
`python verify_saved_results.py` in an environment with NumPy to reconstruct the
HAR and Hydraulic primary complete-time/risk numbers from saved arrays and
per-case timing logs. These checks neither load frozen model pickles nor rerun
prediction, fitting, certification, raw processing, timing or bootstrap inference.

The unchanged `files/current/scripts/analyze_heldout.py` can rebuild the saved
shared-subject analysis with explicit `--test-dir` and `--output` paths. Its
legacy NumPy search path is preserved; when that directory is absent it falls
back to the installed NumPy. This analysis portability has only been inspected,
not exercised as a cross-machine bootstrap run for this package.

## Full experiment replay: remaining conditions

The frozen main drivers retain hardcoded original Windows D-drive roots and
strict file/model/source/lock hashes. The test caches retain the original lock
SHA. Moving a lock or changing its bytes and scientific paths does not silently
authorize a new pretest freeze. No frozen scientific file was edited or rebased
to make this archive appear self-contained. Original-to-bundled paths and hashes
are explicit in `manifest.json`.

HAR main import chain: `har_certified_test` -> `har_development_benchmark`,
`har_checkpoint_development`, `har_stopping_controls` -> `prepare_har_training`,
`certificate_kernel`, `leaf_mask_kernel`, `control_design`. It requires NumPy,
SciPy, scikit-learn, joblib, threadpoolctl, Numba and llvmlite, with the frozen
models, train/test caches, model-selection record, schedules and control state.

Hydraulic main chain: `hydraulic_certified_test` -> `run_leaf_mask_pilot` ->
`run_hydraulic_prototype` -> the old `raw_compute` -> old `methods`. The last
module imports Torch even though the certified HGB processing method does not
use a neural predictor. The two cross-project source files and associated model,
split, development cache and source manifests are captured under `files/compute`
and `files/legacy`. The unchanged formal Hydraulic driver reads the original F
raw archive directly; it cannot use the included test signal cache as a silent
replacement for that archive. The archive must be supplied with its frozen SHA
to reproduce that driver. Training-source parsing likewise needs the source
archives. No archive is downloaded by any package verification command.

Exact installed version metadata is provided in `requirements.txt`. Installing
these versions or loading the model pickles on another OS has not been tested.
Numba JIT cache and third-party wheels/DLLs are not bundled. Code inserting the
original `env_core`, `env_packages`, `D:/ptenv` or `cache/numba_env` paths remains
unchanged; a reproducer must establish a compatible installed environment and
original layout or develop and independently validate explicit path adapters.
This package is therefore **an audit-ready snapshot with a portable saved-output
replay, not a demonstrated self-contained full raw-to-timing reproduction**.

## Evidence boundaries

Primary official HAR results use the same nine test subjects and three frozen
systems; the systems are not independent cohorts. Hydraulic includes435 cycles
in48 proxy segments, after earlier prepared-feature exposure, so it is frozen-
method external confirmation. Mean savings do not guarantee deadlines or faster
tails. Theory is exact-arithmetic conditional on valid bounds; numerical checks
do not prove all-input floating-point soundness. The full-family, component,
latency and order analyses were added after the primary test and are descriptive.
SVC is a competitive quality/cost alternative, not a certified subset adapter.

All source data are the cited public UCI datasets. Source manifests, download
scripts and DOI/URL records identify acquisition and handling. This local build
does not grant an external publication license for third-party data or libraries;
review their original terms before a public release. No public code deposit,
data availability link or public reproduction success is asserted here.
'''

def build(destination):
    destination=Path(destination).resolve()
    if destination.drive.upper()!='D:':raise ValueError('Local package must be written to D')
    if destination.exists():raise FileExistsError('Package output must be fresh; do not overwrite an earlier snapshot')
    if shutil.disk_usage('D:\\').free<4*1024**3:raise RuntimeError('Insufficient D space for snapshot and ZIP')
    lock=read(LOCK);selected=read(RUN/'selected_hgb.json');records=read(RUN/'candidate_records.json')
    if sha(LOCK)!='b5efeccb191b682c415e6395960213ecd586806df61517b30685565b7098f284':raise ValueError('Original root lock changed')
    paths=set()
    def add(path):
        p=Path(path)
        if not p.is_file() or p.drive.upper()!='D:':raise ValueError('Missing/D-only snapshot source: '+str(p))
        paths.add(p.resolve())
    def tree(path,extensions=None):
        p=Path(path)
        if not p.exists():return
        for f in p.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and (extensions is None or f.suffix.lower() in extensions):add(f)
    for f in (ROOT/'scripts').glob('*.py'):add(f)
    for f in (ROOT/'research').glob('*.py'):add(f)
    for f in (ROOT/'research').glob('*.json'):add(f)
    for name in ('METHOD_DERIVATION.md','THEORY_TEXT_REVIEW.md','COMPARATOR_APPLICABILITY.md','STATISTICS_REVIEW.md','HAR_PREPARATION_PLAN.md','HAR_TEST_EXECUTION_PLAN.md','HAR_REFERENCE_TEST_PLAN.md','HYDRAULIC_CERTIFIED_TEST_PLAN.md','HYDRAULIC_CONFIRMATION_REVIEW.md','REVIEW2_FULL_MODEL_TIME.md','REVIEW2_LATENCY_ANALYSIS.md','REVIEW2_ORDER_SENSITIVITY.md','VENUE_AND_REFERENCES.md'):
        p=ROOT/'research'/name
        if p.exists():add(p)
    tree(ROOT/'datasets', {'.json'})
    for folder in ('har_certified_test/locked_execution_v1','hydraulic_external_confirmation/20261002T061642Z_3208a13f','har_reference_test/locked_metric_repair_v1','har_model_selection/20261002T052208Z_75415f24','har_compute_development/corrected_execution','har_checkpoint_development/schedule_development_v1','har_stopping_controls_development/frozen_controls_C_v1','final_analysis/har_locked_v1','leaf_mask_backend_repair','review2_full_model_timing/20261002T071701Z_1526924f','review2_latency_analysis','review2_order_sensitivity'):
        tree(ROOT/'results'/folder,{'.json','.npz','.jsonl','.csv','.md','.py'})
    # Record failed/early implementation aggregates without bulky duplicate logs.
    for folder in ('batch4_repair','compiled_fallback_repair','compiled_fallback_repair_corrected'):
        tree(ROOT/'results'/folder,{'.json','.md','.npz'})
    model_paths=[]
    for seed in (17,29,43):
        for family in ('HGB','RF','ET','LR','SVC'):
            choices=[r for r in records if r['seed']==seed and r['family']==family and (family!='HGB' or r['candidate']==selected[str(seed)]['candidate'])]
            if len(choices)!=1:raise AssertionError('Ambiguous frozen family model')
            r=choices[0];p=Path(r['model_path'])
            if sha(p)!=r['model_sha256']:raise ValueError('Frozen model hash changed')
            add(p);model_paths.append(str(p))
    hyd=lock['hydraulic']
    for key in ('model','development_cache','development_manifest','prepared_split','v4_configuration'):add(hyd[key]['path'])
    model_paths.append(hyd['model']['path'])
    for p in hyd['scripts']:add(p)
    for p in lock['scripts']:add(p)
    for path in (ROOT/'cache'/'har_train_only__20261002T051217Z_533d2ce1.npz',ROOT/'cache'/'har_test_frozen__20261002T060403Z_be087f1d.npz',ROOT/'cache'/'hydraulic_external_raw__20261002T061642Z_3208a13f.npz',COMPUTE/'research'/'predictor_repair_diagnostic.py',LEGACY/'datasets'/'raw_download_manifest.json'):
        add(path)
    tree(COMPUTE/'results'/'redesign_diagnostic',{'.json','.md','.csv'})
    destination.mkdir(parents=True,exist_ok=False)
    entries=[];source_mapping={}
    bases=((ROOT,'current'),(COMPUTE,'compute'),(LEGACY,'legacy'))
    for source in sorted(paths,key=str):
        for base,label in bases:
            if source.is_relative_to(base):relative=Path('files')/label/source.relative_to(base);break
        else:raise ValueError('Unmapped source path '+str(source))
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        source_sha=sha(source);shutil.copy2(source,target)
        if sha(target)!=source_sha or sha(source)!=source_sha:raise RuntimeError('Source/copy changed while packaging')
        entries.append(dict(relative_path=relative.as_posix(),original_path=str(source),bytes=target.stat().st_size,sha256=source_sha))
        source_mapping[str(source).lower()]=relative.as_posix()
    imports=[];local_names={Path(e['original_path']).stem for e in entries if Path(e['original_path']).suffix=='.py'}
    for e in entries:
        if Path(e['original_path']).suffix!='.py':continue
        syntax=ast.parse((destination/e['relative_path']).read_text(encoding='utf-8-sig'))
        roots=set()
        for node in ast.walk(syntax):
            if isinstance(node,ast.Import):roots.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:roots.add(node.module.split('.')[0])
        imports.append(dict(file=e['relative_path'],local_imports=sorted(roots & local_names),third_party_or_stdlib=sorted(roots-local_names)))
    freeze_refs=[]
    for p,h in lock['scripts'].items():freeze_refs.append((p,h,'HAR scientific code'))
    for p,h in hyd['scripts'].items():freeze_refs.append((p,h,'Hydraulic scientific code'))
    for key in ('model','development_cache','development_manifest','prepared_split','v4_configuration','raw_source'):
        item=hyd[key];freeze_refs.append((item['path'],item['sha256'],'Hydraulic '+key))
    for key,item in lock['models'].items():freeze_refs.append((item['path'],item['sha256'],'HAR seed'+key+' model'))
    freeze_checks=[]
    for path,expected,role in freeze_refs:
        rel=source_mapping.get(str(Path(path)).lower())
        if rel and sha(destination/rel)!=expected:raise ValueError('Bundled frozen dependency SHA mismatch')
        freeze_checks.append(dict(role=role,original_path=path,sha256=expected,bundled_relative_path=rel,status='verified_bundled' if rel else 'external_not_bundled'))
    manifest=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='LOCAL_SNAPSHOT_NO_UPLOAD',source_lock_sha256=sha(LOCK),files=entries,total_source_bytes=sum(e['bytes'] for e in entries),frozen_models=len(model_paths),model_original_paths=model_paths,root_mapping={str(p):'files/'+name for p,name in bases},import_inventory=imports,frozen_dependency_checks=freeze_checks,excluded=['F-drive original raw archives','third-party environment directories/wheels/DLLs','Numba JIT caches','nonselected model artifacts','manuscript changing during snapshot'],portability=dict(manifest_hash_validation=True,saved_primary_numerical_replay='prepared_and_to_be_executed',full_raw_to_timing_cross_machine=False,absolute_paths_preserved=True),public_release=False,DOI=None)
    requirements='# Observed installed versions; cross-machine installation is not verified.\n# Python 3.12.14\nnumpy==2.5.3\nscipy==1.18.1\nscikit-learn==1.9.1\njoblib==1.6.0\nthreadpoolctl==3.7.0\nnumba==0.68.0\nllvmlite==0.50.0\n# Unchanged Hydraulic legacy import chain additionally requires:\ntorch==2.13.0\n# Plot creation only (not primary execution or portable validation):\nmatplotlib==3.11.2\n'
    write_new(destination/'README.md',README);write_new(destination/'requirements.txt',requirements)
    write_new(destination/'validate_manifest.py',VALIDATOR);write_new(destination/'verify_saved_results.py',SAVED_AUDIT)
    write_new(destination/'manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    # Archive all snapshot members using storage rather than CPU-heavy compression.
    archive=destination/'local_reproducibility_20261002.zip'
    members=sorted(p for p in destination.rglob('*') if p.is_file() and p!=archive)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in members:z.write(p,p.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC validation failed')
        if set(z.namelist())!={p.relative_to(destination).as_posix() for p in members}:raise ValueError('ZIP member coverage mismatch')
        for e in entries:
            h=hashlib.sha256()
            with z.open(e['relative_path']) as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=e['sha256']:raise ValueError('Archived SHA mismatch')
    validation=dict(status='PASS_STRUCTURE_MANIFEST_AND_ZIP_HASHES',files=len(entries),source_bytes=manifest['total_source_bytes'],frozen_models=len(model_paths),zip_path=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive),zip_members=len(members),raw_archives_bundled=False,environments_bundled=False,external_upload=False,full_pipeline_cross_machine_validated=False)
    write_new(destination/'package_validation.json',json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true',required=True);p.add_argument('--output-directory',type=Path,default=ROOT/'reproducibility')
    args=p.parse_args();build(args.output_directory)
