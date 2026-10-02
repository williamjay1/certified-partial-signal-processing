# Execution and evidence boundaries

The public entry points use repository-relative inputs in `data/` and a separate
work directory. They import installed dependencies, not the author's private
`env_core`, `env_packages`, `D:/ptenv`, or original scientific drivers. The
numerical kernels, feature operators, and saved-bootstrap analysis are public
copies with narrowly documented path/import changes. `config/code_changes.json`
records both file hashes and equality of the retained function/class ASTs.

The original 39 Python source files are also preserved byte for byte under
`frozen/current`, `frozen/compute`, and `frozen/legacy`. These frozen historical
drivers retain their original Windows paths and pre-test hash checks. Their
presence is source preservation, not a claim that a fresh raw-to-timing pipeline
has been run on another computer. No pre-test lock was rewritten or backdated.

## Installation and writable paths

The executed study and this validation used Windows, Python 3.12.14, and the
observed versions pinned under `requirements/`. A clean installation of these
versions on another computer has not been tested. Linux and macOS execution are
unverified. No wheels, DLLs, Python environment, or model-dependent Torch import
is needed for the portable numerical replay.

In Windows PowerShell, use a D-drive checkout and D-drive virtual environment:

```powershell
python -m venv D:\MLWork\ASC_public_env
D:\MLWork\ASC_public_env\Scripts\python.exe -m pip install -r requirements\replay.txt
$env:ASC_WORK_ROOT = 'D:\MLWork\ASC_public_work'
$env:PYTHONPYCACHEPREFIX = 'D:\MLWork\ASC_public_work\cache\python'
```

These are installation instructions, not an independently executed clean
environment build. `ASC_DATA_ROOT` can point to a relocated copy of the release's
`data` directory; otherwise the installed code's own repository `data/` is used.
`ASC_WORK_ROOT` controls generated files, Numba cache, Matplotlib configuration,
temporary files, and Python caches. On Windows it must be an absolute D-drive
path. Generated outputs never overwrite a completed analysis. Source archives
default to `F:/AcademicData/ASC_CertifiedProcessing/raw`. An explicit E-drive raw
destination is accepted; neither drive is used for training or result output.

## Commands that were actually exercised

Run from the repository root or any directory with the repository on
`PYTHONPATH`. The publication validation used a newly created empty working
directory and only package-relative scientific inputs.

```powershell
python -m asc_certified validate
python -m asc_certified replay
python -m asc_certified replay --report D:\MLWork\ASC_public_work\replay.json
python -m asc_certified bootstrap --output D:\MLWork\ASC_public_work\bootstrap_fresh
```

`validate` needs only the Python standard library. It checks each manifest
`relative_path`, byte length, and SHA-256 and rejects duplicate, absolute,
traversal, and escaping paths. It does not claim that hash agreement proves
scientific validity. `replay` and `bootstrap` require only NumPy. Replay reads
the three saved HAR outcome/timing pairs and Hydraulic arrays, checks the shared
timed cohort and every-label certificates, and compares the recomputed numbers
to the released saved summaries. Bootstrap uses the original fixed random seed
7302026 and 10,000 common nine-subject resamples. It writes a fresh JSON and NPZ;
it does not fit a model, predict a new outcome, or remeasure latency.

For numerical kernel tests and optional model conformance:

```powershell
python -m pip install -r requirements\kernel.txt
python -m asc_certified test
python -m pip install -r requirements\models.txt
python -m asc_certified conformance --report D:\MLWork\ASC_public_work\conformance.json
```

The 16 small unit tests cover simplex KKT conditions, known minimax values,
feasible-probability certificate probes, softmax score intervals, missing-value
routing, selected-block observation, real signal-feature units, whole-subject
split ownership, and hash/traversal rejection. Synthetic fixtures are explicitly
marked. Conformance loads only the four deployment HGB references, checks their
published hashes, and compares native and compiled probabilities plus partial
score bounds and every-label certificates on eight synthetic inputs per model,
including missing values. It uses no source signals, labels, new fit, or timer.
Finite checks do not prove all-input floating-point soundness.

The four operational models are the three selected HAR HGB bundles and one
Hydraulic HGB. Complete-quality comparison models are separate evidence. Saved
reference predictions permit numerical checking without any model pickle.
Models use Python serialization and should be loaded only from this trusted
release after hash verification.

## Optional upstream source retrieval

This operation is never triggered by validation, replay, bootstrap, tests, or
conformance. Read the original dataset terms, including the in-archive README,
before reuse. This release does not rehost raw HAR signals or source features.

```powershell
python -m asc_certified download-raw --dataset har
python -m asc_certified download-raw --dataset hydraulic
```

The downloader streams only the official UCI source to a fresh F-drive filename,
checks the study's frozen byte size and SHA-256, makes the saved file read-only,
and writes its acquisition report on D. Changed upstream bytes fail closed;
failure files are retained, never silently replaced or deleted. No archive
content is parsed. This command was inspected but no new network download was
performed during release validation. Its report records this retrieval and does
not replace the historical freeze record.

## Scientific code and the full original pipeline

The two public operator modules expose `HARBlocks` and `RawBlocks`, so a reader
can compute the original selected statistic/spectral blocks from resident signal
arrays. `har_features.load_training_signals` retains the original training-only
ZIP whitelist, and `whole_subject_splits` retains the fixed subject-level split.
The certificate and leaf-mask modules contain the original numeric multiclass
HGB implementation. No neural planner or Torch dependency is imported by these
public modules. Forests support numeric splits and at most 63 leaves per tree
for the leaf-mask backend. Private scikit-learn node layouts are version-sensitive.

The byte-preserved historical launch sequence and its original prerequisites are:

| Stage | Frozen source | Original command/options or prerequisites |
|---|---|---|
| HAR source acquisition | `frozen/current/research/download_har.py` | Direct historical source downloader; current `download-raw` is preferred for a new retrieval record. |
| Train signals/features | `frozen/current/research/prepare_har_training.py` | `--manifest <original source manifest> [--output <fresh D NPZ>]`; opens training members only. |
| Full model registry/temperature | `frozen/current/scripts/har_model_selection.py` | `--execute`; original training cache and preparation manifests; 24 candidate configurations before development selection. |
| Frozen compute development | `frozen/current/scripts/har_development_benchmark.py` | `--selection <selected_hgb.json> --record-tag corrected_execution`; model and train-cache hashes checked. |
| Checkpoint choice | `frozen/current/scripts/har_checkpoint_development.py` | `--record-tag schedule_development_v1`; uses development evidence only. |
| Stopping controls | `frozen/current/scripts/har_stopping_controls.py` | `--development --record-tag <fresh tag>`; original selected schedule, models and training cache required. |
| Pre-test freeze | `frozen/current/research/create_test_lock.py` | Historical root lock, retained as evidence; do not execute to impersonate a new pre-test freeze. |
| Test signals/features | `frozen/current/research/prepare_har_test.py` | `--manifest <source manifest> --test-lock <original lock> --train-cache <original train cache> [--output <fresh D NPZ>]`. |
| Certified HAR evaluation | `frozen/current/scripts/har_certified_test.py` | `--test-lock <lock> --test-cache <test NPZ> --test-preparation-manifest <manifest> --stopping-control-state <state> --output <fresh D directory>`. |
| Full quality comparisons | `frozen/current/scripts/har_reference_test_metric_repair.py` | `--test-cache <test NPZ> --test-lock <lock> [--test-manifest <manifest>] --output <fresh D directory>`; all 15 original family models needed. |
| HAR saved inference | `frozen/current/scripts/analyze_heldout.py` | `--test-dir <saved seed folders> --output <fresh directory>`; public `bootstrap` removes its obsolete NumPy path. |
| Hydraulic preparation | `frozen/compute/scripts/raw_compute.py` | Historical source-row mapping, source archive, selected whole proxy segments, and development cache; this legacy module also imports Torch. |
| Hydraulic HGB fit | `frozen/compute/research/predictor_repair_diagnostic.py` | Historical development cache/split/source mapping; original HGB seed 17 reference fit. |
| Hydraulic core prototypes | `frozen/current/scripts/run_hydraulic_prototype*.py`, `run_leaf_mask_pilot.py` | Recorded development runs and repairs; earlier failures remain preserved. |
| Hydraulic confirmation | `frozen/current/scripts/hydraulic_certified_test.py` | `--test-lock <original lock> --output-root <fresh D directory>`; checks source, model, split and code hashes; directly reads the source ZIP. |

These are the original drivers' interfaces, not plug-and-play commands for the
relocated `frozen/` directory. The original `ROOT`, `SOURCE`, `OLD`, cache names,
script hashes, lock bytes and path-bearing manifests must all match their original
layout to replay them unchanged. Rebasing a path-bearing record changes its bytes;
that must not be presented as the original pre-test freeze. A new scientific
experiment needs a new, honestly dated run record and independent validation.

To rerun complete fitting/processing/timing, obtain the upstream raw archives and
their allowed terms, reconstruct D-drive train/test caches from the source parsing
code, retain fitting-only scalers and subject/unit partitions, install the pinned
backend, and make a documented path adapter for the historical drivers. No such
full relocated run is asserted here. The original Hydraulic development cache
is about 619 MB and is not needed for saved-output replay. HAR raw/features and
three SVC support-vector model files are not rehosted because the source's terms
need separate consideration; their predictions, evaluation evidence, and training
code remain available. Independent reproductions may generate them from upstream
data according to those terms.

Mean complete software time begins with a released window already in host memory.
It excludes acquisition, transmission, file loading, and offline training. Saved
timings are not guarantees of hard deadlines or faster tails. Validation records
separate saved-array replay, finite numerical checks, and unexecuted raw-to-timing
reproduction so that no one is mistaken for another.
