# Certified partial signal processing

Code and recorded research outputs for **Certified partial signal processing for activity recognition and condition monitoring**, by **Junjie Zhang**.

The method omits executable signal-feature blocks while bounding the increase in multiclass half-Brier loss relative to a frozen complete predictor, for every possible label of an individual input. The bound is conditional on valid probability enclosures. It is a fidelity guarantee to that predictor, rather than a guarantee of absolute accuracy or of a hard execution deadline.

The study uses the original inertial signals of UCI smartphone activity recognition and the multirate signals of UCI hydraulic condition monitoring. Complete software timing includes feature processing, control, enclosure updates, certification, prediction and exact fallback, starting with a signal window already resident in host memory. Acquisition, transmission, file loading and offline fitting are outside that timer.

At the locked HAR tolerance δ = 0.025, the three deployed systems reduce pooled mean complete time by **7.29%** on the same 900 windows from nine previously unseen subjects; the original shared-subject bootstrap interval is **5.07–9.45%**. Hydraulic confirmation gives **25.20%** on 435 cycles from 48 operating-segment proxies, without claiming independent physical runs or an independent-cluster confidence interval. All reported tail failures, the retained 72.1447 ms full-processing invocation, the seven-source selective-planning failure, and the failed inexpensive-gate pilot remain available. Faster means do not imply faster P99 latency.

## Contents

| Location | Contents |
|---|---|
| `asc_certified/` | Portable saved-output replay, original shared-subject bootstrap, numerical kernels, source feature operators, integrity checks and explicit upstream retrieval. |
| `data/results/` | Recorded predictions, every-label certificates, timing observations, controls, reference-model comparisons and original summaries. |
| `data/checkpoints/` | Thirteen fitted model artifacts, including the four deployed HGB references. |
| `data/provenance/` | Source checksums, original partition/freeze records, fitted scaler and feature/block metadata, and the source-copy inventory. |
| `data/history/` | Earlier negative results, development repairs, technical failures and recorded design changes. |
| `frozen/` | Thirty-nine byte-preserved historical Python sources. These retain their original execution paths and historical lock checks. |
| `artwork/` and `scripts/` | Six figures, twelve tables, their plotting sources and data cells, editable SVG, vector PDF, direct 1000 dpi PNG, and an 18-page proof. |
| `validation/` | Executed numerical checks, exact bootstrap comparisons, scientific-input isolation and artwork geometry reports. |
| `config/`, `requirements/`, `release_manifest.json` | Pinned observed dependencies, adaptation/model/source metadata and file-level SHA-256 checks. |

The Git tree contains the research outputs themselves. It does not rely on Git LFS pointers or on separately attached release assets. Raw signals and per-case source feature caches are obtained from the cited upstream repositories. Three RBF-SVC checkpoints containing individual support vectors are not redistributed; their recorded predictions, metrics, training code and configurations are included. See [data licenses](DATA_LICENSES.md) and the [data dictionary](docs/DATA_DICTIONARY.md).

## Recompute recorded results

Use a D-drive checkout and writable directory on Windows. The published checks used Python 3.12.14 and the installed versions recorded in `requirements/`; installation instructions are provided separately from claims about executed checks.

```powershell
python -m venv D:\MLWork\ASC_public_env
D:\MLWork\ASC_public_env\Scripts\python.exe -m pip install -r requirements\replay.txt
$env:ASC_WORK_ROOT = 'D:\MLWork\ASC_public_work'
$env:PYTHONPYCACHEPREFIX = 'D:\MLWork\ASC_public_work\cache\python'

D:\MLWork\ASC_public_env\Scripts\python.exe -m asc_certified validate
D:\MLWork\ASC_public_env\Scripts\python.exe -m asc_certified replay
D:\MLWork\ASC_public_env\Scripts\python.exe -m asc_certified bootstrap --output D:\MLWork\ASC_public_work\bootstrap_fresh
```

`validate` uses only the Python standard library. Replay and bootstrap need only NumPy and read the repository's saved outputs. They do not refit models, generate new predictions or remeasure time. Output paths must be fresh; existing results are never silently overwritten.

```powershell
D:\MLWork\ASC_public_env\Scripts\python.exe -m pip install -r requirements\kernel.txt
D:\MLWork\ASC_public_env\Scripts\python.exe -m asc_certified test
D:\MLWork\ASC_public_env\Scripts\python.exe -m pip install -r requirements\models.txt
D:\MLWork\ASC_public_env\Scripts\python.exe -m asc_certified conformance --report D:\MLWork\ASC_public_work\conformance.json
```

The executed validation passed 16 numerical tests, exact reconstruction of the original 10,000-bootstrap analysis, and finite synthetic checks of four deployed HGB models. The latter used 32 synthetic probes and 1,968 certificate checks; these are software checks, not additional empirical participants or timing measurements. See [execution details](docs/CODE_EXECUTION.md) and [validation scope](docs/RELEASE_VALIDATION.md).

For new work with source signals, explicit retrieval commands are documented in `docs/CODE_EXECUTION.md`. The default raw destination is `F:/AcademicData/ASC_CertifiedProcessing/raw`; processing, caches and outputs use D. Full fitting and timing drivers are preserved, with their prerequisites and path adaptation requirements stated. A clean independent installation, cross-platform execution and a complete relocated raw-to-timing run have not been established by the saved-output checks.

## Figures and tables

The original scientific artwork is drawn programmatically with AI-assisted Matplotlib scripting from recorded outputs and fixed table cells. SVG/PDF retain vector elements; 1000 dpi PNG files are direct renders. Executed checks cover all eighteen assets: no embedded raster images in SVG/PDF, embedded PDF fonts, and no findings under the recorded strict geometry checks. The source commits and third-party notices for SciencePlots and the Baker Laboratory figure resources are retained under `scripts/vendor/nature_design/`. This describes a publication-oriented visual style and implies no endorsement by Nature.

Install `requirements/plotting.txt`, then run `scripts/nature_workflow.py`, `scripts/nature_figures.py`, `scripts/nature_tables.py` and `scripts/nature_artwork_delivery.py` in that order. The original Windows Arial font is not redistributed; font availability can change rendering. Numerical replay does not require the plotting dependencies.

## Sources, citation and rights

- J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto and X. Parra. *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository, 2013. [DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K).
- N. Helwig, E. Pignanelli and A. Schütze. *Condition monitoring of hydraulic systems*. UCI Machine Learning Repository, 2015. [DOI 10.24432/C5CW21](https://doi.org/10.24432/C5CW21).

Original study software is MIT licensed. Generated research outputs and original artwork are CC BY 4.0; upstream alignment fields and vendored resources retain their own terms. The HAR archive's historical noncommercial notice and the current UCI license statement differ; both are disclosed in [DATA_LICENSES.md](DATA_LICENSES.md). Do not treat the software license as permission to relicense source data.

Use [CITATION.cff](CITATION.cff) to cite this software, with the exact commit or archived version used, and cite the source datasets. The repository is prepared as software version **1.0.0**; no Zenodo release, deposit DOI or published-article DOI is asserted. The author will connect this repository to Zenodo and archive a new release following the [handoff instructions](docs/ZENODO_HANDOFF.md).

## Author and declarations

**Junjie Zhang**, sole author and corresponding author.

1. Shanghai Academy of Global Governance & Area Studies, Shanghai International Studies University, Shanghai 201620, China.
2. School of Economics and Finance, Shanghai International Studies University, Shanghai 201620, China.

[ORCID 0009-0004-8821-4018](https://orcid.org/0009-0004-8821-4018) · junjiezhang2024@shisu.edu.cn

No specific funding; no competing interests. The study is secondary analysis of released public datasets. OpenAI Codex assisted research design, mathematical development, literature work, implementation, checking, analysis, writing and plotting. Empirical values come from executed computations and saved outputs. Exact historical model revisions and all prompts were not retained. Junjie Zhang retains sole author responsibility and final submission approval; AI is not an author. Full statements are in [AUTHOR_AND_AI_DISCLOSURE.md](AUTHOR_AND_AI_DISCLOSURE.md).
