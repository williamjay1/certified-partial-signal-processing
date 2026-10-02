# Executed release validation

These checks concern this public software and recorded-output package. They do not create new study cohorts or new benchmark times.

| Check actually executed | Result and record |
|---|---|
| Primary saved-output replay | PASS: probabilities, certificates, matched cohorts, complete-time means and raw latency quantiles; `validation/saved_replay.json`. |
| Original shared-subject bootstrap | PASS: 10,000 resamples with seed 7302026; complete analysis JSON equality and exact equality of 1,060,009 elements in seven arrays; `validation/bootstrap_equality.json`. |
| Numerical unit tests | 16/16 PASS; `validation/kernel_test_log.txt`. |
| Four deployed HGB models | PASS on 32 finite synthetic probes and 1,968 certificate checks; maximum native/compiled probability difference 3.608224830031759e-16; `validation/model_conformance.json`. |
| Scientific-input isolation | Eleven observed input/model reads all from the public copy and zero from the original scientific data roots; `validation/data_read_isolation.json`. |
| Original numerical source preservation | Five retained function/class AST comparisons equal; thirty-nine original Python sources byte-identical; `config/code_changes.json`. |
| Figure/table production | Six figures and twelve tables redrawn; direct 1000 dpi PNG, vector SVG/PDF, embedded PDF fonts and strict PDF geometry checks with zero findings; `artwork/proof/ARTWORK_QA.json`. |

The execution used Windows, Python 3.12.14 and the existing installed dependencies whose observed versions are pinned under `requirements/`. The code tests ran from a newly created empty working directory. See `validation/CODE_VALIDATION.json` and `docs/CODE_EXECUTION.md` for the complete scope. Source-copy hashes and provenance are in `data/provenance/source_inventory.json`; release hashes are checked by `python -m asc_certified validate`.

No fresh independent package installation, Linux/macOS test, new upstream download, or complete relocated signal-preparation/fitting/prediction/timing pipeline is reported as completed. Finite model probes do not prove universal floating-point certificate soundness. Mean time savings do not imply a hard deadline or improved latency tails. Historical development failures and original pre-test records are preserved with their original dates and meaning.
