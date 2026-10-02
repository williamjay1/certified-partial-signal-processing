# Complete HAR model time: post-test descriptive diagnostic

This diagnostic addresses the incomplete cost comparison of the fifteen frozen complete predictors. Earlier full-reference evaluation timed prediction on prepared features. It established prediction quality, but did not measure deployment from resident raw windows through all feature processing. The present experiment measures that complete software path, including SVC.

The scope is descriptive after official test access. No model, temperature, scaler, cohort, block definition or primary certified result changes. New native HGB and compiled HGB measurements cannot replace the frozen formal main experiment. Original formal complete timings will be copied with their source identity and shown separately.

## Frozen comparison and execution design

Use the original900 subject-selected windows shared by all systems. The original cohort seed7302026 is reconstructed using only subject IDs and checked against `har_certified_test/locked_execution_v1/frozen_shared_timing_cohort_before_predictions.json`. The same fifteen bundles selected before official test access supply HGB, RF, ET, LR and SVC for systems17/29/43. Each bundle retains its saved fit-only float32 center/scale and its temperature. Calibrated HGB compiled forests are additional implementations of the same three HGB probabilities, not additional fitted models.

For each resident raw window, independently allocate an observation and mask, compute all18 genuine statistic/FFT blocks, apply saved standardization, and evaluate the selected native family followed by its unchanged temperature transform. Allfamilies within a system share that system's frozen eighteen-block order. HGB receives float64 coordinates holding the float32 processing representation, as in the formal full path; other native families receive float32 coordinates as in their earlier reference predictions. HGB uses its raw logits divided by temperature. RF/ET/LR/SVC use softmax(log(clipped native predict_proba)/temperature), exactly matching the earlier frozen reference semantics.

There are18 implementations (15 native plus3 compiled HGB). Within each case and each of three repetitions, shuffle all18 implementations using fixed timing seed7342026. Total planned records:900×3×18=48,600. Every call recomputes its own raw features; no feature processing is shared between competing methods or replaced by cached coordinates. Warm up every implementation on the first common cohort case. Cache loading, JIT, model loading, warmup, reference/feature checks, labels and serialization stay outside complete timers. All18 blocks, allocations, scaling, native prediction and probability/temperature work stay inside.

Compare every timed prediction with the previously saved full-reference array and require the same argmax. Float64 native families and compiled HGB use2e−12 probability tolerance. LR keeps its native float32 probabilities; a prior batch/singleton BLAS rounding discrepancy is allowed only within the fixed16×float32 epsilon (1.90735e−6), recorded in full. No output is replaced or renormalized to pass. Repeated singleton predictions must match exactly. Independently check every computed coordinate against the frozen cached feature/scaler representation. These checks happen after each timer ends.

Use one thread, set Python/Numba/joblib temporary and cache paths to D, check D space, and wait for the priority component-profile CPU release before execution. Allfresh results are stored under `results/review2_full_model_timing/<UTC_uuid>` with the root lock, source/operator/model identities and protocol recorded before this diagnostic. Raw source archives are not read. Frozen source/operator files and allprior outputs are unchanged.

The report will contain mean/median/P95/P99 complete time, half-Brier and accuracy for each of the18 implementations and a separate pooled descriptive summary over three systems. Time pooling includes3×900×3 observations; risk pools3×900 probabilities. The shared nine test subjects and three systems are not independent cohorts. No significance test, new tolerance selection or test-driven model replacement is included.

## Backend scope and possible extensions

This diagnostic supports native full prediction for the fifteen actual frozen sklearn bundles. RF and ET full timings are therefore available, but this does not establish that the partial certificate implementation supports their voting semantics. The current `FrozenForest` implementation extracts sklearn histogram-gradient-boosting `_predictors` and additive multiclass logits. General pointwise proper-loss reasoning can be applied whenever a valid set of full probabilities is available, but an implementation for another model requires independently validated reachable probability/score bounds and complete cost measurement.

XGBoost and LightGBM would require adapters for their own tree storage, missing/categorical rules, base scores, objective links and multiclass grouping; no such frozen tested adapters are present here. Neural networks need an independently sound bound propagation backend and its overhead evaluated on the actual processing workload. No RF, XGBoost, LightGBM or neural-network partial-processing support is claimed from this diagnostic alone. These are implementation requirements, not measured results or guaranteed future improvements.

## Status

Static script preparation only. Execution and results will be appended after coordinated CPU release. The diagnostic has not yet been run at the time of this preparation entry.


## Executed diagnostic and independent verification

The authorized post-test run completed with exit0 in330.7966s after the component-profile CPU release. Fresh output: `results/review2_full_model_timing/20261002T071701Z_1526924f`. All48,600 timed calls were retained, with no fit, scaler estimation, temperature selection or policy change. This completes the previously prepared status entry above.

An independent saved-output audit checked every JSONL row against NPZ, unique3×6×900×3 coverage, all per-system and pooled means/medians/P95/P99, two algebraically equivalent half-Brier formulas, accuracies and artifact hashes. Allchecks passed. Each repeated probability was identical; allraw computed coordinates exactly matched the saved feature/scaler representation. HGB/RF/ET/SVC native probabilities exactly matched their prior full-batch arrays. Calibrated compiled HGB maximum difference was7.77156e−16. LR kept its native float32 singleton outputs: maximum differences from oldbatched predictions were1.31130e−6,9.83477e−7 and1.07288e−6 for systems17/29/43, respectively, below the prespecified1.90735e−6 bound, with zero class disagreements. LR risk uses the actual singleton outputs, not substituted batch arrays.

### Pooled descriptive complete cost and quality

Each time row pools8,100 timings across three frozen systems, same900 windows andthree repetitions. Risk and accuracy pool2,700 predictions across the same three systems. These are not independent cohorts. P95/P99 refer to the mixture of system and case observations and are software diagnostics, not deadline guarantees.

| Native family / implementation | Mean ms | Median ms | P95 ms | P99 ms | Half-Brier | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HGB | 5.935996 | 5.636200 | 9.005495 | 10.461700 | 0.066411277 | 91.1111% |
| RF | 14.798855 | 13.466150 | 22.953085 | 26.324272 | 0.075713274 | 89.8519% |
| ET | 15.155591 | 13.812650 | 23.348470 | 26.783967 | 0.073094314 | 90.7407% |
| LR | 1.295108 | 1.196150 | 2.024415 | 2.360212 | 0.079415334 | 89.6667% |
| SVC | 1.625706 | 1.498900 | 2.519920 | 2.975855 | 0.056552867 | 92.5556% |
| HGB_compiled | 1.149548 | 1.049300 | 1.802545 | 2.151405 | 0.066411277 | 91.1111% |

SVC has lower observed risk and higher accuracy than frozen HGB in allthree systems, with pooled complete time1.625706ms. It is faster than native HGB but slower than the same-run compiled HGB (1.149548ms), which preserves HGB probability semantics. This is an observed quality–complete-cost tradeoff; the proposed partial HGB procedure cannot be described as comprehensively outperforming SVC or as correcting the HGB reference's prediction errors. RF and ET complete online single-window costs are much larger than their earlier amortized batched-prediction costs; neither result should be represented as the other's deployment time. Allfamily outputs include genuine raw processing here.

### Per-system complete cost and quality

| System | Native family / implementation | Mean ms | Median ms | P95 ms | P99 ms | Half-Brier | Accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 17 | HGB | 6.496526 | 6.135300 | 9.432615 | 10.650118 | 0.063736103 | 90.6667% |
| 17 | RF | 14.773042 | 13.469800 | 22.766340 | 26.257887 | 0.079828798 | 89.4444% |
| 17 | ET | 15.087722 | 13.795350 | 23.149375 | 26.546940 | 0.073494219 | 90.0000% |
| 17 | LR | 1.292739 | 1.200450 | 2.027345 | 2.340719 | 0.075272893 | 89.7778% |
| 17 | SVC | 1.625364 | 1.506600 | 2.534295 | 2.975925 | 0.054688168 | 92.6667% |
| 17 | HGB_compiled | 1.171053 | 1.071750 | 1.831035 | 2.181486 | 0.063736103 | 90.6667% |
| 29 | HGB | 6.524434 | 6.152250 | 9.661620 | 10.676556 | 0.083996023 | 89.5556% |
| 29 | RF | 14.833987 | 13.473300 | 22.860425 | 26.512375 | 0.085781010 | 88.0000% |
| 29 | ET | 15.225532 | 13.828700 | 23.665920 | 26.874508 | 0.080459660 | 90.0000% |
| 29 | LR | 1.300259 | 1.196750 | 2.016040 | 2.377744 | 0.102606143 | 87.2222% |
| 29 | SVC | 1.612925 | 1.492100 | 2.464740 | 2.871955 | 0.068264788 | 91.2222% |
| 29 | HGB_compiled | 1.161860 | 1.065200 | 1.844800 | 2.201102 | 0.083996023 | 89.5556% |
| 43 | HGB | 4.787026 | 4.524050 | 7.011935 | 8.020733 | 0.051501704 | 93.1111% |
| 43 | RF | 14.789536 | 13.459800 | 23.202030 | 26.403209 | 0.061530014 | 92.1111% |
| 43 | ET | 15.153518 | 13.820650 | 23.348835 | 26.845212 | 0.065329062 | 92.2222% |
| 43 | LR | 1.292326 | 1.191700 | 2.023940 | 2.363790 | 0.060366965 | 92.0000% |
| 43 | SVC | 1.638828 | 1.500150 | 2.568040 | 3.047841 | 0.046705644 | 93.7778% |
| 43 | HGB_compiled | 1.115732 | 1.016600 | 1.744325 | 2.071118 | 0.051501704 | 93.1111% |

### Unchanged formal HGB complete controls

The following are copied from the frozen official test run, not recomputed or replaced by this diagnostic. Timing conditions differ between experiments; use contemporaneous controls for percentage reductions, and do not pair this new SVC measurement with an old partial HGB timer to manufacture a new controlled win. The original HGB certified result remains assessed against its own faster formal full control.

| System | Formal implementation | Mean ms | Median ms | P95 ms | P99 ms |
|---:|---|---:|---:|---:|---:|
| 17 | native_full | 6.927671 | 6.480100 | 9.797410 | 11.104567 |
| 17 | compiled_full | 1.122237 | 0.982250 | 1.908330 | 2.320208 |
| 29 | native_full | 6.771020 | 6.414300 | 9.252250 | 10.511947 |
| 29 | compiled_full | 1.051749 | 0.915500 | 1.600410 | 2.097817 |
| 43 | native_full | 5.185759 | 4.864550 | 7.417620 | 8.644539 |
| 43 | compiled_full | 1.066047 | 0.930950 | 1.770430 | 2.225132 |

Original formal HGB compiled means average1.080011ms; new mixed-diagnostic compiled means average1.149548ms. This difference is retained as run-to-run timing variation, with no selected favorable rerun. The new native HGB pipeline also differs from its formal native wrapper by allocation/prediction interface details; both reconstruct the same frozen complete probabilities. Neither new complete timer changes the original controlled partial-processing inference.

No RF, XGBoost, LightGBM or neural-network partial-bound adapter was implemented or validated by this experiment. Native RF/ET timings establish only complete prediction costs for the actual frozen bundles. The current certificate backend remains the tested numerical multiclass sklearn HGB backend.

Outcome SHA256: `1cde16bc8fcf6243c5807089c11da87d2439712501ef6efa9176b6fa1e677413`.
Case-record SHA256: `13d9d6ba7a36ebb38cba9359c6208145b66146ad3e6f2c1ed0d86e2fa3f87320`.
Executed script SHA256: `a040b7a7e64f2d240c4b61095c262de74e1da5befa5d9032cadbca3be371bf44`.
