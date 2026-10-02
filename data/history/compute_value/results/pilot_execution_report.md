# Locked raw-signal gate pilot: execution report

Executed once on 2 October 2026 after the primary agent's `PRE_EXECUTION_GO`. Process exited 0 in 69.787 seconds. No technical repair, outcome-dependent rerun, HAR download, extra fitting seed or expanded baseline training followed the result.

## What actually ran

The existing immutable Hydraulic zip was checksum-verified and read once. Only the old outer training and validation signals were parsed: 1770 cycles and 146 contiguous operating-condition proxy segments. The 435 old test signals were not parsed or featurized. The resulting D-resident raw cache has SHA-256 `32bf69c825243b6f99306237e14e11b0d54559eefc295970e4eb0c7c69d93d38`.

Whole-segment allocation preceded case sampling. Predictor fitting used 907 cycles from 76 segments; internal validation used 223 from 25. Gate fitting was allocated 380 cycles from 19 segments, from which 300 were sampled; calibration and audit each had 130 assigned cycles from 13 segments, with 100 sampled from each. The 140 unused gate-partition cases remained excluded from classifier fitting and validation. The five assigned segment sets are disjoint.

The new representation consists of 28 separately computed blocks and 238 features: each of 14 physical sensors has 11 deterministic statistics and six rFFT summaries. Scaling and the donor pool use predictor fitting data only. The common initial block was EPS1 statistics, selected by training-average immediate risk improvement per measured block time among the statistics blocks. It fit within the fixed estimated allowance of 0.30 times the sum of the measured block costs.

The one masked classifier used seed 17 and a 70-epoch cap. It early-stopped at 31 epochs; the saved internal validation cross-entropy was 1.2805999517 and temperature was 1.25. This is the single fitted pilot model, not a multi-seed robustness result. The student learned 200 deep-planner teacher actions on predictor-fitting cases only. It did not fit on the 300 gate-fitting terminal-loss labels. A fixed ridge model learned all five branches' same-case terminal half-Brier losses and median complete runtimes. A separate repeated-value adaptation learned 900 gate-fitting prefixes at depths 0, 2 and 4. It is not an exact reproduction of the old incompatible 154-feature gate.

Three timing repetitions used randomized method order and serial case execution. Torch, OpenBLAS and OpenMP runtime thread counts were 1. The timer started with the raw 60-second window already resident in host memory, and included initialization, real selected-block extraction and scaling, routing, planning, continuation and final prediction. Sensor acquisition, transmission, file transfer and offline fitting are outside this software-time endpoint. No artificial sleep or fabricated sensor purchase price was used.

## Locked selection and result

The calibration static mean risk was 0.3369500622, so the prospectively fixed tolerance produced a risk ceiling of **0.3389500622**. All ten compute-value tradeoffs failed this calibration ceiling. The fixed minimum-risk fallback was lambda 0 and was retained only as a diagnostic. It is the same risk-only depth selector; it cannot establish an independent benefit from learning costs.

The globally optimal fixed mixture under the counterfactual calibration envelope used 9.5434% skip and 90.4566% static. This mixture was frozen and actually deployed using fixed case-keyed random draws. Its direct routing cost and deviations from the expected envelope are reported separately. Entropy and repeated-value controls were also selected on calibration and frozen before opening audit metrics.

| Frozen audit policy | Mean half-Brier risk | Mean full software time (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Skip | 0.3627914169 | 0.225435 | 0.306320 | 0.481585 |
| Static | 0.3565564838 | 0.850881 | 1.056625 | 1.849708 |
| Student | 0.3645779568 | 0.815762 | 1.216275 | 1.968381 |
| Shallow rollout | 0.3597690725 | 2.203653 | 3.285535 | 4.041642 |
| Deep rollout | 0.3701964337 | 3.510959 | 4.974110 | 5.702310 |
| Once compute-value diagnostic | 0.3624303114 | 0.786146 | 3.175750 | 4.521187 |
| Quality-only selector | 0.3624303114 | 0.775114 | 3.044055 | 3.804251 |
| Entropy once | 0.3565564838 | 0.993266 | 1.239490 | 2.467803 |
| Repeated value adaptation | 0.3565564838 | 1.355016 | 1.857935 | 3.005053 |
| Frozen global mixture | 0.3558951449 | 0.958294 | 1.329290 | 2.342183 |

Every frozen audit control exceeded the original calibration-fixed ceiling. The primary constrained speed comparison therefore has **no qualified benchmark**; a percentage saving against the fastest qualified control is undefined. The ceiling was not recomputed from audit data. The once diagnostic is slower than the same risk-only selector and has higher risk than static, so its reduction relative to always-deep cannot support the intended contribution.

The once diagnostic planned on 17% of audit cases and used a cheap branch on 83%; its mean number of computed blocks was 1.11. Separate component profiling measured 0.056107 ms per case for the gate and a signed net reduction of 2.199425 ms in branch decision work relative to the same case's standalone deep branch. This accounting includes all shallow, deep, student and cheap routes, without clipping or case selection. It is a matched descriptive profile, not a causal module-removal experiment. This was the **only expansion criterion that passed**.

There were no illegal actions and no violations of the estimated feature-work allowance. Separate instrumented profiling found actual feature computation exceeding that estimate in 2% of once-policy cases. Estimated allowance compliance is not a hard runtime guarantee. Mean, tails and exceedance rates are retained for all controls.

**Decision: STOP_NO_EXPANSION for this locked implementation and investment plan.** The pilot does not establish unseen-subject generalization, publication readiness, universal failure of inexpensive gates, or an independently novel planning-budget capability. The limited 100-case audit contains only 13 condition-segment proxy units; paired segment intervals are descriptive and conditional on one fitted model and one host.

## Reproducible outputs

- `hydraulic_development_manifest.json` and `frozen_split_indices.json`: source/cache provenance, units and sampled original rows.
- `execution_configuration.json`: frozen code hashes, grids, numerical libraries and runtime thread pools.
- `gate_fit_branches_paired_measurements.npz`: all five same-case branch losses and measured runtimes for 300 independent gate-fitting cases.
- `calibration_candidates_paired_measurements.npz` and `calibration_frozen_global_mixture_paired_measurements.npz`: directly deployed calibration measurements.
- `selection_lock_before_audit.json`: ceiling, infeasibility and frozen policies before audit metrics.
- `audit_frozen_policies_paired_measurements.npz` and companion execution JSONL: matched audit losses, full times, masks and action sequences.
- `audit_separate_component_profile.jsonl`: separately instrumented feature, routing, branch decision and final prediction components.
- `pilot_summary.json` and `.csv`: primary decision, all retained control results, tails, errors and paired segment intervals.
- `pilot_execution_20261002.log` and `execution_completion.json`: actual execution status and duration.

The original seven-source study and manuscript were not edited by this worker. The primary agent will incorporate the supported diagnosis and nearest-work correction into the article.
