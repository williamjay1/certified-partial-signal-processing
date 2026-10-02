# Review 2: latency and processing-outcome analysis

Prospective diagnostic record, 2 October 2026. The user's reviewer asks what accompanies HAR P99 deterioration, how fallback relates to latency, and which operations consume time. The following analyses strengthen that interpretation without retuning the tested method or assigning a causal effect to stopping outcomes.

1. Read only the frozen formal 21-method JSONL. Pair primary tight minimax at delta .025 with compiled full on the same case/repetition. Report per-system, per-subject, 12/16/18-block and early-stop/fallback strata. Case group statistics first average the three original repeats; raw-call tails select calls at or above each system's empirical P99. Report membership and paired time increments for both partial-tail and full-tail sets. This is post-test descriptive diagnosis.
2. Retain the same formal A/C checkpoint comparison on the same900 cases. Original A was already frozen and measured alongside C; no new checkpoint search is performed.
3. After saved-output analysis, profile36 windows drawn from the original900 cohort, four uniformly sampled per held-out subject with RNG9302026, shared across allthree systems. Sampling uses IDs/subjects only; slow or fallback cases are not specially selected. Three repetitions randomize instrumented primary/full order. Models, temperatures, feature order, selected C checkpoints and tolerance remain locked. Inputs are existing D-resident raw windows. No new fitting.
4. Split the former joint compiled stage only for diagnosis: feature operations/scaling, candidate-leaf/enclosure update, softmax interval, output/certificate, explicit control and exact fallback. State setup and signed residual are separately reported. Extra Python/Numba dispatches, clock calls and bookkeeping make this an instrumented diagnostic; its complete times never replace the formal endpoint. All selected outputs/block choices/fallback/certificates must match frozen outputs within2e-12 before profile rows are admitted.

Owned outputs: `scripts/review2_latency_analysis.py`, `results/review2_latency_analysis/*`, this report. Original formal outputs, raw data, models, frozen code and manuscript remain read-only. The initial D space check found about133.12GB free. Scientific figures are assigned to root; machine-readable CSV/JSON and plot specifications are provided here.

Prospective status at writing: neither analysis nor profile had been executed. The execution record below was appended after both completed successfully; the plan above was not selected from the diagnostic outcomes.

## Execution and denominators

Completed all three phases: saved-output analysis, repeat-median sensitivity, and separately instrumented component profile. No model fitting, feature-order selection, tolerance change, formal-time replacement or source-file edit occurred. The original source hashes and frozen-lock check are in the output JSON. Numerical conformance is PASS for the profile: all 108 system-window checks had zero probability/certificate difference and identical stopping blocks and fallback decisions. There were 648 stored instrumented calls (36 windows × three systems × three repetitions × two methods), with four windows from each of the nine held-out subjects. All runtime thread pools reported one thread.

The main saved-output analysis uses the SAME900 windows, not all 2,947 official test windows. There are 2,700 system-window outputs and 8,100 paired timing calls. The systems share the same nine people; the three models, three repetitions, and 27 system-subject combinations do not increase the independent subject count. Group means first average three repeats within a system-window. P99 tails instead select individual calls within each system (2,700 calls per method/system). No new uncertainty estimate is added here; the original shared-subject bootstrap remains the inferential analysis.

## Processing outcome and paired complete time

The all-record pooled means reproduce the formal result: partial 1.001322 ms, compiled full 1.080011 ms, a 7.2859% mean reduction. The signed paired difference is −0.078689 ms. Partial processing is slower for 39.44% of system-window means; average superiority does not mean every input benefits. Fallback occurs for 679/2,700 outputs (25.15%).

| Processing outcome | System-windows | Partial mean ms | Full mean ms | Partial − full ms | Partial slower % |
|---|---:|---:|---:|---:|---:|
| Stop after 12 blocks | 1,024 | 0.781742 | 1.077781 | −0.296039 | 4.10 |
| Stop after 16 blocks | 997 | 1.054817 | 1.072119 | −0.017302 | 45.34 |
| 18 blocks / exact fallback | 679 | 1.253923 | 1.094962 | +0.158962 | 84.09 |
| All early stops (12 or 16) | 2,021 | 0.916456 | 1.074988 | −0.158532 | 24.44 |

These are descriptive, outcome-defined strata. They are compatible with a processing tradeoff: stopping at 12 blocks leaves a larger average opportunity for saved feature work, whereas 16-block stopping is near break-even and full fallback accompanies additional enclosure/certificate work. They do not estimate the causal effect of forcing a particular block count or fallback on the same inputs. The full-case reference times differ between strata, and execution jitter also contributes.

The machine-readable results include all seed-specific and subject-specific strata. Pooled subject results use 100 windows × three systems each:

| Subject | Fallback % | Partial mean ms | Full mean ms | Paired difference ms |
|---|---:|---:|---:|---:|
| 2 | 22.33 | 0.990770 | 1.098791 | −0.108021 |
| 4 | 26.33 | 1.028005 | 1.070983 | −0.042978 |
| 9 | 47.00 | 1.071601 | 1.102460 | −0.030860 |
| 10 | 29.67 | 0.946020 | 1.032518 | −0.086498 |
| 12 | 30.00 | 0.971428 | 1.028421 | −0.056993 |
| 13 | 15.33 | 0.976344 | 1.086790 | −0.110446 |
| 18 | 16.67 | 0.993012 | 1.094552 | −0.101540 |
| 20 | 27.33 | 1.029516 | 1.058636 | −0.029120 |
| 24 | 11.67 | 1.005206 | 1.146948 | −0.141741 |

## Tail membership and tradeoff

Each partial P99 tail contains 27 calls; the corresponding full time is paired on the same window/repetition. Baseline fallback proportions use all 900 outputs for that system. The paired differences below are conditional on membership in the partial latency tail and are diagnostic, not independent comparative endpoints.

| System seed | Partial P99 ms | Full P99 ms | Baseline fallback % | Partial-tail fallback calls / 27 | Partial-tail fallback % | Mean paired difference in partial tail ms |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 2.341802 | 2.320208 | 19.89 | 18 | 66.67 | +1.152974 |
| 29 | 2.187748 | 2.097817 | 38.22 | 25 | 92.59 | +1.188656 |
| 43 | 2.276310 | 2.225132 | 17.33 | 12 | 44.44 | +1.275893 |

Fallback is enriched in all three partial tails, but does not account for all tail observations. In seed 43, the largest partial call is 5.9052 ms at 16 blocks without fallback (case 2553, subject 20, repetition 2); its paired full call is 1.5595 ms. Conversely, selecting the full method's P99 tail yields mean paired differences of −1.429230, −3.377648 and −1.089785 ms for seeds 17, 29 and 43. This change with the tail anchor demonstrates why tail-conditioned comparisons must not be treated as an algorithm effect. The actual complete-time P99 is reported, and no hard-real-time guarantee follows from these measurements.

## Isolated long full call and posthoc summary sensitivity

The immutable seed-29 full record includes a 72.1447 ms call at case 119, subject 2, repetition 2. The paired partial call is 0.9839 ms and uses all 18 blocks with fallback. Its signed difference is −71.1608 ms. The record is retained, and its cause is not identified by these data.

A separately labelled posthoc sensitivity retains every window and repetition but summarizes each system-window with the median of its three repetitions, then averages those medians. This is a different timing estimand, not a replacement of the original mean endpoint:

| System | Mean of repeat medians: partial ms | Full ms | Reduction % |
|---|---:|---:|---:|
| 17 | 0.979057 | 1.083023 | 9.60 |
| 29 | 0.989117 | 0.987465 | −0.17 |
| 43 | 0.929410 | 1.026756 | 9.48 |
| Equal-system pooled | 0.965861 | 1.032415 | 6.45 |

The original 7.29% pooled mean saving is not solely created by that call, but seed 29's positive original mean saving is sensitive to isolated timing variability. A claim of stable improvement in every frozen system would therefore be too strong. No observation is removed, relabelled, or silently replaced.

## Existing A/C checkpoint comparison

Both checkpoint policies were frozen before test and were already timed in the same randomized formal method order. A uses checkpoints 4/8/12/16/18; C uses 12/16/18. This is a secondary existing comparison, not a new post-test schedule selection.

| System | A mean ms | C mean ms | C − A ms | A mean blocks | C mean blocks |
|---|---:|---:|---:|---:|---:|
| 17 | 1.103620 | 1.016754 | −0.086866 | 13.2778 | 14.5089 |
| 29 | 1.196835 | 1.023604 | −0.173232 | 15.7778 | 15.7778 |
| 43 | 1.026521 | 0.963610 | −0.062912 | 13.2578 | 14.6711 |
| Equal-system pooled | 1.108992 | 1.001322 | −0.107670 | 14.1044 | 14.9859 |

C is faster in 70.22% of system-window means even though it processes more blocks overall. This supports the practical importance of checking overhead, while remaining specific to these frozen schedules, models and measurements.

## Independent component profile

The outcome-free sample contains 36 windows, four per subject, shared by all systems. The profile's fallback proportion is 28.70% and mean block count is 15.3889, differing from the formal900 cohort. Component times below are means over 324 instrumented calls per method. They do not replace the original full or partial timing means and do not constitute a causal module-removal experiment.

| Measured component | Instrumented partial ms | Instrumented compiled full ms |
|---|---:|---:|
| State setup | 0.003063 | 0.001543 |
| Selected feature operations and scaling | 0.667747 | 0.766584 |
| Leaf-mask update and score enclosure | 0.100787 | 0 |
| Probability interval | 0.006876 | 0 |
| Output and minimax certificate | 0.013194 | 0 |
| Explicit control | 0.001906 | 0.000908 |
| Exact prediction when required | 0.015987 | 0.055875 |
| Signed unassigned instrumentation residual | 0.004886 | 0.001785 |
| Complete instrumented span | 0.814446 | 0.826696 |

The profile includes zero exact-prediction time for partial calls that do not invoke it; the partial average therefore mixes fallback and non-fallback calls. The full exact-prediction field always represents the complete frozen compiled predictor. Feature processing accounts for 81.99% of partial instrumented time, leaf/enclosure work 12.37%, and probability interval plus output/certificate 2.46%. The enclosure update is the largest measured non-feature component. At this diagnostic sample and time of measurement, seed-specific complete partial/full means are 0.768460/0.825335, 0.890785/0.808661 and 0.784094/0.846091 ms, respectively; seed 29 is again slower. These small-sample fresh-period means are not used to revise the formal mean reduction.

The profile splits the original joint compiled stage into additional Python/Numba calls and clock intervals. Empty counter spans over 1,000 readings have mean 53 ns and empirical median/P99 100 ns; they are not subtracted from measured costs. The signed residual is the observed complete span minus the nonoverlapping recorded component spans. A saved-CSV verification gives a maximum absolute reconstruction discrepancy of 2.23e−16 ms, caused by floating-point representation. The residual is not an invented allocation of unexplained time and does not measure only clock overhead. It includes unassigned loop, dispatch and bookkeeping intervals. The measurements support an engineering cost decomposition; they do not establish energy savings, deadline compliance, or the causal contribution of any one module.

## Reusable data and figure specification

Saved-output folder: `results/review2_latency_analysis/saved_output_v1`. The paired-case CSV stores one row per system/window with means and medians of all three repeats; paired-call CSV preserves all 8,100 individual matched calls. JSON contains all system, subject, system-subject, block and fallback strata and both P99-tail anchors. `repeat_median_sensitivity.json` records the different-estimand sensitivity and the largest observed calls. `checkpoint_A_C_case_data.csv` preserves the existing schedule comparison. `empirical_cdf_quantiles.csv` gives 201 quantiles from 0 to 1 for each system and method.

Component folder: `results/review2_latency_analysis/component_profile_v1`. The prospective configuration records selection before model/raw profiling; `profile_conformance.json` records output checks, `component_profile_case_results.csv` stores all 648 measured calls, and `component_profile.json` provides pooled/system means, thread settings, clock spans and limitations. Both folders include `plot_specifications.json` for later scientific figure generation. Suggested figures are a complete-time CDF with P95/P99, paired difference by processed-block stratum, subject-level fallback versus paired difference (explicitly descriptive), existing A/C comparison, and a component stack that includes the measured residual.

Status: COMPLETE. All original scientific settings and formal results remain unchanged. The actionable interpretation is to report average savings together with fallback/near-break-even strata, the slight P99 deterioration and seed-specific timing sensitivity, rather than implying uniform per-input or hard-real-time acceleration.
