# Hydraulic deterministic processing development prototype report

Date: 2026-10-02. All recorded runs are development implementation diagnostics; none is independent confirmation. No new classifier was fitted. The earlier compute-value pilot remains STOP_NO_EXPANSION.

## Data and online boundary

The frozen 150-iteration, four-class HGB was trained on the existing 907 predictor-fitting rows. Training-only tree split gain divided by prior processing costs fixed the 28-block order. The 223 internal-validation rows support feasibility and numerical checks. Gate-fitting, calibration, audit, the old 435-case test and HAR outcomes were not evaluated or used by these runs. The old D cache contains a broader development pool; each source array is indexed to the two authorized partitions before processing. The original raw archive is unchanged.

There are 14 physical signal channels, statistics and FFT summaries per channel, 28 blocks and 238 standardized features. All 28 blocks appear in at least one split of the frozen forest. The forest has 600 trees and 11,724 nodes; leaves already include learning-rate shrinkage. Initial input is empty; order is fixed from training-model gain/cost. The complete timer starts with the released raw windows resident in host memory, includes allocations, actual selected operators, scaling, tree bounds, certificate, control and output, and excludes acquisition, transport, disk I/O, offline training/JIT and outside-timer reference/label checks.

## Certificate and checks

Observed numerical splits choose their correct child; unknown splits union both child subtrees. Score and probability intervals are conservative float64 engineering enclosures. For each label, bounded-simplex distance uses a Lagrangian dual lower value. The actual emitted vector is certified with B(p)=.5||p||²+max_y(.5−dLB_y−p_y), with conservative upward allowance. Coordinate minimax substitutes the relaxed a_y=u_y−.5u_y²−.5Σ_{j≠y}l_j²; tight midpoint uses the same tight certificate at normalized interval midpoint. Exact reachable-score collapse uses native prediction and analytic zero excess.

The exact-arithmetic theorem is independent of empirical checks. The numerical implementation is not a formally verified all-input machine-arithmetic guarantee. Outward guards and a safe [0,1] interval fallback are explicitly documented.

V1 correctness: 696 fixed-order prefixes plus random-order prefixes on 12 cases; known-all compiled/native maximum probability difference 0. Signed maximum logit interval violation -2.89e-12, probability interval violation -2.01e-13, every-label certificate violation 1.11e-16. The latter is a round-off-scale 1.11e−16, below the stated numerical comparison tolerance, not evidence of bitwise zero error.

All emitted outputs on all 223 validation cases and all variants/tolerances passed the per-label numerical check at tolerance 2e−12. Labels and full probabilities never enter the online controller.

## Complete-time outcomes

Three repetitions per case used fixed-seed randomized method order. Means below include complete software work. The comparison is against the faster of optimized native sklearn full processing and the same-backend compiled exact full path traversal. Both full controls compute actual model-used blocks and share the same raw operators. All fixed tolerances are retained.

### V1: check after every block

| Method | Delta | Mean ms | P95 ms | P99 ms | Blocks | Full fallback | Half-Brier | Accuracy | Time reduction vs faster full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tight_minimax | 0.000 | 18.604118 | 26.992640 | 35.233444 | 28.000 | 100.00% | 0.106148 | 86.55% | -174.01% |
| tight_minimax | 0.010 | 8.889675 | 18.333820 | 23.713376 | 20.233 | 9.42% | 0.105900 | 86.55% | -30.93% |
| tight_minimax | 0.025 | 7.839913 | 15.691920 | 21.558528 | 18.686 | 4.04% | 0.109718 | 86.55% | -15.47% |
| tight_minimax | 0.050 | 7.033908 | 13.068120 | 18.339232 | 17.242 | 0.90% | 0.120051 | 87.00% | -3.60% |
| tight_minimax | 0.100 | 6.313571 | 11.782320 | 17.423876 | 15.404 | 0.00% | 0.137432 | 87.00% | 7.01% |
| coordinate_minimax | 0.000 | 18.143324 | 26.486880 | 34.162924 | 28.000 | 100.00% | 0.106148 | 86.55% | -167.22% |
| coordinate_minimax | 0.010 | 8.566504 | 17.329460 | 23.599472 | 20.242 | 9.87% | 0.105999 | 86.55% | -26.17% |
| coordinate_minimax | 0.025 | 7.812270 | 15.476480 | 20.724580 | 18.713 | 4.04% | 0.109452 | 86.55% | -15.06% |
| coordinate_minimax | 0.050 | 6.942148 | 12.505040 | 17.130924 | 17.291 | 0.90% | 0.120649 | 87.00% | -2.25% |
| coordinate_minimax | 0.100 | 6.370486 | 12.380360 | 16.233832 | 15.466 | 0.00% | 0.137588 | 87.44% | 6.17% |
| tight_midpoint | 0.000 | 18.235666 | 25.673760 | 33.513392 | 28.000 | 100.00% | 0.106148 | 86.55% | -168.58% |
| tight_midpoint | 0.010 | 10.852165 | 20.172840 | 24.876748 | 24.251 | 20.18% | 0.106095 | 86.55% | -59.83% |
| tight_midpoint | 0.025 | 9.657946 | 18.141060 | 23.501472 | 22.807 | 8.52% | 0.105866 | 86.55% | -42.25% |
| tight_midpoint | 0.050 | 8.641327 | 15.754760 | 20.405456 | 21.229 | 4.93% | 0.105403 | 87.00% | -27.27% |
| tight_midpoint | 0.100 | 7.724831 | 14.017180 | 20.034064 | 19.265 | 0.00% | 0.102588 | 87.44% | -13.77% |
| native_full | full | 14.171999 | 19.671320 | 32.621412 | 28.000 | 100.00% | 0.106148 | 86.55% | -108.73% |
| compiled_full | full | 6.789610 | 10.610380 | 15.649532 | 28.000 | 100.00% | 0.106148 | 86.55% | 0.00% |

Execution wall time: 137.181 s. No silent rerun; all original per-case records and the V1 source snapshots are retained.

### V2: recorded fixed-four-block repair

| Method | Delta | Mean ms | P95 ms | P99 ms | Blocks | Full fallback | Half-Brier | Accuracy | Time reduction vs faster full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tight_minimax | 0.000 | 8.672988 | 12.009780 | 13.966024 | 28.000 | 100.00% | 0.106148 | 86.55% | -148.97% |
| tight_minimax | 0.010 | 4.407461 | 10.345460 | 12.428852 | 21.794 | 26.91% | 0.104909 | 86.55% | -26.52% |
| tight_minimax | 0.025 | 3.893236 | 9.967820 | 12.960464 | 20.287 | 20.63% | 0.107168 | 86.55% | -11.76% |
| tight_minimax | 0.050 | 3.351020 | 8.819960 | 12.289184 | 18.798 | 12.56% | 0.114163 | 86.55% | 3.80% |
| tight_minimax | 0.100 | 2.724081 | 5.357480 | 10.609648 | 17.130 | 4.04% | 0.124166 | 87.00% | 21.80% |
| coordinate_minimax | 0.000 | 8.611344 | 11.859340 | 13.515724 | 28.000 | 100.00% | 0.106148 | 86.55% | -147.20% |
| coordinate_minimax | 0.010 | 4.396340 | 10.550800 | 12.776048 | 21.812 | 26.91% | 0.105171 | 86.55% | -26.20% |
| coordinate_minimax | 0.025 | 3.890291 | 9.704460 | 12.750700 | 20.305 | 20.63% | 0.107104 | 86.55% | -11.68% |
| coordinate_minimax | 0.050 | 3.332632 | 8.898280 | 12.332252 | 18.834 | 12.56% | 0.113971 | 86.55% | 4.33% |
| coordinate_minimax | 0.100 | 2.725954 | 5.748200 | 9.480852 | 17.184 | 4.04% | 0.123562 | 87.00% | 21.75% |
| tight_midpoint | 0.000 | 8.634970 | 12.112160 | 14.328840 | 28.000 | 100.00% | 0.106148 | 86.55% | -147.88% |
| tight_midpoint | 0.010 | 5.998658 | 11.175380 | 13.148120 | 25.274 | 51.12% | 0.106069 | 86.55% | -72.20% |
| tight_midpoint | 0.025 | 5.328065 | 11.024460 | 12.983124 | 24.251 | 39.01% | 0.105968 | 86.55% | -52.95% |
| tight_midpoint | 0.050 | 4.527267 | 10.208600 | 12.792084 | 22.780 | 25.56% | 0.105737 | 86.55% | -29.96% |
| tight_midpoint | 0.100 | 3.638042 | 9.125300 | 11.746460 | 20.843 | 11.66% | 0.103981 | 87.00% | -4.44% |
| native_full | full | 7.985928 | 11.094780 | 13.659088 | 28.000 | 100.00% | 0.106148 | 86.55% | -129.25% |
| compiled_full | full | 3.483521 | 5.166680 | 5.825684 | 28.000 | 100.00% | 0.106148 | 86.55% | 0.00% |

Execution wall time: 68.111 s. No silent rerun; all original per-case records and the V1 source snapshots are retained.

### V3: same-backend exact fallback, corrected implementation

| Method | Delta | Mean ms | P95 ms | P99 ms | Blocks | Full fallback | Half-Brier | Accuracy | Time reduction vs faster full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tight_minimax | 0.000 | 4.660971 | 6.729840 | 7.698188 | 28.000 | 100.00% | 0.106148 | 86.55% | -16.85% |
| tight_minimax | 0.010 | 3.545700 | 5.856580 | 7.206604 | 21.794 | 26.91% | 0.104909 | 86.55% | 11.11% |
| tight_minimax | 0.025 | 3.314271 | 5.608880 | 6.534808 | 20.287 | 20.63% | 0.107168 | 86.55% | 16.91% |
| tight_minimax | 0.050 | 3.068834 | 5.417920 | 6.641104 | 18.798 | 12.56% | 0.114163 | 86.55% | 23.07% |
| tight_minimax | 0.100 | 2.864293 | 5.216980 | 6.713676 | 17.130 | 4.04% | 0.124166 | 87.00% | 28.19% |
| coordinate_minimax | 0.000 | 4.633928 | 6.947580 | 8.642456 | 28.000 | 100.00% | 0.106148 | 86.55% | -16.17% |
| coordinate_minimax | 0.010 | 3.569358 | 6.097820 | 7.286944 | 21.812 | 26.91% | 0.105171 | 86.55% | 10.52% |
| coordinate_minimax | 0.025 | 3.307863 | 5.667460 | 6.506672 | 20.305 | 20.63% | 0.107104 | 86.55% | 17.07% |
| coordinate_minimax | 0.050 | 3.034753 | 5.430500 | 6.388984 | 18.834 | 12.56% | 0.113971 | 86.55% | 23.92% |
| coordinate_minimax | 0.100 | 2.833179 | 5.241660 | 6.283296 | 17.184 | 4.04% | 0.123562 | 87.00% | 28.97% |
| tight_midpoint | 0.000 | 4.616240 | 6.538420 | 8.064088 | 28.000 | 100.00% | 0.106148 | 86.55% | -15.73% |
| tight_midpoint | 0.010 | 4.175812 | 6.290860 | 7.441908 | 25.274 | 51.12% | 0.106069 | 86.55% | -4.68% |
| tight_midpoint | 0.025 | 3.966055 | 6.202240 | 7.335136 | 24.251 | 39.01% | 0.105968 | 86.55% | 0.57% |
| tight_midpoint | 0.050 | 3.783113 | 5.840420 | 7.167828 | 22.780 | 25.56% | 0.105737 | 86.55% | 5.16% |
| tight_midpoint | 0.100 | 3.427351 | 5.627660 | 6.993012 | 20.843 | 11.66% | 0.103981 | 87.00% | 14.08% |
| native_full | full | 8.991422 | 11.866380 | 13.040852 | 28.000 | 100.00% | 0.106148 | 86.55% | -125.41% |
| compiled_full | full | 3.988947 | 5.719760 | 6.796548 | 28.000 | 100.00% | 0.106148 | 86.55% | 0.00% |

Execution wall time: 50.068 s. No silent rerun; all original per-case records and the V1 source snapshots are retained.

## Interpretation and next action

At the predetermined .025 tolerance, V1 and V2 remain slower than compiled exact full processing. V2 tight minimax is 3.893236 ms versus 3.483521 ms, an 11.76% increase, despite computing 20.287 of 28 blocks. At .1, V2 tight minimax is 2.724081 ms, a 21.80% reduction, with 17.130 blocks. The .1 result does not replace the locked .025 goal. The coordinate relaxation is nearly identical in block count and complete time: at .025, 20.305 blocks and 3.890291 ms. Therefore Hydraulic does not yet establish a practically meaningful independent tight-simplex increment over its cheaper coordinate control.

V2 computes blocks in fixed groups of four and uses their union affected-ancestor list in one compiled bounds/certificate call. This change reduces repeated maintenance but can miss a valid exit until the next checkpoint. Native full fallback occurs in 20.63% of .025 cases and adds sklearn single-case interface overhead; this differs from the compiled full comparator and is a candidate implementation bottleneck. A later same-backend fallback or bitmask implementation must have a separate prospective record, preserve both runs, and again compare complete times within the same run. It is not an algorithm/model/tolerance selection claim.

The separately authorized V3 keeps V2 checkpoints/order/tolerances and uses the compiled exact evaluator for full/collapsed fallback. The reference is explicitly the same-backend frozen full pipeline, with native sklearn retained as a semantic conformance comparator. All 223 native/reference probabilities matched exactly in these checks. V3 .025 tight minimax takes 3.314271 ms versus compiled full 3.988947 ms: 16.91% complete-time reduction, with P95 5.60888 versus 5.71976 ms and P99 6.534808 versus 6.796548 ms. Its mean half-Brier is .107168 versus .106148 for full processing and accuracy remains 86.55%. It computes 20.287 blocks on average and has 20.63% full fallback.

V3 demonstrates development net-cost feasibility at .025 against the faster full implementation, but does not establish a substantial tight-simplex advantage over coordinate minimax: the coordinate control takes 3.307863 ms and computes 20.305 blocks. Tight midpoint takes 3.966055 ms, with almost no net saving. The all-delta tight result is retained: .0 is 16.85% slower; .01 saves 11.11%; .025 saves 16.91%; .05 saves 23.07%; .1 saves 28.19%. No independent-validation or submission-readiness claim follows from development evidence.

V3 first technical attempt was interrupted after progress50/223 because its batch4 call site still used the native fallback despite the new parameter. Initial configuration, first24 results and failed source snapshot are preserved, with results/compiled_fallback_repair/technical_failure.json. The corrected call-site implementation was executed once in a fresh directory, results/compiled_fallback_repair_corrected, and completed in 50.068 s. This repair was disclosed immediately and is not a silent favorable rerun.

Absolute timing varied between runs: each run retained randomized contemporaneous full controls. Cross-run absolute differences cannot be attributed solely to the implementation repair. Each primary within-run comparison uses its own faster full control. The reported P95/P99 are empirical diagnostic quantiles, not hardware deadline guarantees.

## Fitting-only detailed profile

Eight fitting cases were profiled separately, with warm summaries excluding the first case. No outcomes were consumed. These instrumented values are diagnostic and do not replace uninstrumented complete timers.

| Component | Mean per block ms | Full-28 accumulated mean ms |
|---|---:|---:|
| feature_ms | 0.176853 | 4.951871 |
| affected_node_update_ms | 0.025920 | 0.725757 |
| score_interval_ms | 0.012835 | 0.359371 |
| probability_interval_ms | 0.004051 | 0.113429 |
| tight_certificate_ms | 0.006718 | 0.188100 |

Decision at this development stage: V3 establishes limited net-cost feasibility against optimized full processing; the contribution remains CONDITIONAL because the strong coordinate certificate is practically tied and independent subject-held-out evidence is absent. Preserve negative controls and the old pilot. The root decides whether train-only external data preparation or another recorded kernel optimization is now justified; no test access is authorized by this report.

Records: results/prototype_execution.json; results/prototype_case_results.jsonl; results/prototype_separate_profiles.json; results/batch4_repair/prototype_execution.json; results/batch4_repair/prototype_case_results.jsonl; results/batch4_repair/fitting_only_detailed_components.json; results/compiled_fallback_repair_corrected/prototype_execution.json. Plans: research/PROTOTYPE_PLAN.md, research/BATCH4_REPAIR_PLAN.md and research/COMPILED_FALLBACK_REPAIR_PLAN.md.
