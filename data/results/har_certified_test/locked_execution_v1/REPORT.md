# Frozen HAR test execution report

Executed successfully under the root pre-test lock, with unchanged models, temperatures, feature orders and schedule C=[12,16,18]. Three seeds, all 2947 official test windows, and the same outcome-free 900-window cohort for 21 specifications and three actual-raw repetitions. Total script wall time 303.643 s. No fitting or post-test selection.

Main endpoint: tolerance 0.025, complete mean online processing time relative to the faster contemporary native/compiled full comparator (compiled for every seed). Software time begins with a released prefiltered/windowed raw input in memory. Acquisition, prefiltering/windowing, data load, cache construction, JIT, reference checks and serialization are excluded; feature extraction/scaling, bounds, output/certificate and exact fallback are included.

| Seed | Tight time ms | Compiled full ms | Time reduction | Blocks of18 | Full fallback | Cohort risk / full risk |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 1.016754 | 1.122237 | 9.40% | 14.5089 | 19.89% | 0.066525 / 0.063736 |
| 29 | 1.023604 | 1.051749 | 2.68% | 15.7778 | 38.22% | 0.090309 / 0.083996 |
| 43 | 0.963610 | 1.066047 | 9.61% | 14.6711 | 17.33% | 0.054615 / 0.051502 |

Equal-seed mean complete time is 1.001322 versus 1.080011 ms, a 7.286% reduction. On that same900 cohort, half-Brier risk is 0.07048308 versus 0.06641128; observed-label mean excess is 0.00407180. Full2947 outcomes are separate: risk 0.06561069 versus 0.06136345, mean excess 0.00424724. Accuracy of primary and full reference is identical for each frozen system, with mean 91.9014%. These are three systems on the same nine held-out subjects, not8841 independent observations.

Every-label maximum excess across all three systems and all2947 cases is 0.02486915; primary tolerance violations are zero. All17 canonical specifications passed their numerical certificate assertions. The first24 raw/cache paths agreed exactly in output, selected block count and fallback; all score/probability containment checks passed. Native/compiled probability difference is zero for seeds17/43 and 7.77e-16 for seed29. This is numerical verification, not a formal all-input floating-point proof.

P99 is slightly worse than compiled full in every seed. Seed29 P95 is also worse. Mean savings do not establish tail dominance or a hard-real-time deadline. Root shared-subject bootstrap supplies uncertainty; no independent-window confidence interval is computed here.

All controls and failures retained:

- coordinate_minimax: equal-seed mean 1.001969 ms (7.226% reduction against full).
- tight_midpoint: equal-seed mean 1.137513 ms (-5.324% reduction against full).
- uniform_A_tight_minimax: equal-seed mean 1.108992 ms (-2.683% reduction against full).
- Tight versus coordinate relaxation has a very small operational increment here; the data do not justify a large practical superiority claim.
- Score-separated class preservation is faster and preserves full argmax, but does not preserve probability loss: 340/550/399 of the900 cohort cases, and1159/1835/1277 of all2947 cases, exceed the primary0.025 every-label increment.
- Fixed-prefix and entropy controls are faster, with prefix12 and entropy threshold1.0 chosen before test; their outputs are identical, while entropy pays an additional stopping check. Their cohort every-label exceedances are176/237/228 and whole-test exceedances528/729/715. Maximum every-label excess reaches0.51793; these are empirically tuned average-risk controls, not certificate failures.

| Delta | Tight time ms | Reduction vs full | Same900 risk | Reference excess | Blocks | Full fallback |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.262918 | -16.94% | 0.066411 | 0.000000 | 18.0000 | 100.00% |
| 0.01 | 1.065939 | 1.30% | 0.067638 | 0.001226 | 15.8148 | 37.04% |
| 0.025 | 1.001322 | 7.29% | 0.070483 | 0.004072 | 14.9859 | 25.15% |
| 0.05 | 0.949258 | 12.11% | 0.074526 | 0.008115 | 14.2696 | 14.37% |
| 0.1 | 0.886820 | 17.89% | 0.082493 | 0.016082 | 13.4844 | 3.04% |

Artifacts: canonical all_test_outcomes.npz and timing_cohort_outcomes.npz per seed; separate auxiliary schedule and stopping-control arrays; complete actual-raw JSONL rows; standalone profiles, conformance and executed summaries. `primary_readout.json` is a lightweight derivation from saved JSON only, with no new predictions/timing. The old negative uniform-schedule development results, technical baseline-alias failure and checkpoint development alternatives remain retained.
