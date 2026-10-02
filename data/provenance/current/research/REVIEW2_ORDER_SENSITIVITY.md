# Development-only feature-order sensitivity

Prospective plan recorded on 2 October 2026 before executing this diagnostic. The reviewer asks whether pruning/stopping depends on feature order. This post-test revision diagnostic uses only the original four development subjects for each of the three frozen HAR HGB systems and the existing official-training feature cache. It never opens the official test cache, recomputes raw blocks, refits a model/scaler, retimes the processing pipeline, or chooses a replacement order.

Frozen elements are the selected HGB models, temperatures, fit-only scalers, tight minimax output/certificate, delta .025 and C checkpoints 12/16/18. The four orders are: the locked training gain/measured-time order, decreasing training split gain with block ID breaking ties, reverse locked order, and one deterministic permutation of the locked order with RNG seed9302026+system seed. Every candidate and system is retained. All development windows are included, with both pooled-window and subject-level descriptive summaries.

The primary diagnostic summaries are processed blocks, fallback proportion, realized half-Brier excess against the same frozen full predictor, maximum every-label excess and certificate compliance. The initial12-block outer probability box is additionally summarized by the sum of coordinate widths and sqrt(min(2,sum(width squared))). The latter is an upper bound on the Euclidean diameter of the interval-simplex outer set; neither quantity is its exact diameter or a measured tree-correlation gap. The initial12-block feature dimension count is recorded. Cached feature access supports prediction/processing-count diagnosis only, not software latency or energy claims.

Implementation reuses the frozen calibrated forest, leaf-mask kernel and selected-coordinate observer but omits runtime timers from the diagnostic execution function. A small same-case check compares this function with the existing pipeline for identical probabilities, certificates, stopping blocks and fallback. All orders are saved before their development outcomes. No new statistical test treats the systems or windows as independent participants.

Owned files are scripts/review2_order_sensitivity.py, results/review2_order_sensitivity/* and this report. Old sources and root manuscript are read-only. Status at prospective writing: not executed. Execution results will be appended without changing this plan.

## Executed results

The single execution completed successfully (exit 0). No official test or raw-window array was loaded, and no fitting or latency benchmark was performed. The outcome function has no timer; the existing API used only for the small semantic checks internally returns a time field, which is discarded and neither saved nor analyzed. The result's latency-measurement flag refers to the absence of a latency benchmark/estimate. The code explicitly reads only cached features, labels, subjects, development indices and existing scaler arrays. There are 1,454, 1,507 and 1,389 development windows in systems 17, 29 and 43, respectively: 4,350 system-window outputs per order and 17,400 case/order rows across all four orders. These are reused development sets, not new held-out confirmation. The development subjects are 17/19/28/29 (system17), 16/27/28/30 (system29), and 1/6/7/25 (system43); subject28 is shared across two splits. Twelve system-subject combinations correspond to eleven unique people, not twelve independent samples. Model selection had previously used these development sets.

All 96 small semantic checks (eight cases × four orders × three systems) match the existing pipeline with zero probability difference and identical certificates, block choices and fallback decisions. Native/compiled full probability differences remain below2e−12. Each visited prefix was checked against the frozen full scores/probabilities and every possible label. No interval or certificate violation exceeds the numerical tolerance; the maximum signed certificate violation is zero, attained by exact fallback. All terminal certificates and maximum every-label excesses are at most .025.

| System | Order | Mean blocks | Fallback % | Mean realized half-Brier excess | Maximum every-label excess |
|---|---|---:|---:|---:|---:|
| 17 | Locked gain/time | 13.8762 | 12.93 | 0.000949 | 0.024880 |
| 17 | Gain only | 13.8377 | 12.93 | 0.000770 | 0.024816 |
| 17 | Reverse locked | 18.0000 | 100.00 | 0 | 0 |
| 17 | Fixed random | 18.0000 | 100.00 | 0 | 0 |
| 29 | Locked gain/time | 15.5143 | 24.42 | 0.004739 | 0.024335 |
| 29 | Gain only | 15.2900 | 13.21 | 0.005640 | 0.024622 |
| 29 | Reverse locked | 18.0000 | 100.00 | 0 | 0 |
| 29 | Fixed random | 16.5322 | 26.61 | 0.002093 | 0.023783 |
| 43 | Locked gain/time | 14.7631 | 17.64 | 0.002925 | 0.024781 |
| 43 | Gain only | 14.6105 | 17.64 | 0.003299 | 0.024779 |
| 43 | Reverse locked | 18.0000 | 100.00 | 0 | 0 |
| 43 | Fixed random | 17.0050 | 79.91 | 0.001961 | 0.024402 |

The equal-system means are:

| Order | Mean blocks | Fallback % | Mean realized excess | Maximum every-label excess |
|---|---:|---:|---:|---:|
| Locked gain/time | 14.7179 | 18.33 | 0.002871 | 0.024880 |
| Gain only | 14.5794 | 14.59 | 0.003236 | 0.024816 |
| Reverse locked | 18.0000 | 100.00 | 0 | 0 |
| Fixed random | 17.1791 | 68.84 | 0.001351 | 0.024402 |

These results show material dependence of early stopping on the processing order. Reversal eliminates early stopping for every development input in every system. One fixed random ordering is also substantially less effective, but differs across systems. It is one deterministic permutation per system, not an average over many random orders. Gain-only ordering has modestly lower block counts than the cost-aware locked order; that observation does not imply lower runtime because the identities and costs of processed blocks differ. It also has a larger pooled realized loss excess, although both remain within the same bound. No replacement order is chosen, and the formal test policy/result remains unchanged. Zero excess for reversed order follows from always returning the exact full predictor, rather than superior partial prediction.

## Development subject summaries

Complete 48 order/system/subject rows are in subject_summaries.json. For the locked order:

| System | Subject | Windows | Mean blocks | Fallback % | Mean realized excess | Maximum every-label excess |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 17 | 368 | 13.9837 | 11.68 | 0.003404 | 0.024880 |
| 17 | 19 | 360 | 13.9556 | 20.56 | 0.001861 | 0.024867 |
| 17 | 28 | 382 | 13.5812 | 3.66 | −0.005570 | 0.024582 |
| 17 | 29 | 344 | 14.0058 | 16.57 | 0.004606 | 0.023374 |
| 29 | 16 | 366 | 15.8689 | 21.31 | −0.001453 | 0.024212 |
| 29 | 27 | 376 | 14.7394 | 13.03 | 0.010281 | 0.024335 |
| 29 | 28 | 382 | 15.6230 | 19.90 | 0.002766 | 0.024273 |
| 29 | 30 | 383 | 15.8277 | 43.08 | 0.007184 | 0.023701 |
| 43 | 1 | 347 | 14.4899 | 4.03 | 0.004494 | 0.024781 |
| 43 | 6 | 325 | 14.2954 | 15.69 | 0.002002 | 0.024130 |
| 43 | 7 | 308 | 15.0260 | 24.03 | 0.002493 | 0.023981 |
| 43 | 25 | 409 | 15.1687 | 25.92 | 0.002651 | 0.023979 |

Equal weighting of the twelve system-subject combinations, as a descriptive aggregation rather than an independent-person analysis, gives blocks/fallback of14.7138/18.29%,14.5784/14.56%,18/100%,17.1758/68.71% for locked, gain-only, reverse and random. The corresponding mean realized excesses are .002893, .003288,0,.001351. This separate aggregation is saved in subject_balanced_aggregation.json; it is not a hypothesis test or new primary endpoint. Negative realized excess can occur because the partial probability happens to improve the realized label's loss relative to the fixed full model; the guarantee is the one-sided upper bound for every possible label.

## Initial12-block outer-set proxies

| Order | Equal-system mean sum of probability widths | Mean outer-set diameter upper proxy |
|---|---:|---:|
| Locked gain/time | 0.623226 | 0.340698 |
| Gain only | 0.594592 | 0.326628 |
| Reverse locked | 5.670742 | 1.414214 |
| Fixed random | 3.468061 | 1.246648 |

These are summaries of the first checkpoint's outer intervals. The exact values of probability coordinates are not assumed independently attainable. The width-squared box bound and the universal simplex diameter bound are combined as sqrt(min(2,sum(width squared))). This is a valid geometric upper proxy in exact arithmetic, not a measure of tightness relative to the true coupled tree-completion set and not a causal explanation of the order results. Smaller outer-set uncertainty under useful orders is compatible with earlier certification. The first-checkpoint processed feature dimension counts are saved in each orders_before_outcomes.json because blocks have unequal widths.

## Reproduction and limits

Result folder: results/review2_order_sensitivity/development_cached_v1. It contains the prospective configuration, orders frozen before outcomes, executed_results.json, all48 subject summaries, the separate subject-balanced aggregation, 17,400 selected-coordinate case rows, per-system outcome arrays and plot specifications. The executed script SHA256 is b3e8fa04075540afdcf9dc9a749af13015d7c89d93818e78a61645668a0a4ebe; aggregate result SHA256 is b554f14f9fc8f37b81ff14260c8bc431764b8c2357971b3dad076f57da225699. These hashes are research records, not manuscript prose.

Status: COMPLETE, development diagnostic only. This analysis supports an order-sensitivity statement and transparent boundaries of pruning performance. It does not establish that the locked order is optimal, does not claim runtime/energy gains for another order, and does not change the tested method.
