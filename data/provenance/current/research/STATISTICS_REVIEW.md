# Read-only review of held-out HAR statistics

Reviewed 2026-10-02 while the formal timing run was active. No analysis execution, prediction, fitting, test-array loading, or policy modification was performed for this review. Inputs were `scripts/analyze_heldout.py`, `research/HAR_STATISTICAL_ANALYSIS_PLAN.md`, and the frozen driver's cohort/serialization code. This review applies the quantitative-analysis and paper-verification skills only to the authorized statistical/code audit.

**Decision: GO for the current locked 900-cohort estimand, with the interpretation limits below. No P1 arithmetic or independence defect was identified.** This decision is a code/design judgment, not a claim that the final analysis has executed successfully.

## Formula and implementation alignment

| Object | Implementation checked | Judgment |
|---|---|---|
| Half-Brier | `.5*(sum(p*p)+1-2*p_y)` | Correct normalization; additive .025 means .025 half-Brier units. |
| Relative risk | Case loss minus the same case's frozen reference loss | Correct paired difference; risk differences may be negative and are retained. |
| Every-label excess | `.5*(sum(p*p)-sum(q*q))-p+q` | Correct exact labelwise loss difference; labelwise maximum and .025 exceedance counts are retained. |
| Repetitions | Average the three timings for a case before subject aggregation | Correct; repetitions do not become independent participant observations. |
| Full-test risk | Subject loss sums divided by resampled original subject case counts | Correct original-window weighting for all 2947 cases. |
| Timed-cohort risk/time | Subject cohort sums divided by resampled cohort case counts | Correct for the locked sampled 900-window cohort; see estimand distinction below. |
| Shared-subject bootstrap | One 10000-by-9 subject draw matrix, seed7302026; common multiplicities applied to every method and all three systems | Correct paired cluster resampling. The three fitted systems are averaged inside each replicate and do not create 27 independent subjects. |
| Time reduction | `1 - mean_system_partial_time / mean_system_full_time` within each bootstrap replicate | Correct ratio of means. It is distinct from the mean of per-case or per-system percentage reductions. |
| Complete comparator | Faster original full backend for each system; its identity remains fixed within bootstrap | Matches the prospective comparator rule. Both full backends remain reported. |
| Intervals | .025/.975 quantiles of paired subject bootstrap estimates | Appropriate descriptive percentile intervals for the stated conditional estimands, with only nine independent subject clusters. |

The code verifies common method keys, the same nine subject IDs and shared cohort, auxiliary case/probability identity, no repeated timing rows, complete finite timing tensors, and saved timed probabilities against all-test outputs. Full outcome indices are the official arange(2947), so direct indexing by timed case ID is valid. The driver locks 100 cases per each of nine subjects; every denominator is therefore positive.

## NaN stopping-control certificate semantics

The three weak controls are serialized with `delta=None` in JSON and NaN delta/certificate in NPZ. The `key` helper maps NaN delta to None. Certificate-subtraction results are consequently NaN for these methods and are reported as `maximum_certificate_violation_signed=None`. This is appropriate: lack of a certificate is not converted to a valid zero certificate.

Only the named certified partial methods enter the certificate-violation assertion. Any nonfinite certificate in one of those selected tensors would fail the assertion rather than pass silently. Weak controls still receive actual maximum every-label excess, number of .025 every-label exceedances, risk differences, and reference-class disagreement counts. Their `.025` comparison is an empirical diagnostic, not a certificate claim. The class-preserving rule must retain its argmax-only wording even if its observed average risk happens to be small.

## Severity-ordered interpretation notes

**P2 — distinguish the 900-cohort estimand from a full-population time estimate.** The recorded pre-access plan states that the main time/risk display uses the same outcome-free 900 windows, with at most100 windows per subject; the current code implements that plan. Since the nine subjects each contribute100 timing windows, these timing and cohort-risk means give equal weight to the subjects. Full2947 loss uses unequal original subject window counts. Earlier coordination mentioned original2947 subject weights for time; that alternative is not currently computed. Do not describe the principal time mean as measured over all2947 windows or as original-window-weighted. A separately labeled sensitivity could reweight each subject's sampled mean time by its original full-test window count, without changing the locked principal endpoint or any method setting.

**P2 — equal-subject full-risk sensitivity is absent.** The principal full2947 risk is correctly window-weighted. A requested equal-subject sensitivity would instead average subject risk means within each system and then average systems; its paired bootstrap would average the resampled subject means. It is not currently emitted. This omission does not invalidate the locked main analysis but should not be claimed as completed.

**P2 — conditional uncertainty only.** The subject bootstrap conditions on the three frozen trained systems, their reference models/temperatures, development-selected order/schedule/control settings, and the sampled timing cohort. It estimates variation among the nine held-out subjects; it does not estimate retraining randomness, uncertainty in all prior selection stages, device-to-device variation, or hard deadlines. P95/P99 remain empirical timing quantiles, not deadline guarantees.

**P2 — keep the primary contrast separate.** The locked tight-minimax .025 endpoint may support the main empirical time claim after execution. Other deltas and controls should remain secondary/descriptive. The code does not perform significance-based selection, p-value testing, or model replacement. Counts pooled across all three systems are counts of system-case evaluations (8841), not counts of unique people or unique test windows.

## Remaining execution checks

After all seeds finish, confirm the analysis output has 21 methods, 900 common timing cases, 2947 full cases, nine subjects, all three systems, and10000 common bootstrap replicates. Compare the analyzed full reference probabilities/risk against the separately frozen 15-model reference run. Retain any weaker-control exceedances and any slower system; no inference or post-test choice should alter the frozen predictor or controls.
