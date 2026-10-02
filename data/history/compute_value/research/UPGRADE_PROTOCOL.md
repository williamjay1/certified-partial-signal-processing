# ASC substantive upgrade: execution protocol and stage record

Recorded 2 October 2026 before new classifier fitting and pilot outcome evaluation. This is a local prospective execution record, not a public preregistration. The original seven-source experiments and manuscript remain developmental evidence and are preserved.

## Research route and contribution boundary

SCI/SCIE empirical method and system evaluation for Applied Soft Computing. The question is whether a cheap, observable-state computation decision can improve the prediction-risk versus complete software-time frontier of partial-signal classification. The target contribution is joint allocation of feature computation and planning computation, with the cost of estimating value inside the accounting. It is not a claim of new causal identification, new generic budget selection, universal superiority, hard real time, or guaranteed acceptance. Use the SCI workflow and applicable experiment-design, literature-review and quantitative-analysis modules. The user's instructions override template file counts and indiscriminate top-journal requirements.

Nearest-work screening has already identified material overlap: Finding the Time to Think learns cheap multiple-budget planning selection with clock-aware execution and reflex fallback; Conformal Thinking chooses computation mechanisms under risk control; AACO includes non-greedy selection and cheap imitation; SEFA and AACO report inference time. Thus neither the one-shot gate, student, risk tolerance, nor reporting latency is independently novel. Entry decision: CONDITIONAL for a bounded feasibility pilot only. A direct adapted planning-budget router and strong applicable AFA methods are required if the pilot warrants expansion. The final capability audit will give exact source versions and locators.

## Data, storage and isolation

Read the existing immutable UCI Hydraulic zip. Retain only 14 physical channels, excluding CE, CP and SE. Use 60-second cycles; pressure and motor-power channels have 6000 values, flow 600, temperature and vibration 60. Target is the four-class valve condition used in the old study. Only old outer training and validation rows (1770 cycles, 146 contiguous condition segments) may enter this pilot; old test signal values and outcomes are not evaluated.

New raw literature goes to F:/AcademicData/ASC_ComputeValue/raw with unique filenames. All scripts, working copies, features, checkpoints and outputs go to D:/MLWork/ASC_ComputeValue_20261002. Before work, available capacity was approximately 155 GB on D and 3700 GB on F. Existing raw files are not modified, moved or deleted. Training and repeated computation read D caches after one conversion from the immutable zip.

Create predictor fitting/internal validation and gate fitting/calibration/audit partitions by entire condition segment before sampling. Select exactly 300/100/100 cases from the last three partitions with a fixed, outcome-independent random seed. All other rows from their segments remain excluded from predictor fitting and its validation. Persist assigned units and sampled original rows; verify no unit intersections. Long segments must not be excluded to manufacture the 500 total. Segment identifiers are proxies for contiguous operating conditions, not independent replicate experiments.

## Signal computation and common system

Each sensor has one statistics block (11 deterministic summaries) and one frequency block (six deterministic rFFT summaries specified in code), producing 28 selectable computation blocks. Avoid expensive artificial signal operations or fabricated sensor purchase prices. Raw windows have already arrived in memory at decision start; reported time covers processing and inference, not sensor sampling, radio transfer or disk ingestion.

Fit a new masked classifier because frequency features alter the representation. Fit scaling, predictor, static ranking and donor pool using predictor fitting data; use only internal validation for early stopping and temperature. Use one seed (17) at the pilot stage. Common classifier and donor pool are frozen before branch labels. The static score is training-average immediate half-Brier improvement divided by warmed, measured block computation time. Block costs are measured using fitting cases only and do not include artificial sleep. A fixed 0.30 fraction of the sum of these measured block estimates is the pilot acquisition-computation allowance. This is an estimated feature-work allowance, not a deadline or a sensor-cost model. Report its realized overrun separately.

The environment computes a selected block from that case's raw channel on demand, scales it, and reveals it to the controller. The planner sees only observed features and fitting donors; it cannot inspect the current case's unobserved cache or outcome. Static continuation, candidates and donor count are common. Record both feature masks and action sequences to detect illegal actions or hidden-information access.

## Candidate computation actions and controls

At the first eligible prefix after one static block, choose among:

1. Skip: terminate and predict from the initial observation.
2. Static: finish with the cheap static acquisition policy.
3. Student: a cheap distilled next action, then the same static continuation.
4. Shallow: one-step donor lookahead with a partial-observation leaf predictor, then the same real static continuation.
5. Deep: donor rollout through the terminal static continuation, then the same real static continuation.

Shallow and deep use 16 donors and up to eight common acquisition candidates plus STOP. Depth is an actual continuation-depth change, not just a donor-count change. A branch can contain a STOP. Planning is called at most once; budgeted prefix legality is enforced before each actual acquisition. If the initial block leaves no legal continuation, the gate is bypassed and the terminal prediction is timed.

Controls: each standalone branch; entropy once with calibration-only threshold choice; a repeated value gate adaptation, explicitly not represented as an exact reproduction of the old model; and a quality-only one-shot budget selector with the same inexpensive state features. Also choose the best global randomized mixture from calibration branch means by enumerating single branches and feasible two-branch mixtures under the same ceiling. Freeze probabilities before audit and measure its deployment directly using a fixed case-keyed random draw independent of observed values and outcomes. Expected counterfactual mixture risk/time is only a diagnostic; realized calibration and audit deviations remain visible, with no seed replacement. A mixture matching the learned gate's action proportions is a separate diagnostic if retained. Old gain timings remain separate evidence. Student training must be independent of its gate-fitting terminal-risk labels (use grouped cross-fitting or predictor-fitting teacher cases); never report in-sample student losses as independent branch labels.

The inexpensive gate sees the initial observed statistics, observed mask, legal-set/budget summaries and known measured block estimates. It must not make an extra classifier call to construct probabilities before gating. Fit fixed-complexity multi-output ridge risk and time models, with preprocessing and regularization fixed before audit. All input construction and matrix multiplications are timed. Actual branch loss labels are final half-Brier losses of the common classifier on the same case, not donor-estimated losses. Actual branch cost labels are complete branch execution times.

## One primary endpoint and selection

Primary endpoint: minimum mean complete software execution time under a locked empirical risk ceiling. On calibration cases set the ceiling to the standalone-static mean half-Brier risk plus 0.002 absolute risk. This tolerance is a local feasibility criterion, not a journal standard or a conformal guarantee. It is fixed before execution. Choose the compute-value tradeoff from a fixed grid using calibration results, selecting the fastest feasible policy; ties go to the simpler/cheaper computation action. Fit data and calibration data may select policies; audit data may only evaluate the resulting frozen policies. Record the actual ceiling and selected setting before opening audit metrics.

The fixed compute-value tradeoff grid is lambda = {0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} in half-Brier risk per millisecond. The entropy grid is {0, 0.25, 0.5, 0.75, 1, infinity} times log(4). No denser grid is added after audit.

Gate overhead must be included when evaluating feasibility and timing. Estimated branch costs are not assumed to include the gate automatically: deployed gate execution is measured separately and in the enclosing total. A risk-only selector changes only the cost term. Entropy chooses its threshold on the same calibration cases to minimize time subject to the same risk ceiling. If no calibrated setting is feasible, record infeasibility rather than change the tolerance.

Compare the frozen compute-value policy against all frozen cheap and planning controls. A pilot expansion requires (i) audit point risk within the locked ceiling, (ii) at least 20% mean full-time saving against the fastest competing control that also satisfies that ceiling, (iii) measured gate overhead below the planning work it actually avoids, and (iv) evidence that the improvement is not reproduced by standalone skip/static/student or the quality-only depth selector. The 20% threshold is a resource-investment decision, not an accepted publication criterion. A positive pilot remains CONDITIONAL and cannot establish generalization or publication readiness.

## Timing, uncertainty and diagnostics

Warm each path before measurement. Use serial case execution, randomized method order, three timing repetitions and a single fixed CPU thread configuration. Timing repetitions are not training seeds. Measure each deployed method directly; branch timing reused for policy exploration is not enough for a final speed claim. Save per-case timings and component profiles for feature computation, routing, planning and final prediction. Any separately instrumented profile is labeled as such and not silently substituted for uninstrumented totals.

Report mean, P50, P95 and P99 full time; actual planning depth/call rate; fallback and STOP rates; feature-work allowance overrun rate; illegal-action count; mean-risk ceiling violation and its magnitude; and diagnostic exceedance rates at fixed full-time thresholds 1, 2, 5 and 10 ms. The latter are empirical tails, not hard-deadline guarantees. Label undefined avoided-planning savings explicitly rather than treating them as zero success.

Use paired segment-resampling intervals on audit differences as diagnostics when enough segments exist, preserving the same case pairing. Timing repetition variation and segment uncertainty are distinct. Do not treat 100 audit cycles as 100 independent experiments or one fitted model as five seeds. Disclose limited effective unit count and between-condition differences. If calibration and audit distributions differ, retain the failure rather than recompute the ceiling on audit.

## Stop and expansion rule

If any expansion criterion fails, stop this route at the pilot: no HAR download/training, no large baseline expansion, no rerun with a more favorable threshold, and no removal of failing sources. Update the manuscript with the negative pilot and corrected nearest-work positioning where useful, keeping the article's supported diagnostic contribution. A stop applies to this locked implementation and investment plan, not to all possible future algorithms.

Only a passing pilot authorizes the next empirical phase: UCI HAR raw inertial windows, official unseen-subject test partition untouched until final evaluation, five fitting seeds, three budgets, applicable DIME/AACO/SEFA/TAFA plus direct planning-budget selection controls, and four single-variable ablations. Strong native predictors are separately identified rather than confused with matched-classifier adapters. Formal generalization, independent capability and closed computation cost must all be demonstrated before strengthening the article's main claim.

Pre-execution entry: nearest-work capability gate CONDITIONAL for one bounded pilot; independent core/driver audit and synthetic semantic smoke checks passed. Primary agent issued PRE_EXECUTION_GO on 2 October 2026. Engineering checks cover spectra, hidden-channel invariance, legal actions, actual shallow/deep continuation, gate bypass, timing boundaries and ten executable paths. These are not fitted research outcomes. The worker is authorized to execute exactly the locked 500-case main pilot. No new pilot outcome was available when GO was recorded.

Fixed implementation identifiers at entry: engine SHA256 07077eeba181d779d639d26ae6f5ef56d8d71b4b6e4c6d04b0788e5c295b1478; driver SHA256 78cae7e9c47e9c59dba846decdddc444b44d1dfd3a0eba3cff6b278b2fe6ef85. Execution configuration also records actual script hashes, Python/library versions and thread settings. Do not modify source during the run.

## Post-execution stage decision: STOP_NO_EXPANSION

The single authorized pilot completed with exit code 0. Its 500 gate-related cases remain split 300/100/100; the original outer test was not evaluated. All ten one-shot tradeoff settings fail calibration risk feasibility. The retained diagnostic uses lambda=0. Calibration static risk is 0.3369500622153282 and the unchanged ceiling 0.3389500622153282. Audit once risk is 0.3624303114414215, complete mean time 0.786146 ms, P95 3.175750 ms and P99 4.521187 ms. No frozen comparator qualifies under the fixed risk ceiling, so the constrained 20% saving is undefined.

Risk-only and once give identical audit branches, action sequences, masks, probabilities and losses. Once chooses 82 skip, one student and 17 deep branches. All selected deep calls terminate through STOP; the student case selects block15 and then follows static continuation, reaching 12 total blocks. Gate profile mean is 0.056107 ms and signed matched saved action-computation mean is 2.199425 ms. Only that overhead criterion passes. Estimated feature allowance violations and illegal actions are zero, while separately profiled actual one-shot feature allowance overruns occur on 2% of cases.

Post-stop anchors from the same checkpoint show full-information audit risk 0.3693979681 versus zero-mask 0.3644064367; the frozen predictor limits interpretation. These are descriptive diagnostics, not a refit or basis for a new threshold. Mean-risk transport also fails for static: audit-minus-calibration risk is 0.0196064216, with an arithmetic class-mix component 0.0049749359 and within-class component 0.0146314857. This is not causal attribution.

No HAR download, additional training, new favorable seed, changed tolerance or expanded strong-baseline experiment was performed. The stage decision applies to this implementation and locked investment plan. It does not establish impossibility for other controllers.

Nearest-work independence and pilot criteria do not support the proposed superior-algorithm contribution: **NO-GO for expansion of this route**. Full-manuscript integration is authorized and completed around the supported computational-accounting contribution, preserving all original seven-source evidence and making the failed repair visible. **ASC submission remains CONDITIONAL, not submission GO**: the current evidence supports a reproducible diagnostic assessment; independent superior capability and unseen-subject generalization have not been demonstrated. Current publisher scope/author-guide pages returned403 and were not verified; topical fit is supported by the cited published ASC acquisition papers, without inferred acceptance probability.

The consolidated article is manuscript/ASC_compute_value_revision_20261002.md. Independent numerical and implementation reviews verify the new results, all six unchanged original tables and the scientific claim boundary. Raw inputs, fitted outputs and the pre-audit lock are preserved; article editing did not alter experiments. Details are in PILOT_FEASIBILITY_REPORT.md, PILOT_CODE_AUDIT.md and FINAL_NUMERICAL_AUDIT.md.
