# Methods mathematical and implementation text review

2 October 2026. Read-only review of `manuscript/ASC_certified_partial_processing_20261002.md`, principally §§3.1–3.6, against `research/METHOD_DERIVATION.md`, `research/CERTIFICATE_THEORY_AUDIT.md`, `scripts/certificate_kernel.py`, `research/leaf_mask_kernel.py`, and the temperature-copy implementation in `scripts/har_development_benchmark.py`. Citation capability descriptions were cross-checked against `research/VENUE_AND_REFERENCES.md`. Only this review file was written; manuscript, frozen test driver and kernels were not edited.

**Decision: mathematical GO in exact arithmetic, conditional on the stated enclosing-set assumption. No theorem-invalidating error was found.** The illustration is correct, the dual-to-primal certificate direction is correct, and the distance/class-margin corollaries are correct. Four implementation-description ambiguities and two minor precision improvements are listed below. They require short prose corrections, not new experiments or kernel changes. The fixed floating guards remain engineering choices, exactly as the manuscript already says; this audit does not upgrade them to a formal IEEE proof.

## Independent exact illustration

For `l=(4/10,3/10,1/10)` and `u=(5/10,4/10,3/10)`, the nearest feasible points to the three label vertices are respectively:

| Label | Nearest complete probability | Half-squared distance d_y | Support a_y = 1/2 − d_y |
|---|---|---:|---:|
|1| (1/2,3/10,1/5) |19/100|31/100|
|2| (2/5,2/5,1/5) |7/25|11/50|
|3| (2/5,3/10,3/10) |37/100|13/100|

All three points satisfy the box and sum-to-one constraints. The optimum label coordinate is at its greatest feasible value; the other coordinates meet the bounded water-level optimum. Thus `a=(.31,.22,.13)`, `sum(a)=.66`, and the projection adds `17/150` to each coordinate:

`p*=(127,100,73)/300`.

`||p*||²/2 = 5243/30000` and `max(a−p*) = −17/150`, so

`B*=1843/30000 = 0.06143333333333333`.

The coordinate support upper values are `a_coord=(13/40,47/200,13/100)=(.325,.235,.13)`. Their sum is `.69`; the simplex projection adds `31/300` to each coordinate:

`p_coord=(257,203,140)/600`.

`||p_coord||²/2 = 21143/120000` and `max(a_coord−p_coord) = −31/300`, so

`B_coord=8743/120000 = 0.07285833333333333`.

The six-decimal manuscript values **0.061433 and0.072858 are correct**, and `.061433333… < .065 < .072858333…` gives the stated different stopping decisions. At this particular coordinate output the exact-Q certificate also equals0.072858333…, because label3 is the active worst case and its support relaxation is already tight. Therefore the example illustrates the coupled output/certificate construction; it should not be described as a strictly looser bound for every fixed output. At the exact minimax output, the coordinate certificate would instead be0.076433333…, so the role of simplex coupling is real.

A lightweight independent check used Python `fractions.Fraction` for the values above, and AST-extracted only `simplex_projection`, `distance_dual_lower`, `certificate_at`, and `choose_probability` from the current kernel, removing decorators. It did not import Numba, JIT-compile, load models/data, train, run timing or access test cases. The isolated source functions returned:

| Variant | Returned p | Guarded certificate |
|---|---|---:|
|tight|(.42333333333333334,.33333333333333337,.24333333333333335)|0.061433333355333415|
|coordinate|(.42833333333333334,.3383333333333333,.23333333333333334)|0.07285833335533339|

The approximately2.2×10⁻¹¹ increase over exact arithmetic matches the2×10⁻¹² support/dual adjustment plus2×10⁻¹¹ primal adjustment and rounding. The differences are far below the displayed precision. This verifies the inspected source functions, not a new compiled benchmark or a formal proof of elementary-operation rounding bounds.

## Core derivation and formula/code map

| Object | Independent reasoning | Source/code correspondence | Verdict |
|---|---|---|---|
|Half-Brier units, Eq.(1)|`.5||p−e_y||²` lies in[0,1] for simplex p. δ is an additive half-Brier tolerance, not a relative percentage.|`label_losses` implements `.5*(p·p+1−2p)`. The manuscript normalization explanation is correct.|PASS|
|Relative envelope, Eqs.(2),(4)|Expand loss difference: `.5||p||²−p_y + q_y−.5||q||²`. The label support is `a_y=.5−min_Q .5||q−e_y||²`.|`certificate_at(a,p)` evaluates the primal expression at the actual returned probability.|PASS|
|Projection theorem|Introduce `w` on the simplex for the maximum. Convex/affine compact-domain minimax interchange applies; inner p optimum is p=w. Maximize `a·w−.5||w||²`, equivalently project a. Projection KKT gives a saddle point; the strongly convex quadratic plus convex maximum gives a unique primal minimizer.|`simplex_projection` uses the standard sorted-threshold formula, with normalization/residual correction before its certificate is evaluated.|PASS; mathematical p* and numerical p are appropriately distinguished|
|Label-coordinate optimum|For valid probability bounds, increasing q_y transfers mass from other coordinates toward e_y and reduces the objective until `min(u_y,1−sum(other lower))` is reached. With q_y fixed, remaining coordinates minimize squared norm with a clipped common water level.|The code instead solves the equivalent whole bounded projection using `v_i=clip(e_yi−nu,l_i,u_i)`.|PASS; equivalent formulations|
|Dual lower value, Eq.(5)|For any nu, v minimizes the Lagrangian over the box; weak duality gives `g_y(nu)≤d_y`. A merely feasible primal objective gives the opposite inequality and cannot be inserted as a lower distance.|`distance_dual_lower` bisects48 times, evaluates loss plus `nu*(sum(v)−1)`, subtracts2e−12/downward-nextafter, and clips the result below at0.|PASS in exact arithmetic; fixed rounding guard is empirical engineering|
|Bisection bracket/direction|For `0≤l≤u≤1` and `sum(l)≤1≤sum(u)`, mass at nu=−1 is sum(u), mass at nu=1 is sum(l); clipped mass is continuous and nonincreasing. If mass>1, increase nu.|Code updates low=nu for mass>1, otherwise high=nu.|PASS|
|Conservative primal bound, Eq.(6)|`d_lower≤d` implies `a_upper=.5−d_lower≥a`. Hence `.5||actual_p||²+max(a_upper−actual_p)` bounds the true envelope for that same actual p, even if p is not the exact minimax optimum.|`choose_probability` obtains actual p first and calls `certificate_at(a,p)`; it does not use the approximate dual optimal-value identity as the output certificate.|PASS|
|Coordinate support relaxation|For q in[0,1], `q_y−q_y²/2` is nondecreasing; subtracting other coordinates' squared norm is bounded above using their lower endpoints. Thus `u_y−u_y²/2−.5sum(other lower²)` is a valid support upper bound.|Variant1 constructs this vector, projects it, and certifies the returned p.|PASS|
|Tree/softmax enclosure, Eq.(7)|Known splits keep the branch selected by the actual feature; unknown splits retain both. Extra unreachable combinations are conservative. Summed class score intervals and softmax own/competitor monotonicity give Eq.(7).|Leaf masks clear only the inconsistent subtree, preserving leaves outside the current split's subtree. Lowest/highest surviving leaf values bound each tree; stable exponent shifts implement the same formula.|PASS conditional on frozen predictor semantics and valid arithmetic enclosure|
|Temperature|A fixed positive T scales each score and interval by1/T before softmax.|HAR code copies baseline before division, then divides stored leaf values, baseline, static bounds and root absolute values by T. Native full uses `_raw_predict/T`, consistent with compiled scores.|PASS; earlier aliasing error is explicitly repaired in inspected source|
|Empirical-average implication, Eq.(9)|Average valid per-case loss increases≤δ; no label sampling/exchangeability condition is needed for this deterministic implication.|Manuscript does not convert it into absolute population predictive quality or full-model accuracy.|PASS|
|Probability distance, Eq.(10)|Weight labelwise excesses by actual q: `.5||p||²−.5||q||²−q·p+||q||²=.5||p−q||²`. Therefore `||p−q||₂≤sqrt(2δ)`.|No assumption that q is the true label distribution is used.|PASS|
|Winning margin|For winning i and competitor j, `q_i−q_j ≥ (p_i−p_j)−||e_i−e_j||₂||p−q||₂ ≥ margin(p)−2sqrt(δ)`. Strict margin>2sqrt(δ) preserves the unique winning class.|The manuscript uses the correct factor2sqrt(δ), not2sqrt(2δ), and does not claim correctness against the unknown true label.|PASS|

Also correct: singleton Q gives p=q and zero excess; the full simplex gives uniform p and `(1−1/K)/2`; shrinking Q cannot increase the **exact optimum**. That monotonicity statement does not assert that a finite numerical upper estimate, a particular approximate output or complete time is monotonically improving.

## Targeted text fixes

No HIGH issue was found. Four MEDIUM implementation-description issues and two LOW precision issues follow. These are suggestions only; no manuscript edit was made.

1. **MEDIUM — checkpoint description, §3.5.** The current text says reveal four blocks and repeat at four-block checkpoints as the general method. §4.4 reports the HAR schedule-selection candidates, and root's frozen HAR schedule is12/16/18. Replace the general paragraph with a fixed prespecified sequence of prefix checkpoints; explain that Hydraulic uses batches of four and HAR uses the locked development-selected schedule. The theorem is independent of schedule choice. Do not alter the locked Hydraulic v4 driver to make prose uniform.
2. **MEDIUM — supported model family, §3.4.** “The current kernel supports numerical trees with at most63 leaves” is broader than actual code support. `FrozenForest` assumes multiclass HGB (`n_trees_per_iteration_ == len(classes_)`), numerical noncategorical nodes, scalar leaf outputs embedded with shrinkage, and the local private HGB node representation. State that the current implementation supports **numerical multiclass histogram gradient boosting with at most63 leaves per tree**; the enclosing-set argument is more general. Binary HGB's one-logit representation is not implemented by this constructor.
3. **MEDIUM — held-fixed partial evaluator, §2.1 versus §4.4.** §2.1 says controlled comparisons keep the partial evaluator fixed, but the added cheap mean-completion controls run the full classifier on zeros for unknown standardized features. They share the classifier and genuine block processing, but do not use the same interval-minimax partial-output evaluator. Change the held-fixed statement to common complete classifier, preprocessing, block operators/order and ownership of cases; explicitly allow output/stopping rules to differ. No mathematical guarantee should be attached to mean-completion controls.
4. **MEDIUM — when order is frozen, §3.5.** “Before evaluating development outcomes” can imply that no development outcome entered the selected reference model. §4.3 explicitly uses development Brier loss to select HAR HGB configuration and temperature. Say order is built from fit-data gains and fit-measured costs **after freezing the selected reference and before processing/timing development comparisons**, or simply “before its processing evaluation.” This is legitimate development selection; the prose should not suggest stricter blindness than was used.
5. **LOW — half-squared distance, §3.3 opening.** `d_y` is **half** the squared Euclidean distance to e_y; the opening “Each d_y is the squared distance” omits the factor1/2. The displayed formulas and numerical illustration already use the correct normalization.
6. **LOW — explicit feasible-bound and tolerance assumptions.** Add `0≤l≤u≤1`, `sum(l)≤1≤sum(u)` and δ≥0 where the interval-simplex numerical routine is introduced. Nonempty Q is sufficient for Theorem1's abstract projection result, but the displayed maximal q_y formula and the stated[−1,1] bisection bracket rely on valid probability endpoints. The actual softmax-bound implementation supplies those valid endpoints or safely reverts to[0,1]^K. This is an assumption clarification, not a discovered counterexample in the intended domain.

## Citation and guarantee boundaries

The inspected numbered capability claims agree with the existing verified source map: [5] is calibrated/distributional early-exit risk including relative multiclass Brier; [6] is a complete-input tree-prefix residual/class certificate; [7] concerns adversarial uncertainty/calibration; [8] supports the established candidate-leaf/feature-wise evaluator principle; [9] supports computational dependencies; [10] and the expanded working paper[11] support the minimax proper-loss precedent, with the binary Brier derivation specifically attributed to the expanded version. Temperature scaling[18] is empirical probability calibration, not an enclosing-set certificate.

The current text correctly avoids first-ever safe early exit, first Brier minimax, first shared computation, official QuickScorer reproduction, native LazyGBM reproduction, absolute prediction accuracy guarantees, guaranteed calibration, and hard-real-time deadlines. Keep these boundaries in abstract/discussion. The citation mapping was checked against the supplied source audit, not a new broad novelty search or a claim that all bibliography formatting has been finalized.

Mathematical final recommendation: retain the central theorem, exact illustration, primal-at-actual-output certificate, and distance/margin consequences. Apply the six short text corrections above, then preserve the distinction between exact mathematics, guarded implementation, native conformance, and measured test outcomes. This review authorizes no test access and does not alter a root release lock.
