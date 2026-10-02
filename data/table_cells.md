# Unchanged manuscript table cells for artwork regeneration


## Table 1

| Research line | Main object addressed | Relation to the present method |
|---|---|---|
| Active feature acquisition [12–16] | Predictive utility and observation cost under subset observations | May supply a feature schedule; does not substitute for checking the stated relative-loss certificate |
| Risk-controlled early exit [5] | Calibrated distributional risk or a final-model performance gap | Establishes the relative Brier target; the quantifier here is individual input and every label |
| Certified lazy boosted-tree inference [6] | Preservation of the complete winning class | Supplies a distinct output criterion, tested by score separation |
| Adversarial uncertainty certification [7] | Worst calibration quantities over a perturbation set | Related certification tools with a different domain and loss object |
| Present construction | Worst relative multiclass half-Brier loss over complete outputs consistent with observed blocks | Constructs and certifies a probability vector, while charging its full processing cost |


## Table 2

| Source | Task | Computed representation | Fitting/development | Frozen evaluation |
|---|---|---|---|---|
| Hydraulic | Four valve-condition classes |14 channels;28 blocks;238 features |907/223 cycles |435 previously held-out cycles, grouped by operating-segment proxies |
| Smartphone HAR | Six activity classes |9 channels;18 blocks;153 features |17/4 official training subjects per system |2,947 windows from nine official test subjects |


## Table 3

| Complete family | System 17: half-Brier / accuracy | System 29: half-Brier / accuracy | System 43: half-Brier / accuracy | Mean half-Brier / accuracy |
|---|---:|---:|---:|---:|
| HGB, certified method's reference | 0.056736 / 91.79% | 0.077797 / 90.33% | 0.049557 / 93.59% | 0.061363 / 91.90% |
| Random forest | 0.071221 / 90.74% | 0.079527 / 88.97% | 0.060011 / 92.37% | 0.070253 / 90.69% |
| Extra trees | 0.068181 / 91.21% | 0.076067 / 90.50% | 0.064143 / 92.09% | 0.069464 / 91.27% |
| Logistic regression | 0.069117 / 90.67% | 0.094257 / 88.60% | 0.056360 / 92.60% | 0.073245 / 90.62% |
| RBF SVC | 0.050195 / 93.28% | 0.063627 / 91.86% | 0.042154 / 94.67% | 0.051992 / 93.27% |


## Table 4

| Complete implementation | Mean / median time, ms | P95 / P99, ms | Same-cohort half-Brier | Same-cohort accuracy |
|---|---:|---:|---:|---:|
| Native HGB | 5.935996 / 5.636200 | 9.005495 / 10.461700 | 0.066411 | 91.11% |
| Compiled HGB | 1.149548 / 1.049300 | 1.802545 / 2.151405 | 0.066411 | 91.11% |
| Random forest | 14.798855 / 13.466150 | 22.953085 / 26.324272 | 0.075713 | 89.85% |
| Extra trees | 15.155591 / 13.812650 | 23.348470 / 26.783967 | 0.073094 | 90.74% |
| Logistic regression | 1.295108 / 1.196150 | 2.024415 / 2.360212 | 0.079415 | 89.67% |
| RBF SVC | 1.625706 / 1.498900 | 2.519920 / 2.975855 | 0.056553 | 92.56% |


## Table 5

| System | Complete full time, ms | Partial time, ms | Time reduction | Partial / full half-Brier | Mean blocks of 18 | Full fallback |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 1.122237 | 1.016754 | 9.40% | 0.066525 / 0.063736 | 14.5089 | 19.89% |
| 29 | 1.051749 | 1.023604 | 2.68% | 0.090309 / 0.083996 | 15.7778 | 38.22% |
| 43 | 1.066047 | 0.963610 | 9.61% | 0.054615 / 0.051502 | 14.6711 | 17.33% |
| Mean | 1.080011 | 1.001322 | 7.29% | 0.070483 / 0.066411 | 14.9859 | 25.15% |


## Table 6

| Procedure | Complete mean time, ms | Time reduction | Half-Brier | Observed-label mean increase | Every-label exceedances / 2,700 |
|---|---:|---:|---:|---:|---:|
| Compiled full | 1.080011 | — | 0.066411 | 0 | 0 |
| Simplex-aware minimax, 0.025 | 1.001322 | 7.29% | 0.070483 | 0.004072 | 0 |
| Coordinate minimax, 0.025 | 1.001969 | 7.23% | 0.070369 | 0.003958 | 0 |
| Midpoint with tight certificate, 0.025 | 1.137513 | −5.32% | 0.066579 | 0.000168 | 0 |
| Minimax with uniform checkpoints, 0.025 | 1.108992 | −2.68% | 0.072028 | 0.005617 | 0 |
| Score-separated class preservation | 0.862145 | 20.17% | 0.093246 | 0.026834 | 1,289 |
| Fixed-prefix mean completion | 0.775521 | 28.19% | 0.073406 | 0.006995 | 641 |
| Entropy mean completion | 0.768284 | 28.86% | 0.073406 | 0.006995 | 641 |


## Table 7

| Half-Brier tolerance | HAR complete time, ms | HAR reduction | HAR same-cohort mean increase | Hydraulic complete time, ms | Hydraulic reduction | Hydraulic mean increase |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.262918 | −16.94% | 0 | 3.108712 | −8.68% | 0 |
| 0.01 | 1.065939 | 1.30% | 0.001226 | 2.261661 | 20.93% | 0.000713 |
| 0.025 | 1.001322 | 7.29% | 0.004072 | 2.139557 | 25.20% | 0.002957 |
| 0.05 | 0.949258 | 12.11% | 0.008115 | 1.974194 | 30.98% | 0.004959 |
| 0.10 | 0.886820 | 17.89% | 0.016082 | 1.726200 | 39.65% | 0.013851 |


## Table 8

| Evaluation system | Full / partial P50, ms | Full / partial P95, ms | Full / partial P99, ms |
|---|---:|---:|---:|
| HAR 17 | 0.982250 / 0.914200 | 1.908330 / 1.819120 | 2.320208 / 2.341802 |
| HAR 29 | 0.915500 / 0.941050 | 1.600410 / 1.706305 | 2.097817 / 2.187748 |
| HAR 43 | 0.930950 / 0.869850 | 1.770430 / 1.687745 | 2.225132 / 2.276310 |
| Hydraulic confirmation | 2.509400 / 1.943500 | 4.245220 / 3.986300 | 4.843356 / 4.813808 |


## Table 9

| Processing outcome | System–windows | Partial / full mean, ms | Partial-minus-full mean, ms | Partial slower |
|---|---:|---:|---:|---:|
| Stop at 12 blocks | 1,024 | 0.781742 / 1.077781 | −0.296039 | 4.10% |
| Stop at 16 blocks | 997 | 1.054817 / 1.072119 | −0.017302 | 45.34% |
| 18 blocks, exact fallback | 679 | 1.253923 / 1.094962 | +0.158962 | 84.09% |


## Table 10

| Hydraulic procedure at tolerance 0.025 | Complete time, ms | Reduction against compiled full | Half-Brier | Mean blocks | Full fallback |
|---|---:|---:|---:|---:|---:|
| Compiled full | 2.860459 | — | 0.126473 | 28.0000 | — |
| Simplex-aware minimax | 2.139557 | 25.20% | 0.129430 | 19.0069 | 19.77% |
| Coordinate minimax | 2.070535 | 27.62% | 0.129353 | 19.0345 | 19.77% |
| Midpoint with tight certificate | 2.642566 | 7.62% | 0.126445 | 23.4299 | 35.40% |


## Table 11

| Order | Mean blocks of 18 | Exact fallback | First-12-block sum of interval widths | First-12-block diameter upper proxy |
|---|---:|---:|---:|---:|
| Locked gain/time | 14.7179 | 18.33% | 0.623226 | 0.340698 |
| Descending gain only | 14.5794 | 14.59% | 0.594592 | 0.326628 |
| Reverse locked order | 18.0000 | 100.00% | 5.670742 | 1.414214 |
| One fixed random order per system | 17.1791 | 68.84% | 3.468061 | 1.246648 |


## Table 12

| Complete predictor | Route to a sound partial enclosure | Additional requirement |
|---|---|---|
| Random forest / extra trees | Bound each leaf's class-probability vector and average tree bounds | Preserve probability averaging and the frozen temperature transformation; a boosted-logit sum is not equivalent |
| XGBoost / LightGBM multiclass trees | Sum reachable-leaf bounds for each class score, then bound the link | Validate intercepts, class/tree indexing, missing and categorical splits, and libraries' exact output semantics |
| Larger numerical trees | Use multiple mask words or another candidate-leaf representation | Charge extra update and interval-extraction work; the one-word runtime does not transfer automatically |
| RBF SVC | Enclose kernels and decision values under partial features | Also enclose the fitted pairwise probabilities and multiclass coupling; the present softmax score formula cannot simply replace native SVC semantics |
| Neural network | Propagate sound feature bounds to logits, then probabilities | Justify bounds for uncomputed signal features and control relaxation cost; empirical training minima/maxima alone do not enclose unseen inputs |
