# HAR development full model selection

Completed 24/24 fixed models; 0 failures. No official test file or source ZIP was opened.

Development metrics below include model/temperature selection on held-out subjects. They do not estimate independent official-test performance.

| Seed | Model | T | Half-Brier | Accuracy | Log loss | Fit s |
|---:|---|---:|---:|---:|---:|---:|
| 17 | hgb_i100_l15 | 2 | 0.064616 | 0.916094 | 0.266434 | 7.341 |
| 17 | hgb_i100_l31 | 2 | 0.066455 | 0.916781 | 0.282523 | 8.079 |
| 17 | hgb_i150_l15 | 2 | 0.063989 | 0.917469 | 0.264988 | 9.256 |
| 17 | hgb_i150_l31 | 2 | 0.065933 | 0.920220 | 0.283112 | 14.374 |
| 17 | random_forest | 0.75 | 0.078774 | 0.904402 | 0.387712 | 13.222 |
| 17 | extra_trees | 0.75 | 0.077470 | 0.903026 | 0.311246 | 4.530 |
| 17 | logistic | 1.5 | 0.062261 | 0.915406 | 0.225858 | 0.308 |
| 17 | rbf_svc | 1.25 | 0.059324 | 0.926410 | 0.248311 | 3.139 |
| 29 | hgb_i100_l15 | 2 | 0.093925 | 0.892502 | 0.416664 | 6.257 |
| 29 | hgb_i100_l31 | 2 | 0.096705 | 0.889847 | 0.438381 | 8.245 |
| 29 | hgb_i150_l15 | 3 | 0.093058 | 0.894492 | 0.420406 | 8.929 |
| 29 | hgb_i150_l31 | 3 | 0.095523 | 0.891175 | 0.434281 | 12.996 |
| 29 | random_forest | 0.75 | 0.092796 | 0.891838 | 0.473168 | 12.899 |
| 29 | extra_trees | 0.75 | 0.090552 | 0.908427 | 0.376412 | 4.709 |
| 29 | logistic | 3 | 0.117351 | 0.857996 | 0.427378 | 0.423 |
| 29 | rbf_svc | 1.5 | 0.087731 | 0.892502 | 0.385752 | 4.255 |
| 43 | hgb_i100_l15 | 2 | 0.073731 | 0.892729 | 0.289983 | 6.759 |
| 43 | hgb_i100_l31 | 2 | 0.075838 | 0.895608 | 0.300243 | 9.720 |
| 43 | hgb_i150_l15 | 2 | 0.074290 | 0.892009 | 0.284790 | 11.095 |
| 43 | hgb_i150_l31 | 2 | 0.076498 | 0.892729 | 0.296313 | 14.051 |
| 43 | random_forest | 0.75 | 0.074087 | 0.897768 | 0.272630 | 17.249 |
| 43 | extra_trees | 0.75 | 0.073328 | 0.899208 | 0.265938 | 5.907 |
| 43 | logistic | 2 | 0.094783 | 0.871130 | 0.417369 | 0.621 |
| 43 | rbf_svc | 1.25 | 0.080012 | 0.886249 | 0.323808 | 4.574 |

## Selected complete HGB references

- Seed 17: hgb_i150_l15, T=2, half-Brier=0.063989; checkpoint `D:\MLWork\ASC_CertifiedProcessing_20261002\checkpoints\har_models\20261002T052208Z_75415f24\seed17__hgb_i150_l15.joblib`.
- Seed 29: hgb_i150_l15, T=3, half-Brier=0.093058; checkpoint `D:\MLWork\ASC_CertifiedProcessing_20261002\checkpoints\har_models\20261002T052208Z_75415f24\seed29__hgb_i150_l15.joblib`.
- Seed 43: hgb_i100_l15, T=2, half-Brier=0.073731; checkpoint `D:\MLWork\ASC_CertifiedProcessing_20261002\checkpoints\har_models\20261002T052208Z_75415f24\seed43__hgb_i100_l15.joblib`.

## Full predictor competence and boundaries

- Seed 17: strongest full model on the selected development score is rbf_svc (0.059324); chosen HGB risk excess is 0.004664.
- Seed 29: strongest full model on the selected development score is rbf_svc (0.087731); chosen HGB risk excess is 0.005327.
- Seed 43: strongest full model on the selected development score is extra_trees (0.073328); chosen HGB risk excess is 0.000403.

All full models use the same recomputed 153 features and same saved subject-separated fitting/development rows. Every supplied scaler was verified exactly against fitting rows; it was not refit. HGB uses six-class additive logits divided by selected T. RF/ET/LR/SVC use log native probabilities clipped at 1e-15 divided by T. Positive temperature preserves each model probability argmax.

SVC probability=True performs LIBSVM internal fitting-data-only calibration folds. Those folds are not subject-grouped and may contain overlapping windows from the same fitting subject. The four development subjects remain untouched by predictor fitting; all reported temperature choices nevertheless reuse them.

The complete baseline comparison is limited to this fixed registry; it is not an exhaustive HAR benchmark or deep model comparison. HGB selected from four configurations is a reference predictor for certification, not a guarantee of best possible activity recognition. The method must be evaluated against frozen full references and cannot claim that its risk certificate makes an inferior reference optimal.

The three development splits reuse official training subjects and may overlap across seeds. Treat them as sensitivity replications, not three independent test cohorts. No significance, subject-population generalization, or independent-test claim is made here.

Warnings and failures are retained in candidate_records.json; all calibration trials, raw T=1 metrics, class counts, shape, package/thread records, and hashes are included there or in prospective_registry.json. Development per-window probabilities/losses are preserved for later paired analysis. Training times describe these fits and are not the selective processing timing experiment.

## Descriptive comparison across fixed splits

These are equal-weight averages of the three overlapping subject splits, not independent-replication estimates or significance tests.

| Full family | Mean half-Brier | Mean accuracy | Mean log loss |
|---|---:|---:|---:|
| Selected HGB | 0.076926 | 0.901563 | 0.325126 |
| RandomForest | 0.081885 | 0.898003 | 0.377837 |
| ExtraTrees | 0.080450 | 0.903554 | 0.317865 |
| Logistic | 0.091465 | 0.881511 | 0.356868 |
| RBF SVC | 0.075689 | 0.901720 | 0.319290 |

The chosen complete HGB is competitive within this fixed registry, with average half-Brier 0.001237 above SVC and better than RF, ET, and logistic on this average. It does not dominate the baselines: SVC has the lowest risk for seeds 17 and 29; ET has the lowest risk for seed 43. All three chosen HGBs are slightly worse in risk than the best complete model for their individual split. The paper should retain these full-model results and describe the certified method as controlling approximation loss relative to a frozen HGB reference, rather than establishing the reference as the globally strongest HAR classifier.

All three RF/ET temperature optima occur at the grid's lower endpoint 0.75; the seed-29 HGB and logistic optima occur at the upper endpoint 3. This fixed prospective exercise does not expand those ranges after looking at development scores. A bounded grid optimum is not proof of optimal calibration over all positive temperatures. HGB has four candidate configurations whereas each other family has one configuration; the comparison is a competence check with a disclosed finite tuning budget, not equal exhaustive family optimization.

There were no fit failures or convergence warnings. The three retained SVC warnings concern future deprecation of `probability=True` in the installed scikit-learn 1.9.1; the specified models completed successfully. The release is recorded for reproducibility.

Post-run artifact verification recomputed half-Brier and probability-argmax accuracy from all 24 saved development prediction files, checked the three chosen HGB rankings, checked unchanged prospective plan/script hashes, and reloaded each chosen checkpoint. Reloaded `softmax(raw_logits / T)` on 32 development windows per seed matched stored calibrated probabilities with maximum absolute difference 0. All fitted scaler arrays were verified against fitting-only rows before the first fit. Full details are in `verification.json`.
