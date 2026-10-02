# HAR frozen complete predictors: official test outcomes plan

Prepared on 2026-10-02 after fixed training-only development model selection and before official test access. This task writes code and performs only static compilation. Do not execute or open any test cache/member until the root task issues its immutable test lock.

## Frozen predictors

Use `results/har_model_selection/20261002T052208Z_75415f24/selected_hgb.json` for the three previously selected complete HGB references. They are seed17 HGB150/15 T2, seed29 HGB150/15 T3, and seed43 HGB100/15 T2. Use the RF, ET, LR, and SVC bundles already recorded in the same run's `candidate_records.json`. There is exactly one predeclared fitted candidate of each of those four families per seed. Preserve its previously chosen development temperature. This gives 15 complete predictors, all frozen before test access.

No predictor, scaler, temperature, parameter, or family is fitted or selected on test data. The models continue to use all 153 recomputed features. No accuracy, risk, processing time, subgroup score, or test outcome may change the selected reference models or processing method. Poor test competence must be disclosed and affect claims, not trigger test-guided refitting or replacement.

## Mandatory authorization and inputs

`har_reference_test.py` requires `--test-cache` and `--test-lock`. The immutable lock JSON must include:

```json
{
  "stage": "FROZEN_BEFORE_TEST_ACCESS",
  "har_method_frozen": true,
  "official_test_access_authorized": true,
  "selected_hgb_sha256": "SHA256 of the frozen selected_hgb.json"
}
```

The selected HGB manifest hash is validated before requesting any test array. If the lock contains `model_selection_records_sha256` or `reference_script_sha256`, validate those too. The root freezes the relevant source, train cache, loader, benchmark, and reference script hashes in its unified record. The lock must not be changed after access.

The root-approved test loader creates the test NPZ after authorization, with keys `raw_test`, `features_test`, `labels_test`, and `subjects`. This script only requests the latter three; it does not read raw signals, archives, or downloads. Expected arrays: features `(2947,153)`, encoded labels `(2947,)` in 0..5, and subjects `(2947,)` containing nine official held-out subjects. Case IDs are `arange(2947)`. Verify all nine subjects are disjoint from every saved fitting/development subject and that each has all six classes.

`--test-manifest` optionally supplies the loader's post-access manifest. Its `cache_sha256` must equal the actual test cache digest. The test cache hash is generated after access and is not required inside the pre-access lock. All inputs are read-only. Every output and temporary file is written on D; no result overwrites existing artifacts.

## Predictions and outcomes

Load each recorded joblib bundle and verify its stored file hash, family, seed, temperature, six-class column order, feature dimension, and fit-only scaler shape. Use `(features_test - saved float32 center) / saved float32 scale`, preserving the training/preprocessing arithmetic. HGB probability is stable `softmax(model._raw_predict(scaled_X) / saved_T)`. RF/ET/LR/SVC probability is `softmax(log(clip(model.predict_proba(scaled_X),1e-15,1)) / saved_T)`. Positive temperature preserves probability argmax.

Save the probability tensor `(3 seeds,5 families,2947 cases,6 classes)` and half-Brier tensor `(3,5,2947)`, with case IDs, labels, subjects, ordered family/seed metadata, and frozen temperatures. Also save the long per-case CSV for all 15 predictors. Report window-weighted half-Brier, probability-argmax accuracy, and multiclass log loss for all 15 predictors, plus the corresponding metrics and class counts for each of the nine subjects.

For every seed and baseline family, save case-level `HGB half-Brier - baseline half-Brier`, along with each subject's sum, mean, and case count. Positive values mean the frozen HGB reference has greater risk. A descriptive all-test ranking may be displayed to report competence but does not constitute model selection. The three HGB references remain frozen even if another family performs better.

## Shared nine-subject inference

The same nine held-out subjects are evaluated by all three seed-trained models. They are nine clusters, not 27 independent subject/seed observations. This script exports outcomes for the root's unified statistical analysis; it does not run a duplicate bootstrap.

The planned main bootstrap uses 10,000 repetitions and RNG seed 7302026. Uniformly resample nine subjects with replacement, sharing each resample across all seeds and methods. For a predictor/contrast, take resampled subject **loss sums divided by resampled case counts**, then average across the three training seeds. This preserves window-count weighting for the original 2,947-window risk/time estimand. A separate sensitivity estimate can average nine subject means equally. Do not switch estimands based on favorable intervals, and do not resample windows independently when reporting subject-generalization uncertainty.

## Checks and interpretation

- Static compile only before the root lock. This file and script are not evidence that official test evaluation has run.
- At execution, verify all 15 probability arrays are finite simplex points and all encoded classes align. Preserve every model's result and failure; never silently retain only a favorable seed/model.
- Report calibration/competence independently of selective processing savings. A certificate controls approximation loss relative to a frozen complete predictor; it does not prove the reference is optimal or has negligible recognition error.
- Official released inertial signals are already prefiltered/windowed. Full predictor evaluation does not establish raw sensor acquisition or device communication savings.
- SVC fitting used its prescribed LIBSVM internal fitting-only calibration and temperature selection on development subjects. No test recalibration is permitted. Its internal fitting folds are not subject-grouped, a pre-existing disclosed limitation.
- Only nine held-out subjects support cluster uncertainty. Report that boundary; do not claim broad population or device transfer without additional evidence.

Current status: code design and static verification only; official test access remains locked until authorized by root.
