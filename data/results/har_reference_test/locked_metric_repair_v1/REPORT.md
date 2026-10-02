# Frozen complete-reference test competence

All15 frozen HGB/RF/ET/LR/SVC family/system combinations evaluated all2947 official test windows from nine held-out subjects. No model, temperature, scaler, or partial policy was fitted or reselected. Full-model prediction time is diagnostic only and is not a complete raw processing-time comparison.

|System|Family|Frozen T|Half-Brier|Accuracy|Secondary log loss|
|---:|---|---:|---:|---:|---:|
|17|HGB|2|0.056736|91.79%|0.218316|
|17|RF|0.75|0.071221|90.74%|0.251516|
|17|ET|0.75|0.068181|91.21%|0.247169|
|17|LR|1.5|0.069117|90.67%|0.276814|
|17|SVC|1.25|0.050195|93.28%|0.197515|
|29|HGB|3|0.077797|90.33%|0.363432|
|29|RF|0.75|0.079527|88.97%|0.292185|
|29|ET|0.75|0.076067|90.50%|0.278164|
|29|LR|3|0.094257|88.60%|0.408141|
|29|SVC|1.5|0.063627|91.86%|0.257200|
|43|HGB|2|0.049557|93.59%|0.207279|
|43|RF|0.75|0.060011|92.37%|0.219846|
|43|ET|0.75|0.064143|92.09%|0.236783|
|43|LR|2|0.056360|92.60%|0.212969|
|43|SVC|1.25|0.042154|94.67%|0.158442|

|Family|Mean system half-Brier|Mean system accuracy|Mean secondary log loss|
|---|---:|---:|---:|
|HGB|0.061363|91.90%|0.263009|
|RF|0.070253|90.69%|0.254516|
|ET|0.069464|91.27%|0.254039|
|LR|0.073245|90.62%|0.299308|
|SVC|0.051992|93.27%|0.204386|

SVC has lower observed test half-Brier than frozen HGB in every system; HGB exceeds SVC by0.006541,0.014170 and0.007403. The descriptive three-system means are HGB0.061363 and SVC0.051992, a difference0.009371. HGB beats RF and LR on this representation in every system, and beats ET in systems17/43 while ET is slightly better in system29. The reference is therefore a competent tree predictor, with an observable gap to SVC; the study cannot claim the strongest available HAR accuracy or probability predictor.

These comparisons diagnose the frozen reference. They do not change it. The method certificate currently supports numerical trees, so replacing the reference with SVC based on test outcomes would both violate the freeze and change the evaluated capability.

All15 case probability/loss arrays,135 subject-model records, paired HGB-minus-baseline case/subject differences and shared-nine-subject sums are preserved. Root owns the common subject bootstrap; this run has no confidence interval or independent27-cluster claim.

The first attempt retained12 successful records and three LR metric-conformance failures in locked_execution_v1. Root authorized a separate post-lock technical recovery. Only secondary log loss uses a separately row-normalized float64 copy; original probabilities, half-Brier and accuracy retain the frozen calculation. The original locked script and root lock are byte-identical; the recovery manifest records POSTLOCK_SECONDARY_METRIC_CONFORMANCE_REPAIR and the new script hash. All12 original successful half-Brier/accuracy results are exactly identical in the recovery.

Saved-case risk/accuracy/log-loss, paired subject sums, recovery prediction AST, unchanged locked hashes and complete-reference agreement with the formal21 driver passed verification. The maximum HGB probability discrepancy is below2e-12 (see verification.json). No retraining or runtime rerun occurred for this audit.
