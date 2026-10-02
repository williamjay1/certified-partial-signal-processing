# HAR stopping controls: development evidence

Completed all three frozen seeds in 63.97 seconds. This is development evidence; the four development subjects per seed were previously used for full-model and temperature selection. No official test was accessed and no new model was fitted.

All seeds select fixed prefix 12 and normalized entropy threshold 1.0; entropy_force_full=false. Their predictions are identical. This degeneration is retained, rather than expanding the candidate grid after observing the results.

Selection used observed-label sample mean relative half-Brier <=0.025 and then minimum original fit-priced block cost. It did not use measured development runtime. All 14 candidates per seed remain available in all_candidate_summaries.json and all_candidate_outcomes.npz.

|Seed|Method|All-dev half-Brier|Observed mean excess|Every-label violation fraction|Maximum every-label excess|Mean blocks|
|---:|---|---:|---:|---:|---:|---:|
|17|Score-separated argmax|0.076027|0.012038|29.57%|0.170339|12.601|
|17|Fixed prefix 12|0.068457|0.004469|16.64%|0.414161|12.000|
|17|Entropy 1.0|0.068457|0.004469|16.64%|0.414161|12.000|
|17|Tight minimax .025|0.064937|0.000949|0.00%|0.024880|13.876|
|17|Native full|0.063989|0.000000|0.00%|0.000000|18.000|
|17|Compiled full|0.063989|0.000000|0.00%|0.000000|18.000|
|29|Score-separated argmax|0.127463|0.034405|67.55%|0.172890|12.472|
|29|Fixed prefix 12|0.097167|0.004109|19.91%|0.282792|12.000|
|29|Entropy 1.0|0.097167|0.004109|19.91%|0.282792|12.000|
|29|Tight minimax .025|0.097797|0.004739|0.00%|0.024335|15.514|
|29|Native full|0.093058|-0.000000|0.00%|0.000000|18.000|
|29|Compiled full|0.093058|0.000000|0.00%|0.000000|18.000|
|43|Score-separated argmax|0.095410|0.021679|45.50%|0.169882|12.881|
|43|Fixed prefix 12|0.087765|0.014034|26.78%|0.492370|12.000|
|43|Entropy 1.0|0.087765|0.014034|26.78%|0.492370|12.000|
|43|Tight minimax .025|0.076655|0.002925|0.00%|0.024781|14.763|
|43|Native full|0.073731|0.000000|0.00%|0.000000|18.000|
|43|Compiled full|0.073731|0.000000|0.00%|0.000000|18.000|

Every-label violations mean at least one possible label has loss excess >0.025, with 2e-12 numerical tolerance. The mean-completion controls pass the observed mean-risk target but violate the individual target in 16.64–26.78% of development cases. They therefore establish a useful cheap cost/risk comparison, not a pointwise or CRC guarantee.

Score-separated argmax agrees with the full reference on every development window. It does not constrain probability loss; seed 29 observed mean excess is 0.034405, above the primary tolerance. It is an explicit score-interval reference control, not a reproduction of native LazyGBM.

The tight minimax method has zero development every-label violations; this check is numerical evidence under the frozen software arithmetic, rather than a machine-level formal proof.

|Seed|Method|Mean ms|P95 ms|P99 ms|Saving vs faster full|
|---:|---|---:|---:|---:|---:|
|17|Score-separated argmax|0.901463|1.470990|1.857219|18.70%|
|17|Fixed prefix 12|0.833379|1.360930|1.734148|24.84%|
|17|Entropy 1.0|0.820958|1.329145|1.642447|25.96%|
|17|Tight minimax .025|0.995042|1.603125|2.124606|10.26%|
|17|Native full|6.780254|9.245085|10.525869|-511.50%|
|17|Compiled full|1.108786|1.765515|2.224144|0.00%|
|29|Score-separated argmax|0.971144|1.552955|1.825522|23.00%|
|29|Fixed prefix 12|0.894193|1.407615|1.735238|29.10%|
|29|Entropy 1.0|0.888223|1.421515|1.608554|29.57%|
|29|Tight minimax .025|1.260414|2.091775|2.570775|0.06%|
|29|Native full|7.635722|10.612670|11.946142|-505.44%|
|29|Compiled full|1.261184|1.984405|2.303962|0.00%|
|43|Score-separated argmax|0.902090|1.487915|1.810742|20.25%|
|43|Fixed prefix 12|0.796689|1.278375|1.536683|29.57%|
|43|Entropy 1.0|0.800772|1.291025|1.484173|29.21%|
|43|Tight minimax .025|1.051256|1.738415|2.051152|7.06%|
|43|Native full|5.285404|7.437715|8.272455|-367.26%|
|43|Compiled full|1.131144|1.809345|2.039883|0.00%|

Timings use the original 300-case development cohort for each seed, three repetitions, randomized method order, and actual raw statistics/FFT plus fit-only scaling. All six methods were measured contemporaneously; compiled full was faster than native full in every seed. Per-case probabilities, block/fallback flags, every-label and observed-label excesses, and raw timings are preserved in timing_case_results.jsonl.

Fixed/entropy controls save roughly 25–30% on this cohort. Tight minimax saves 10.26%, 0.06%, and 7.06% in seeds 17, 29, and 43. The near-zero seed 29 benefit is retained; no independent test or uncertainty claim is drawn from this development comparison.

Saved outcomes, all candidate gates, deterministic selection, probability validity, every-case argmax preservation, main certificate tolerance, cache/raw selected outputs, repeat outputs, and mean/P95/P99 timings were independently recomputed from saved arrays/rows; verification.json records PASS. No kernels, predictors, or settings were changed after freezing.

Frozen state SHA256: `cff68cd1f3a165b5e8f7317b29d96ec603f0185ba5a4617955dd244c7536f6fb`. Public script SHA256: `9678d965b35c92a6b1245186730e6c1f34defa2f5cdc1169bf11164971ea0b21`. Root formal-test integration must load this state and use the frozen dispatcher; official test access still requires the immutable root lock.
