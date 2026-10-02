# Hydraulic frozen confirmation: independent saved-output review

Review date: 2 October 2026. No new fitting, policy selection, raw parsing or timing was performed. This review independently computed half-Brier and all-label relative losses from saved probability arrays, and checked all22,185 JSONL records against the17×435×3 outcome tensor. It did not import the frozen driver, certificate kernel or loss helper. Authoritative run: `results/hydraulic_external_confirmation/20261002T061642Z_3208a13f`.

## Audit result

**PASS**: All17 complete means, medians, P95/P99, risks, accuracies, block counts, fallback rates, class agreement and violation summaries agree with independently recomputed values (absolute check tolerance2e−12). Every method/case/repetition occurs exactly once. JSONL probabilities, references, complete times and processing counters agree exactly with NPZ; maximum scalar loss difference from the alternative expanded half-Brier formula was2.79e-16. Probabilities are finite, nonnegative and sum to1 within2.22e-16. All repeated probabilities are exactly identical. Both complete predictors and allthree delta0 variants exactly equal saved full reference probabilities on all435 cases.

Observed four-class label support: [70, 70, 70, 225]. There are435 unique original rows and48 contiguous same-condition proxy segments. The saved source/scaler provenance verifies zero row and proxy overlap against all1770 non-test cases (146 proxies), including907 fit and223 internal development. The source hash was unchanged, and no old test feature matrix was decoded by the driver. This review verified the saved outcome and JSONL hashes against the execution manifest.

Independent maximum certificate violation over all emitted probabilities and allfour possible labels: 0; maximum delta violation across all15 partial settings: 0. Positive violations above2e−12: **0**. Earlier driver prefix checks covered9135 stages on all435 cases andthree variants. Those untimed trajectory checks are confirmed by their saved report, not rerun here. Neither this numerical check nor the driver's guards constitute an all-input floating-point proof.

## Complete17-setting comparison

Complete time is averaged over435×3 observations per setting. Risk is averaged over435 cases, so repetitions are not treated as independent prediction observations. Time reductions use the contemporaneous faster complete implementation, compiled full. Violations column gives case counts for certificate/delta respectively, using2e−12 only as the frozen numerical check tolerance.

| Method | Delta | Mean complete ms | Half-Brier | Risk difference vs full | Time reduction vs compiled | Mean blocks | Certificate/delta violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| tight_minimax | 0 | 3.108712 | 0.126473069 | +0.000000000 | -8.6788% | 28.0000 | 0/0 |
| tight_minimax | 0.01 | 2.261661 | 0.127185983 | +0.000712915 | +20.9336% | 20.5425 | 0/0 |
| tight_minimax | 0.025 | 2.139557 | 0.129430330 | +0.002957261 | +25.2023% | 19.0069 | 0/0 |
| tight_minimax | 0.05 | 1.974194 | 0.131432315 | +0.004959247 | +30.9833% | 17.4437 | 0/0 |
| tight_minimax | 0.1 | 1.726200 | 0.140324042 | +0.013850973 | +39.6531% | 15.6046 | 0/0 |
| coordinate_minimax | 0 | 3.119357 | 0.126473069 | +0.000000000 | -9.0509% | 28.0000 | 0/0 |
| coordinate_minimax | 0.01 | 2.292632 | 0.127160358 | +0.000687289 | +19.8509% | 20.5609 | 0/0 |
| coordinate_minimax | 0.025 | 2.070535 | 0.129353377 | +0.002880308 | +27.6153% | 19.0345 | 0/0 |
| coordinate_minimax | 0.05 | 1.974214 | 0.131860557 | +0.005387488 | +30.9826% | 17.4621 | 0/0 |
| coordinate_minimax | 0.1 | 1.756832 | 0.139927944 | +0.013454876 | +38.5822% | 15.7241 | 0/0 |
| tight_midpoint | 0 | 3.082730 | 0.126473069 | +0.000000000 | -7.7705% | 28.0000 | 0/0 |
| tight_midpoint | 0.01 | 2.740871 | 0.126469077 | -0.000003992 | +4.1807% | 24.9839 | 0/0 |
| tight_midpoint | 0.025 | 2.642566 | 0.126444809 | -0.000028260 | +7.6174% | 23.4299 | 0/0 |
| tight_midpoint | 0.05 | 2.409094 | 0.126499260 | +0.000026191 | +15.7795% | 21.7839 | 0/0 |
| tight_midpoint | 0.1 | 2.162737 | 0.125187694 | -0.001285375 | +24.3920% | 19.7241 | 0/0 |
| native_full | full | 6.819340 | 0.126473069 | +0.000000000 | -138.4002% | 28.0000 | 0/— |
| compiled_full | full | 2.860459 | 0.126473069 | +0.000000000 | +0.0000% | 28.0000 | 0/— |

The locked primary tight minimax delta=.025 saves25.2023% complete mean time, with observed-label half-Brier increase0.002957261; its maximum every-label increase is0.024909060<.025. Its accuracy and reference class agreement are84.3678% and100%. Its fallback rate is86/435. Coordinate minimax is faster at the same delta; tight minimax is therefore not the universally fastest implementation. Delta0 incurs additional certificate work and is slower than compiled full. Midpoint keeps different proper-loss behavior; its lower observed-label risk at some settings does not imply a uniformly better certificate. Allthese counterresults must remain reported.

## Proxy-segment descriptive analysis

For each proxy, each case is first averaged over its three timings. The proxy's time difference is then the equally weighted mean of its case differences: tight(delta=.025) minus compiled full. Its percentage is1−proxy mean tight/proxy mean full. Across all435 cases, the main25.2023% value weights cases; the48-proxy distributions below weight each proxy equally. These are distinct summaries.

37/48 proxies have lower mean tight time, 11/48 do not. 38/48 have positive mean observed-label risk increase. Distribution (minimum / lower quartile / median / upper quartile / maximum):

- n: 1 / 10 / 10 / 10 / 19
- difference_ms: -3.52180667 / -1.30963833 / -0.891936667 / -0.0538408333 / 0.431253333
- reduction_pct: -14.3271536 / 2.07884352 / 32.8594114 / 47.8033317 / 64.7759709
- risk_excess: -0.0398477486 / 0.00152586909 / 0.00728436854 / 0.0106939623 / 0.0199359299

Proxy groups are derived from contiguous identical-condition segments, not independently verified experiments or machines. This descriptive consistency check supplies neither48 independent replications nor a population-level confidence statement. No t-test, p-value or bootstrap interval was added, because resampling these constructed groups would require an unverified independence/exchangeability premise. The retained per-proxy rows permit clearly conditional sensitivity work if needed; no new significance claim is necessary for the manuscript.

| Proxy ID | Cases | Tight mean ms | Compiled mean ms | Paired difference ms | Time reduction | Mean risk excess | Full fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.863667 | 3.257067 | -1.393400 | +42.7808% | +0.002446870 | 0 |
| 2 | 10 | 3.074660 | 2.988927 | +0.085733 | -2.8684% | -0.011220277 | 7 |
| 4 | 10 | 3.751200 | 3.319947 | +0.431253 | -12.9898% | +0.000000000 | 10 |
| 8 | 10 | 3.232620 | 3.558347 | -0.325727 | +9.1539% | +0.010487672 | 1 |
| 10 | 10 | 2.167893 | 3.080840 | -0.912947 | +29.6330% | +0.012145615 | 0 |
| 14 | 10 | 3.294443 | 3.110750 | +0.183693 | -5.9051% | -0.013794143 | 6 |
| 21 | 10 | 3.152933 | 2.827483 | +0.325450 | -11.5102% | -0.018253939 | 8 |
| 22 | 1 | 2.688233 | 2.523900 | +0.164333 | -6.5111% | -0.039847749 | 0 |
| 31 | 10 | 2.457937 | 2.917690 | -0.459753 | +15.7574% | +0.003817577 | 1 |
| 38 | 1 | 3.440000 | 3.218333 | +0.221667 | -6.8876% | +0.000000000 | 1 |
| 41 | 10 | 3.360487 | 3.115140 | +0.245347 | -7.8759% | -0.035064935 | 7 |
| 50 | 10 | 2.032493 | 3.253587 | -1.221093 | +37.5307% | +0.013950994 | 1 |
| 53 | 10 | 3.150953 | 3.150057 | +0.000897 | -0.0285% | -0.027606955 | 2 |
| 55 | 10 | 3.121170 | 3.116270 | +0.004900 | -0.1572% | +0.007188135 | 3 |
| 70 | 10 | 2.340263 | 2.561887 | -0.221623 | +8.6508% | +0.007575788 | 1 |
| 73 | 10 | 2.668163 | 2.644707 | +0.023457 | -0.8869% | +0.005122362 | 6 |
| 79 | 10 | 2.753653 | 2.408573 | +0.345080 | -14.3272% | +0.000224034 | 8 |
| 85 | 10 | 2.440980 | 2.825000 | -0.384020 | +13.5936% | +0.002423132 | 0 |
| 91 | 10 | 1.305217 | 2.608443 | -1.303227 | +49.9619% | +0.011312832 | 0 |
| 97 | 10 | 1.332667 | 2.578130 | -1.245463 | +48.3088% | +0.009395831 | 0 |
| 102 | 10 | 3.516953 | 7.038760 | -3.521807 | +50.0345% | +0.011379311 | 0 |
| 104 | 10 | 1.464923 | 2.863117 | -1.398193 | +48.8347% | +0.014619634 | 0 |
| 107 | 10 | 1.540447 | 2.941740 | -1.401293 | +47.6348% | +0.014064443 | 0 |
| 110 | 10 | 1.832523 | 3.019863 | -1.187340 | +39.3177% | +0.014340612 | 0 |
| 114 | 1 | 0.991333 | 2.814367 | -1.823033 | +64.7760% | +0.003657345 | 0 |
| 118 | 10 | 0.998517 | 2.327390 | -1.328873 | +57.0971% | +0.009917617 | 0 |
| 120 | 10 | 1.420753 | 2.782593 | -1.361840 | +48.9414% | +0.006297135 | 0 |
| 123 | 10 | 1.926143 | 2.711347 | -0.785203 | +28.9599% | +0.010146142 | 1 |
| 128 | 10 | 1.095560 | 2.670127 | -1.574567 | +58.9697% | +0.009183162 | 0 |
| 133 | 10 | 1.365143 | 2.512383 | -1.147240 | +45.6634% | +0.013908342 | 0 |
| 135 | 10 | 1.241110 | 2.623083 | -1.381973 | +52.6851% | +0.007380602 | 1 |
| 139 | 10 | 1.568467 | 2.712883 | -1.144417 | +42.1845% | +0.009798122 | 1 |
| 140 | 10 | 1.654653 | 2.919847 | -1.265193 | +43.3308% | +0.002096774 | 0 |
| 146 | 10 | 2.468753 | 2.500237 | -0.031483 | +1.2592% | -0.037648614 | 6 |
| 147 | 1 | 0.992933 | 2.768033 | -1.775100 | +64.1286% | +0.019935930 | 0 |
| 149 | 10 | 2.159273 | 2.533650 | -0.374377 | +14.7762% | +0.008730041 | 0 |
| 151 | 10 | 1.040437 | 2.580103 | -1.539667 | +59.6746% | +0.009286992 | 0 |
| 155 | 10 | 1.694797 | 2.654870 | -0.960073 | +36.1627% | +0.012512956 | 0 |
| 158 | 10 | 1.397127 | 2.385000 | -0.987873 | +41.4203% | +0.010450375 | 0 |
| 162 | 10 | 2.093380 | 2.693243 | -0.599863 | +22.2729% | -0.016537396 | 2 |
| 163 | 1 | 1.009167 | 2.346933 | -1.337767 | +57.0006% | +0.016972632 | 0 |
| 174 | 10 | 1.410557 | 2.565967 | -1.155410 | +45.0283% | +0.013761880 | 0 |
| 175 | 10 | 2.073967 | 2.482293 | -0.408327 | +16.4496% | +0.003259182 | 0 |
| 178 | 10 | 2.968290 | 3.125580 | -0.157290 | +5.0323% | +0.003581384 | 3 |
| 181 | 10 | 2.544657 | 2.605950 | -0.061293 | +2.3521% | +0.000763750 | 5 |
| 183 | 10 | 1.542563 | 2.413490 | -0.870927 | +36.0858% | +0.005755684 | 2 |
| 188 | 10 | 1.318807 | 2.454183 | -1.135377 | +46.2629% | +0.010190681 | 0 |
| 193 | 19 | 2.103512 | 2.396649 | -0.293137 | +12.2311% | +0.001779909 | 3 |

## Main-text corrections for root

1. Section4.1 currently correctly states435 outer cycles and1770 non-test cycles, but omits actual proxy counts. Add48 outer and146 non-test proxies, and preserve the warning that these are grouping proxies, not certified independent experimental runs. The source table's435 denominator is correct. No literal23 or30 Hydraulic count currently occurs in Sections4.1–4.2; HAR's30 volunteers is correct and must remain.
2. Section4.1's907 fitting,223 internal-development and640 other cycles are arithmetically consistent. The certified model uses1130 cases for fitting/internal development; 640 are excluded from that use, but the old non-test universe remains1770. Do not replace1770 by1130 when checking outer isolation.
3. Section4.2 correctly gives11 statistical+6 spectral outputs per physical channel:14×17=238 Hydraulic features and28 blocks;9×17=153 HAR features and18 blocks. No23-dimensional or23-block representation should be introduced. Hydraulic slope uses a centered−.5..+.5 cycle coordinate; HAR uses centered sample seconds. FFTs operate on demeaned raw float64 signals, return float32 block outputs, and float32 center/scale arithmetic precedes float64 partial evaluation. Feature reductions/FFTs themselves are not performed entirely in float32.
4. Section4.2 wording "Blocks sharing a quantile or FFT calculation are revealed together" is ambiguous: alloutputs within each statistical/spectral block are revealed together; separate blocks are not a merged DAG. Prefer "Each block reveals together the outputs obtained from its shared quantile or FFT calculation." A statistical block and the spectral block for the same signal remain separate, and each recomputes its own mean.
5. Primary frozen confirmation uses complete case435 timing, unlike HAR's900 timed windows. The Hydraulic late-tail evidence is modest: tight/full P99 4.813808/4.843356ms. Avoid converting its mean-time reduction into a hard real-time claim. Identify this as frozen-method external confirmation given earlier prepared-feature exposure.

## Traceability

Outcome SHA256: `5ceb15f39dd80e88da0dbcb5ea6be5003a48f5055c18e8e64ff9a4f03ae6be6e`.
JSONL SHA256: `815286be1201cfc0ea2a3060e023fb2a53bdaf56851c5fb37ab96a9efae91bf1`.
Unified lock SHA256: `b5efeccb191b682c415e6395960213ecd586806df61517b30685565b7098f284`.
No frozen script or manuscript was modified by this review.
