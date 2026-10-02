# HAR training-only preparation

Status: training-only preparation has now completed after the root agent recorded that the Hydraulic certificate feasibility gate passed and explicitly authorized HAR extension. No predictor was fitted and no test member was opened. The design below preceded source loading; the execution and verification record is appended at the end. The project protocol controls later research decisions.

## Source and access boundary

Use a fresh official UCI dataset 240 source through the root-owned `research/download_har.py`. The download manifest supplies the immutable F-drive ZIP path and SHA-256. The loader checks that checksum and reads F only; caches, scripts, manifests and any interpreter bytecode stay on D. Existing caches and source files are never overwritten.

The official archive can expose `UCI HAR Dataset/train/...` directly or contain an inner `UCI HAR Dataset.zip`. For the latter, the inner ZIP is read as compressed container bytes in memory. This necessarily touches the container containing encoded test entries, but the loader never opens, decompresses, decodes, or parses any test member. It writes no inner ZIP or test cache. Source hashing similarly reads opaque archive bytes without interpreting test contents. ZIP directory listings are metadata.

The exact content whitelist contains eleven training members: `train/subject_train.txt`, `train/y_train.txt`, and the nine `train/Inertial Signals/*_train.txt` files. Training CRC checking occurs as these members are read. No `test/*`, `X_test`, `y_test`, `subject_test`, or test inertial member is opened. Do not run a whole-archive `testzip()` during this preparation if the intended boundary is that test members remain unopened. The loader records all opened members in its preparation manifest.

Official source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones. Released inertial windows are author-preprocessed data, not untouched prospective sensor acquisition.

## Arrays and block definitions

Expected official training arrays: 7,352 windows from 21 subjects, each with 128 readings at 50 Hz. The nominal window duration is 128/50 = 2.56 seconds; the elapsed time between the first and last recorded sample is 127/50 = 2.54 seconds. These quantities must not be conflated.

The stable signal order is total acceleration X/Y/Z, body acceleration X/Y/Z, and body gyroscope X/Y/Z. Acceleration uses the released g units; gyroscope uses rad/s. Raw cache shape is `(7352, 9, 128)` with float64 values. Labels are mapped from source 1..6 to 0..5, and subject IDs and zero-based original training-row positions are retained.

Each signal supplies two independently executable blocks: eleven statistics followed by six spectral summaries. There are 18 blocks and 153 final coordinates, with widths `[11,6]` repeated nine times. Block indices 2j and 2j+1 belong to signal j. The loader saves block offsets, feature names and coordinate-to-block mapping.

The statistical block retains mean, population standard deviation (`ddof=0`), min, max, q25/q50/q75, first, last, least-squares slope and RMS. Slope uses centered physical seconds `t[k]=(k-63.5)/50`, with `sum(x*t)/sum(t*t)`. Its units are g/s for acceleration and rad/s² for gyroscope. It is not the previous Hydraulic slope per 60-second cycle. Quantiles use the same NumPy default interpolation as the existing RawBlocks operator.

The spectral block centers the window by its own mean, performs `rfft`, and forms squared absolute coefficients. It retains `log1p(total_power/128²)`, power-weighted frequency centroid, frequency spread, normalized spectral entropy, dominant frequency, and power fraction at frequencies no greater than one-third of Nyquist. The FFT frequency grid is 0..25 Hz with 50/128 = 0.390625 Hz resolution. Centroid/spread/dominant are Hz; entropy and the fraction are dimensionless. Log power is a numeric log transform of released-unit squared power, not a new physical sensor measurement. Zero-power windows produce zero centroid/spread/entropy and select the zero-frequency bin as dominant.

These definitions reproduce the earlier operator principles while changing the physical time base. Computing all 153 training coordinates is offline preparation, not a deployment latency measurement. Later online comparisons must call the same operators only for the blocks actually required by the frozen model or partial executor. A native full baseline computes the union of blocks containing features used by its fitted trees, not a deliberately excessive set of unused blocks.

## Whole-subject development splits

Before inspecting model outcomes, use seeds 17, 29 and 43. For each seed: sort the 21 training IDs, apply `numpy.random.default_rng(seed).permutation`, allocate the first 17 subjects to fitting and the last four to development. Save the full permutation and memberships. Seeds do not share one cherry-picked split.

The partition depends on subject IDs and the fixed seeds, not labels or predictive performance. Every window of a subject has the same owner within a seed; fit and development indices are disjoint and cover all official training rows. This prevents same-subject overlapping windows from crossing the inner split. Subjects can have different roles across seeds because these are separate fitted systems. The preparation script does not verify official outer test-subject membership, since opening `subject_test.txt` is deferred; it explicitly records that this check has not yet been performed.

For each seed, fit scaler means and population standard deviations on its 17 fitting subjects only. Replace scales below 1e-8 with one, and save the 153-coordinate center/scale arrays. Development windows cannot influence scaling. `features_train` remains unscaled so that tree models may use physical-unit features or a consistently frozen fit-only transform; either choice must be the same for that seed's full and partial executor. The loader fits no predictor and makes no claim about predictive competence.

The four development subjects can support tree-model competence checks, implementation verification and development choices under the project protocol. They cannot be relabeled as independent test subjects or support a post-selection confirmation claim. Later fitting on all 21 training subjects is a distinct explicitly recorded model, requiring new scaling/order metadata; the current cache does not silently change its split.

## Saved training cache

The NPZ contains `raw_train`, `features_train`, `labels_train`, `subjects`, `original_train_row`, schema arrays and a JSON Unicode scalar named `splits`, readable with `allow_pickle=False`. For each seed it contains `fit_indices_seed17` / `development_indices_seed17`, corresponding subject arrays, and `scaler_center_seed17` / `scaler_scale_seed17`, with equivalent keys for 29 and 43. No test array is present.

The preparation manifest records source/script/cache hashes, parsed training-member names, shapes, units, permutations and all test-read counts. A fresh time-and-UUID filename is generated by default. An existing user-supplied output path causes an error rather than replacement. The anticipated cache is comfortably below one gigabyte; preparation checks at least two gigabytes free on D before computation.

After feasibility passes and the source download manifest exists, run the bundled Python with `-X utf8` and a D-drive `PYTHONPYCACHEPREFIX`, using `research/prepare_har_training.py --manifest <D-drive download manifest>`. The script has no download or training side effects. Its only dependency for preparation is the already-installed NumPy package under the original project's D-drive environment directories; it imports no Torch wrapper.

## Outer test release conditions

Keep all official test member contents unopened while model selection, feature definitions, output rule, conservative certificate implementation, primary delta/endpoint, feature ordering, comparison methods, training seeds and timing protocol remain changeable. Unseal only after correctness and feasibility pass, the final method and run manifest are frozen, and the root agent explicitly initiates the authorized confirmation stage.

The separate confirmation loader then verifies outer subject disjointness and runs the frozen full/partial pipelines. It must not alter features, scaling, model selection, certificate tolerance or schedules after viewing test results. Label-based performance checks and certificate verification occur outside the online timer. Subject-cluster uncertainty respects correlated and overlapping windows; timing repeats are not extra participants or fitting seeds. This training-only loader does not implement or authorize that confirmation loader.

## Executed training-only preparation, 2026-10-02

The root agent authorized execution after its recorded Hydraulic v3 feasibility decision. This preparation task did not independently re-audit that decision. Before execution D had 142,346,022,912 bytes free and F had 3,631,720,345,600 bytes free. OMP/MKL/OpenBLAS/NumExpr were restricted to one thread; interpreter bytecode and temporary paths were assigned to this project's D-drive temp directory. No fitting or concurrent deployment timing was performed by this task.

Authorized source: `F:\AcademicData\ASC_CertifiedProcessing\raw\uci_har_240__20261002T051056Z_9247235203.zip`, 61,005,872 bytes, filesystem ReadOnly. SHA-256 was checked against `datasets/raw_source_manifest_20261002T051056Z.json`: `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. The outer ZIP contains two members and supplies the nested dataset ZIP. The source was only read and was not modified.

The successful loader produced:

- Cache: `D:\MLWork\ASC_CertifiedProcessing_20261002\cache\har_train_only__20261002T051217Z_533d2ce1.npz`.
- Preparation manifest: `D:\MLWork\ASC_CertifiedProcessing_20261002\datasets\har_training_preparation_manifest__20261002T051217Z_533d2ce1.json`.
- Cache size: 39,133,545 bytes. Verified SHA-256: `a23c485d48a8eca5f2f37648abe0400eebf535837998892b2ab93ab252c1756c`.
- `raw_train`: `(7352,9,128)`, float64; `features_train`: `(7352,153)`, float32. Raw signals and features are finite. Six training class counts in encoded order 0..5: `[1226,1073,986,1286,1374,1407]`.
- Training subject IDs: `[1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]`.

| Seed | Fitting windows | Development windows | Development subjects | Development class counts, 0..5 |
|---|---:|---:|---|---|
| 17 | 5898 | 1454 | 17,28,19,29 | 220,188,179,269,295,303 |
| 29 | 5845 | 1507 | 16,30,28,27 | 227,218,199,273,296,294 |
| 43 | 5963 | 1389 | 7,25,1,6 | 283,220,202,215,237,232 |

Each split has exactly 17 fitting and four development subjects, all six classes in both sets, no shared subject or row, and complete coverage of training rows. Recomputing the seeded permutations reproduced every saved subject assignment. Every saved center and scale was bitwise identical to a fresh recomputation using only that seed's fitting rows. No development features or labels enter those scaler estimates.

Cached features for the prespecified positional checks 0, 3676 and 7351 were rebuilt from D-cached raw windows using all 18 online block operators and were bitwise identical to the cached 153-coordinate vectors. The cache was opened with `allow_pickle=False`; block mappings and shapes were checked. All stored keys were inspected: no test array exists. The preparation manifest lists exactly eleven opened training members and zero opened test members, with zero test signal/outcome files parsed. No test cache or predictor was created. Official outer subject disjointness remains deliberately unchecked until the separately frozen confirmation stage.

These checks establish successful training-data preparation and operator/scaler consistency. They do not establish predictive competence, computational benefit on HAR, or certificate correctness on its unopened test set. Heavy preparation and verification ended before the independent timing worker was notified to begin its measurements.
