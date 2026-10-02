# Recorded development implementation repair

After the preserved v1 prototype, the locked .025 development setting saved feature blocks but remained about 15% slower than same-backend exact full processing. Independent 12-case profiling placed tree-bound maintenance at about 1.0–1.7 ms and the certificate at .16–.31 ms. The root authorized one fixed checkpoint implementation repair before the new timings.

The classifier, training-only gain/cost order, scaler, five deltas, output/certificate variants, native fallback and two full controls are unchanged. Checks occur after fixed batches of four blocks, at 4, 8, 12, 16, 20, 24, 28. Each batch computes its blocks, then updates the precompiled union of affected nodes and ancestors in children-before-parents order. Bounds and certificate execute in one compiled call. The policy intentionally performs its first batch before checking; it does not assume model-specific empty bounds are the whole simplex.

New outcomes are written only under results/batch4_repair. V1 output and source snapshots are retained. Eight predictor-fitting cases supply separate diagnostic component profiles. The same fixed first24 and all223 internal-validation cases then assess numerical correctness and complete time. These are development implementation diagnostics and cannot serve as independent confirmation.
