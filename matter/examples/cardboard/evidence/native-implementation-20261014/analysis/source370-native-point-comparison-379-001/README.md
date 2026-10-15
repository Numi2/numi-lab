# Native point replay 379: source-370 tet 1820

This package adjudicates one rejected source-370 candidate material point. It does not replay a Matter transaction, and it does not qualify the coupon or a material recipe.

Run the comparison with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/source370-native-point-comparison-379-001/compare_point.py
```

The script first verifies the native point-run receipt and its unchanged inputs, then verifies hashes for the source-370 compiled package, candidate nodes, accepted material-state export, manifest, and medium material asset. It compares the native FP32 high/low deformation words for tet 1820 against the separately retained CPU source transcription. It widens those exact native words and state words to FP64 for an independent equation replay.

The native point checker was built from `source-exact-material-point-112`. Its immutable 370 world was compiled with a 16-iteration local material budget. The point checker separately evaluates the same captured point with private diagnostic descriptors set to 16 and 32 iterations. Therefore budget 32 is a bounded point diagnostic, not the source-370 trajectory or a modified raw compiled package.

For each native-returned state, the script reports the independent FP64 residual and a newly solved, dimensionlessly scaled next Newton correction, using the retained source replay policy and the unchanged `2e-8` correction threshold. These are derived FP64 metrics; the native checker does not export its per-iteration scaled correction or Armijo history. A passing derived correction is not a native certificate. Stress and tangent comparisons are emitted only where the native projector marks the corresponding output valid; they are diagnostic comparisons against a Python FP64 transcription, not qualification gates.

The result reproduces the local outcome split: budget 16 exits after 16 updates with projection failure 7, while budget 32 succeeds after 18 native updates. For budget 16, the independently computed residual is below its residual bound, but the derived next correction is `1.5214879e-7`, above `2e-8`. For budget 32, the derived next correction is `2.3454478e-9`; the FP64 source replay converges in 17 updates, while native takes 18. Native and source-transcribed F high/low words match bit-for-bit for this point.

The comparison leaves unresolved the native projector's internal correction sequence and Armijo choices, full-run behavior after earlier accepted steps, and whether the point-local 32-update diagnostic improves complete coupon execution. No physical or calibration claim follows from this single-point result.
