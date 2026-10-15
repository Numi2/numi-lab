# Free-box angular checker classification repair (overlay 072)

This checker-only overlay is based on the source-071 angular checker whose SHA-256 is `855b816c2f4f814446f382e58bcd0acc58dc6d31543b91c84760309b94dbb11f`. It does not modify the sealed source-071 snapshot, any production Metal kernel, tolerances, or constitutive/contact law.

The 855b checker removed the independent sampled FP64 path oracle and inferred physical sweep clearance from `diagnostics.y > diagnostics.z`. Those fields are the minimum sampled physical gap and the floor; a positive minimum does not mean the conservative CCD completed a swept-clear certificate. In case 213, the rotating mid-path hit had status 6, alpha gap `0.0649999976158 m`, physical minimum sampled gap `1.00387260318e-5 m`, floor `1.00000006569e-5 m`, and alpha certified. The FP64 fixture found a collision at fraction 0.5 with gap `-0.00499999958 m`. Case 223 stopped at the bad assertion before it printed raw diagnostics, so those values must not be attributed to 223. Its production contact source SHA is the same as 213, but the source/build snapshots differ.

The replacement restores the prior 65,536-point FP64 fixture oracle, reports raw native status/gaps before classification, treats a production rejection as `physical_sweep_rejected_or_uncertified` when the alpha path is certified, and continues to the contact/inertia checks even if the path suite fails. A sampled clear oracle is only a fixture cross-check; only production CCD acceptance is labeled `certified_clear`. The existing 855b `A=-dR` residual sign and branch-stability checks remain intact.

Files:

- `cardboard_free_box_angular_native_check.mm`: replacement checker source.
- `checker.patch`: unified diff from `base-855b.mm`.
- `source-overlay.json`: exact hashes, source binding, comparison evidence, and validation status.
- `base-855b.mm`: immutable overlay base copy.

Validation so far: Objective-C++ syntax-only compilation passed against the source-071 headers. No GPU run has been performed for this replacement; native classification and subsequent action results remain pending.
