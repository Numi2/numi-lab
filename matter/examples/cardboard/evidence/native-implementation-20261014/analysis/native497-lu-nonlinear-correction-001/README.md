# Native497 warm-CGLS nonlinear residual replay (CPU only)

Evaluates the captured control 20 / Newton 63 CGLS candidate at alpha 0, 1/128, ..., 1 with the strict compensated FP64 material reference. It reconstructs trial velocity/displacement from captured node words and LU direction x, recomputes paper projection/stress and glue stress, and separates mass and constitutive-force residual increments anchored to the native center residual.

No GPU, native rerun, source edit, or full-coupon qualification was performed. See report.json for hashes, strict gates, curve and limits.
