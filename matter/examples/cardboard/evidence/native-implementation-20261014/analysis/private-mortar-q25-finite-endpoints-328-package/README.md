# Finite endpoint supplement

This is a fail-closed supplement to the sealed 326 Q25 paired-boundary patch. Exact endpoint equality is not sufficient when either endpoint is nonfinite: IEEE infinity can compare equal to itself. The shared segment, paired-world-segment, and mapped-vertex predicates now reject nonfinite coordinates before equality or quantized comparison. Finite geometry behavior is unchanged.

The CPU suite passes 23 checks, including the infinity and NaN controls. The patch is only a predicate hardening change; it does not change the CCD or suppress any finite contact. No GPU/native run was made.

Apply after the 326 patch with `git apply finite-endpoints.patch`. See `manifest.json` for the exact 326 package and file bindings.
