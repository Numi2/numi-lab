# FP64 capture reader: explicit unresolved report values

This supersedes source166-001 after native469 replay001 computed and retained its
NPZ but failed strict JSON serialization on a nonfinite reference intermediate.
The failed command, stderr, receipt and partial arrays remain unchanged. The
constitutive equations, projection iteration policy and comparison thresholds
are unchanged. Every NaN/Inf JSON value becomes null with its exact field path
and original value in `nonfinite_diagnostic_fields`; raw NPZ arrays are retained.
This is a reporting fix and cannot turn an inconclusive row into a passed gate.
