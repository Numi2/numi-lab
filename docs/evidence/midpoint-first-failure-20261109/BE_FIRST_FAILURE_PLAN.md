# Backward Euler first-failure trace

Using the same frozen 1 microsecond backward Euler binary, shader, physical
fixture and initial state, rerun through zero-based step 3186 with the
existing bounded Newton trace enabled only for that root. Retain raw solver
and candidate diagnostics, per-iteration correction, residual and line
search information available from that instrument. Compare its accepted
prefix and restored physical state with the untraced failure before treating
any trace as representative. A trace that does not reproduce the same
failure, or an unavailable diagnostic field, is an inconclusive diagnostic.
Do not change solver tolerances, iteration budgets, the contact law, or the
step. This is not a force-accuracy or performance result.
