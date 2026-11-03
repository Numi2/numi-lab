# Physical certificate and scaled-restart recovery

The GPU regression fails on the previous shader and passes after the fix. When the unscaled certificate rejects a row-scaled Krylov result, the next restart now tightens its internal scaled reference to the remaining scaled norm and reactivates work. The original raw RHS norm, physical forcing target, nonlinear tolerances and final accept/reject checks stay unchanged. Certified completed cycles still emit zero work.

An explicit default-off coupled-FEM/vascular row-scaling mode admits the existing checked selector-6 domain without depending on Human attachments registered after runtime initialization. It applies the original physical right preconditioner to S^-1*v, yielding S*A*M^-1*S^-1. Pressure and coupled tail coordinates stay identity. It reuses existing RHS scratch, prohibits unsupported auxiliary solvers, and excludes prefix/flat cached graphs whose input bindings have not been extended. Legacy admission is preserved. EW is admitted only with relative forcing and mandatory full physical certification.

Native comparison on SSH Numi:
- Before the restart fix: the first physical linear residual remained 0.248174 through all restarts.
- After: it decreased to 0.0485925. Later Newton solves also performed effective restarts.
- Both full-scene trials accepted ZERO physical steps. The nonlinear residual remains about 0.71 and skin/fat geometry still severely limits updates. The scaling mode is not a convergence or performance qualification and remains opt-in.

Focused GPU checks cover inverse conversion, identity tails and pressure, a sixteen-decade diagonal scaling range, a noncommuting nonsymmetric oracle, the failed-certificate restart, normalized basis initialization and completed-cycle suppression. The full existing kernel suite also passed; full FEM operator qualification is explicitly not claimed by that suite.

The initialization-only fixed-cavity candidate was held identical in these two native trials. It is not the accepted anatomy scene. Exact binary, library, metallib, dirty source and plan hashes are retained in the manifests. Native outputs remain on Numi; duplicate immutable outputs were hard-linked only after full SHA-256 equality checks, retaining every original evidence path and byte.
