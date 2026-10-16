# Source 213 checked true-residual arena extension

Based directly on immutable source207-buildfix-002. Six changed source paths are bound in source-overlay.json and the 825-entry candidate map is frozen in the snapshot.

Only the true-residual capability receives the N<=8192 allowance. Other Newton-trace and residual-Armijo call sites keep their N<=4096 check. The plan computes exact logical row counts, typed allocation extents including zero-count typed sentinels, each maxBufferLength check, and the aggregate size of all six simultaneous true-residual arrays. The aggregate ceiling 12,058,880 bytes is derived from the maximum supported E=4,N=8192,T=32768,C=4096 non-rigid envelope, compared against the E=1 six-tail alternative. It is not a whole-process memory/peak bound.

The native checker extension exercises the production true-residual measurement/replacement kernel over N=4096 and N=8192 rows with nonzero mechanical and field residuals and exact-zero identity rows. Its operatorValue is supplied by the fixture, so this does not qualify production FEM tangent assembly at N8192. No GPU execution is authorized in this package.
