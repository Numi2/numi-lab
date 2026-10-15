# Source125 run 419 independent FP64 small-step audit

This CPU-only diagnostic binds the exact run419 source125 Newton48 rich capture, the 422 fresh production operator replay, and the source125 residual endpoint sweep424. It recomputes the independent 15-state response at exact captured 1/64 high/low endpoints, then at smaller unquantized FP64 perturbations with the same accepted history.

The native 1/64 endpoints provide a direct model check: captured stress is compared with independent projected stress and the endpoint residual secant is compared with a separately assembled FP64 response. Smaller-epsilon values are independent constitutive evaluations, not native outcomes. The separate 424 report is the source-bound native smaller-step record and explicitly does not resolve the native derivative/noise question.

All per-environment status codes are treated as propagated stage status; they do not mean every tetrahedron independently failed. The private status5 record identifies one first failing tet at a different stage/iteration.

No tolerance, material parameter, accepted history, or physical state is changed. This does not qualify the failed coupon trajectory or a global tangent. See report.json and manifest.json for hashes and limits.
