# Coupon 392 status-5 CPU replay audit

## Bound evidence

This is a CPU replay of the retained rejected-conditioning candidate nodes for source-392, against the compiled package and source-118 medium material. It does not capture the exact first failing kernel dispatch. It therefore cannot distinguish the material internal-force status-5 writer from final FEM validation, and it does not establish that the rejected-conditioning node state equals the generation that first latched the failure.

The failed native run receipt records `exit_code=2`, `timed_out=false`, and `accepted_steps=0`. Its final status is `NM_STATUS_INVALID_DEFORMATION` (5), environment 0, object 0, index 858, diagnostics `(0.9977478981, 0, 0, 0)`. The tuple is compatible with a failing tetrahedron but has no producer-site field.

## CPU replay result

The replay reads the compiled package and the retained rejected-conditioning node-state binary. It reconstructs production-style FP32 high/low reference deformation inputs, zero accepted history, material parameters, and local deformation rate for tet 858. The independent source-law FP64 projection converged to residual infinity norm `1.1102230246251565e-15`; the local production-float projection also returned valid after 14 iterations with residual `5.9061398350115923e-13`, scaled correction `3.1599135200816031e-12`, and dynamic tolerance `3.9861483003278408e-08`. The resulting FP32 maximum stress was `9,136,496 Pa`.

This replay is evidence that the retained rejected-conditioning candidate can be projected successfully under this reconstruction. It is not evidence about the exact first status-5 generation, and it does not resolve status-site ambiguity or supersede the native failure.

A first attempt using the compressed `.nmatter.gz` source failed with `medium source parse failed`; it is preserved in `replay-attempts.json`. The successful run used the uncompressed, source-118 compiled medium asset.

## Status-site capture requirement

A future read-only observer must run after each status-5-producing FEM dispatch, before rollback or status reset. Global and private status buffers need independent first-event slots. Each captured record must identify stage/domain and preserve the status tuple, microstep, candidate/center context, exact four accepted and candidate node states, tet/material/object descriptors where the index semantics support them, accepted material history, full environment parameter vector, and production F high/low plus rate. Geometry-gate indices must remain labeled as solver iterations unless an in-kernel witness is captured.

No GPU observer run was performed in this audit.
