# Tangent-path audit: verified current implementation

Baseline 245a7610, SSH Numi, Apple M6. No visual verification. Default constitutive equations and solver acceptance are unchanged.

The alleged split in canonical selection is absent in this revision. The full stateless evaluator and columns 1-8 call evaluateTangentVectorWithCandidate, which already owns canonical specialization and precision fallback. Added comments documenting that invariant and regression coverage for all three canonical source materials with the tag on/off: nine affine and six fixed-seed random nodal directions, two environments and rotated material frames. All six cases pass, maximum relative cached/direct action error 8.5021983284e-7. Serial/parallel cache entries remain bitwise equal. The full material test passes, including stateful plastic histories and late-rejection rollback.

Repeated tetrahedral reconstruction in the nine-column builder is real. Completed-Newton latches already skip converged tails. Shared 15-state projection/A/B/factorization is already wired into the ordinary builder in 245a7610; it does not require lifting coupon diagnostic restrictions. However the actual uploaded Human world has 20 stateless materials and zero eligible local systems. Its stateful finite-difference projection fallback is dormant.

Added NM_FEM_TANGENT_CENSUS=1: upload-time, CPU metadata census using existing compiled descriptors and current source object/tetrahedron ranges. Default off; strict 0/1. It emits material identities, active source tetrahedron counts, state/projection eligibility and scalar bytecode instruction/EXP/LOG counts. This is not a per-step CPU implementation. It adds no simulation buffers, physical state, material format or runtime equations.

The census and unchanged default one-root Human runs both exit 0, pass FEM/fascia terminal audits and have exactly equal retained physical endpoints. Five dynamic multiphysics/rollback tests pass. The run is only 0.125 ms of an unbalanced release transient; neither timing nor contact state qualifies sustained supported rest.

Per-scalar expression CSE already exists. Across-bundle/direction reuse remains a separate optimization. Source-weighted bytecode counts are not GPU instruction counts or a decomposition of the earlier 8.06 seconds. Projection iterations, actual cache consumers/applications, per-family GPU durations, register spills/occupancy and memory traffic remain unmeasured. No CSE speedup, cooperative-group speedup, lagged tangent or assembled-operator result is claimed.

Next useful experiment: exact bundle-level reuse for the dominant actual stateless families, preserving expression operation order and compensated arithmetic. Compare one-thread full-map and cooperative construction with cache/direct random actions and accepted-step parity, then time matched warmed full caches. Keep default canonical off because its earlier integrated endpoint comparison failed. Keep the existing coupon-only diagnostic admission unchanged. No 0.1x result is claimed.

Provenance: invocation source_head inherited a stale aadcd68e value from the 073 template. See PROVENANCE-CORRECTION.json; actual execution baseline is 245a7610 with the recorded working-source hashes.
