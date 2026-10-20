# Stateful tangent factorization eligibility audit

**Repository base HEAD:** ae35df813006f30ab8cace588465a9fdbe19f062 (2026-10-20). This is a source-level working-tree audit, not a native run or performance qualification.

## Dependency proof

The compiler constructs the implicit-state Jacobian roots from authoredResidual differentiated with respect to each candidateState in matter/src/constitutive_impl.cpp:1611-1629. It separately constructs stress/state roots by differentiating each stress component with respect to candidate state (:1632-1641). For n states these are exactly n*n A programs and 9*n B programs, with B stored component-major as component*n + state; at n=15, that is **225 A programs and 135 B programs**. The directional residual roots are compiled separately (:1618-1624, :1678-1694), so the A/B roots are not built from tangent-direction seeds.

The symbolic directional builder creates ExprKind::deformationDirection for directional deformation and deformation-rate inputs (constitutive_impl.cpp:613-629); bytecode compilation lowers that node to NM_EXPR_DF (:959-962). NM_EXPR_RATE is the current deformation-rate value, not a directional-rate seed (:964-967); there is no separate DFDOT opcode in the shared bytecode enum (matter/include/numi/matter/shared.h:110-135). Therefore the reusable-factor eligibility check can scan the existing compiled A and B instruction ranges for NM_EXPR_DF: absence in both ranges proves those programs do not depend on DF/DFDOT tangent direction. The existing offsets/counts identify the ranges (); no new metadata format is required. This is a bytecode dependency proof, not a claim that all materials satisfy it.

## Scope and limits

The checked-in Human .nmatter materials have no state declarations. The checked-in 15-state examples are Haj-Ali bilinear paperboard fits (hajali2009_liner_hill_bilinear_fit.nmatter and hajali2009_medium_hill_bilinear_fit.nmatter), not Human materials. Thus this audit establishes a compiler-side eligibility test for a future or explicitly selected stateful material; it does not establish that any current Human material exercises the 15-state factorization, or that every source law is eligible.

The explicit bilinear-15 opt-in is additionally gated to the expected 28 named parameters and 15 named states, generic implicit integration, and no dissipation or learned model (matter/src/constitutive_impl.cpp:1376-1410; validation guard at matter/src/validation_impl.cpp:1252). Keep those gates: the current B construction differentiates elastic stress roots with respect to state, and must not be generalized to dissipative mixed stress/state derivatives without extending and validating the operator.

## Decision rule

Reuse one projected A factor across the nine deformation-direction right-hand sides only when the selected compiled program has the expected A/B ranges and no NM_EXPR_DF in either range, while retaining all existing material-policy gates. A positive result proves direction independence of those programs; numerical equivalence, solver behavior, and speed still require their own native evidence.
