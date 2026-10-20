# invert3x3 orientation audit

**Repository base HEAD:** ae35df813006f30ab8cace588465a9fdbe19f062 (2026-10-20). Read-only mathematical working-tree audit; no global helper change is proposed here.

## Finding

matter/src/metal/common.metalinc:294-324 computes row cofactors and constructs float3x3(cross(row1,row2), cross(row2,row0), cross(row0,row1)) / det. With the current row extraction and Metal column constructor, this is **A^-T**, not A^-1. For the nonsymmetric shear

    A = [[1,2,0],[0,1,0],[0,0,1]],

it yields [[1,0,0],[-2,1,0],[0,0,1]]; the actual inverse is [[1,-2,0],[0,1,0],[0,0,1]]. The determinant guard does not affect this orientation issue.

## Callers and impact

- common.metalinc:396 (polarRotation) consumes the result as A^-1 in its Newton iteration. The following transpose(inverseRotation) at :404 then uses the wrong update for nonsymmetric deformation; for the shear above the first polar iteration becomes identity rather than the expected rotation toward the polar factor. adaptive.metalinc:200-270 uses this polar result for strain/orientation/quaternion and reconstruction paths; identification_scheduler.metalinc:456,474 uses the derived strain trigger.
- topology_mutation.metalinc:684-714 stores the helper result as the inverse rest matrix. FEM later forms Ds*inverseRest (fem.metalinc:313-343), so a nonsymmetric rest basis can produce Dm*Dm^-T rather than identity.
- common.metalinc:1441-1487 uses it for the plastic state matrix and deformation inverse in specialized plasticity. The stored plasticStateMatrix is full row-major Fp (:1393-1397) and evolves by left-multiplication with the plastic increment (:1522-1527), so it is not guaranteed symmetric. This caller requires a focused nonsymmetric-state check. Its stress/tangent expressions use inverse versus inverse-transpose in distinct places, so correcting only the helper call convention requires a local derivation.
- common.metalinc:981-992 in the canonical neo-Hookean tangent explicitly needs F^-T; the helper's A^-T result matches that use. Preserve this valid caller's semantics in any future repair.
- Other direct callers are inverse-inertia operations (adaptive.metalinc:258, contact.metalinc:13177, and fgmres.metalinc:4819,5134,5400). Those inputs are symmetric inertia matrices, for which A^-T = A^-1, so this orientation difference is immaterial there.

## Test coverage gap

matter/tools/fem_reference_check.mm exercises skew reference geometry and FEM gradients through its CPU cook path (generalReferenceChecks, approximately lines 560-593); it does not invoke the Metal topology-refresh inverse helper or validate polarRotation against a nonsymmetric matrix. No focused native test was found for nm_adaptive_*, topology mutation with a nonsymmetric rest basis, or the specialized plasticity inverse path. Symmetric rigid-inertia coverage cannot reveal the defect. Add narrow nonsymmetric shear cases for polar iteration, topology rest reconstruction, and plastic state before changing shared semantics; the canonical neo-Hookean F^-T case should remain as a positive control.
