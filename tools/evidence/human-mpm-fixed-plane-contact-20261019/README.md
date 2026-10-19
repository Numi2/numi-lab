# Fixed-plane MPM boundary and accepted reaction rollback

This increment keeps Matter's existing GPU MPM, FEM, vascular state, and accepted-step owner. It adds an exact one-sided normal-velocity constraint for eligible immutable free-slip planes. Grid basis nodes are not material particles; a separate GPU certificate checks material-particle endpoints and affine swept paths before publication. Other contact topologies retain their existing geometric contact path.

The explicit blood material interface is free slip, non-adhesive and non-restitutive. Density, viscosity, numerical bulk penalty, and constitutive validity limits are unchanged. This is not a calibrated no-slip blood-wall or low-shear pooling model.

The existing frame-reaction checkpoint is now also used for ordinary transactions. Failed environments restore the preceding accepted reaction ledger, including rejection by the external accepted-step owner. No CPU per-step physics or new simulator was added.

## Native evidence on SSH Numi / Apple M6

- acceptance/: four zero-gravity and four gravity-loaded steps, 0.25 ms each, with a late rejection and same-context retry in each case. Constant particle mass/reference volume, finite normal constraint impulses, equal/opposite proxy reactions, strict particle clearance, and exact reaction-ledger rollback passed.
- compatibility/: two 0.25 ms steps in the existing integrated scene with bilateral optic-nerve FEM, regional skin/fat, aortic wall and prostate enabled; experimental organ batch and MPM disabled. Clean/retry accepted FEM rows and physiological/surface CSVs were byte-identical to the published prior runtime. All eight rollback checks passed.
- scalar-check.json: 2,000 complementarity/JVP checks and 1,000 negative-RHS sign checks. Algebra only.
- The 96-step floor replay accepted 37 steps (9.25 ms) then rejected step 38 with Matter status 10. It did not reach its planned late-rejection/retry case. This is retained failure evidence, not sustained qualification.
- retained-reaction-rollback-failure/ preserves the preceding failing gravity rollback test.

The exact executed flattened Metal source and host runtime are retained under executed-source/, with compiler/link invocations and hashes. mapping-receipt.json verifies byte-exact reconstruction of the flattened Metal candidate from changes to the existing owning source files. Test compile dependency identities are retained separately. Native source pins differ from the publication header only by the narrow exact-material guard merge and preservation of previously published options.

## Limits

These are short numerical tests, not sustained blood pooling, anatomical vessel through-flow, a full-body tissue qualification, physiological validation, or clinical validation. The acceptance fixture starts a particle one grid cell above the floor and specifically tests grid-support boundary behavior. The 50 mm blood-grid cell is underresolved for microlitre pooling geometry. No visual verification was performed. General skin/fat mechanics remain regional; the registered organ batch has a separately retained second-step line-search failure.
