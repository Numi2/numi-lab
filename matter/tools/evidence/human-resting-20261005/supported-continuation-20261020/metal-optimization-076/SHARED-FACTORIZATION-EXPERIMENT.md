# Shared local material factorization: preregistration

Question: can the existing compensated 15-state implicit tangent cache reuse direction-independent work without changing its constitutive equations or accepted state?

Prediction: a guarded batch cache will evaluate the shared 225-entry state Jacobian and 135-entry stress/state derivative block once and factor once, solving nine deformation-direction right-hand sides through that factor. Existing cache projection is already shared; no projection speedup is claimed. No explicit inverse, persistent factor reuse across changed linearizations, new state owner, or lower precision is permitted.

Eligibility: retain the existing exact 15-state compensated/scaled-correction policy and prove all shared bytecode ranges direction-free (NM_EXPR_DF absent; base NM_EXPR_RATE allowed). No dissipation or learned material admission expansion. A proof failure retains legacy evaluation. New runtime flag defaults off.

Native checks before any performance claim: compare projection high/low words, nine cached tangent columns and random-direction actions against existing evaluation; pivoted multi-RHS local residuals and reference solves; singular/breakdown handling; nonfinite/invalid material rejection; unchanged accepted-state/rejected-step history. Retain existing precision and acceptance tolerances. Compile and run on SSH Numi only, one GPU job at a time.

Scope: checked-in Human material sources have no state declarations; the eligible source-fit 15-state materials are paperboard. This feature's local work reduction is not a forecast or measurement of Human performance. Confirm actual uploaded-world eligibility before attributing gains. Full Human throughput and supported-rest qualification remain separate.

## Path clarification before first integrated batch test

The existing optional projection_reuse_diagnostic cache shares the projected state; the ordinary nine-column cache does not share stateful projection. New batch construction therefore changes projection count from nine to one for the ordinary path and one to one for the diagnostic path. The original statement above was too broad. A/B/factor sharing applies to both. Preserve the untouched serial cache as the nine-column reference.
