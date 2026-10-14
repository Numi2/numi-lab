# Synthetic wound-lip traction: full-horizon replication plan

Purpose: one matched, deterministic solver-stability screen on the existing
synthetic wound coupon. It extends the trace-guided candidate from its accepted
seven-step prefix to the complete 60-step load horizon. It is simulation-only;
it does not model a needle, suture thread, robot arm, calibrated skin, or a
clinical procedure.

## Question and prediction

With the same candidate executable and solver policy in both arms, does the
60-step paired-bite traction trajectory complete every unchanged native gate
and reduce the minimum mean top-lip gap by at least 10 micrometres relative to
the matched zero-force trajectory?

The loaded arm uses 10 mN per bite site with the fixed 60-sample half-cosine
ramp. The model predicts a treatment-minus-control minimum-gap difference in
`[-0.600 mm, -0.010 mm]`. The primary prediction is refuted if a valid pair
falls outside that interval. A nonlinear solver stop, missing sample, crossed
lip, determinant or mass failure, or residual-gate failure makes the pair
inconclusive; it is retained and is not retried.

## Frozen implementation and protocol

- One fixed 30 x 24 x 1.5 mm homogeneous coupon, 16 mm incision, 0.6 mm
  initial gap, 24 x 8 x 2 tetrahedral grid (2,304 active tetrahedra), and the
  same uncalibrated neo-Hookean material.
- Paired control and treatment start from the same authored state. Only
  per-site force differs: 0 N versus 0.010 N, applied to the same two bite
  patches with the same full-horizon cosine60 schedule. Run control first,
  then treatment.
- Both arms use the trace-guided native Metal candidate: seven Newton
  iterations, a maximum of 20 FGMRES columns, scheduled forcing amplitude
  0.015625, and forcing floor equal to the equilibrium tolerance. Timestep,
  contact, mesh, material, final `1e-4` mixed-volume gate, determinant gate,
  lip-order gate, and mass gate remain fixed.
- Run the owner probe for all 60 samples at 1 ms each. No diagnostic truncation,
  interpolation, solver-gate relaxation, or wall-clock performance claim is
  allowed. Use the same source snapshot, executable, dylib, metallib, and
  material for both arms.
- Maximum one trajectory per arm. The only experimental unit is this
  deterministic synthetic coupon; its 60 samples are repeated measures, not
  independent units. The seven-step trace-guided prefix is prior evidence, not
  a replicate or a checkpoint for this run.

## Acceptance and limits

Require all 60 samples, all 2,304 tetrahedra, minimum Jacobian above 0.20,
nonnegative signed lip gaps, relative mass error at or below `1e-5`, and the
unchanged maximum mixed-volume residual at or below `1e-4` in both arms. The
preregistered stricter `5e-5` volume target is reported separately as a
secondary diagnostic; it cannot replace the native acceptance gate.

No population inference is made. A complete accepted trajectory would show a
solver-valid response for this model only. It would not demonstrate biological
skin behavior, needle passage, tissue tearing, thread transport, Franka or
dVRK execution, healing, clinical capability, or the requested 100x speedup.

The v2 machine-readable plan, preflight, instrument calibration, build binding,
raw samples, full logs, analysis, and checksums are sealed in the Mac mini
notebook before either arm runs.
