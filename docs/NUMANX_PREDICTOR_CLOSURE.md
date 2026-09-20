# Human predictor and support closure

NumanX now certifies Matter candidates around the same free Human velocity
that the subsequent Stand step uses:

```text
A0 = M(q0) + armature + h D
v_free = v0 + h A0^-1 (tau_source - bias_source)
v_candidate = v_free + delta_v
q_candidate = integrate(q0, h v_candidate)
reaction_force = A0 delta_v / h
```

The former candidate used `v0 + delta_v`, then Stand added the source muscle,
gravity, and bias contribution afterward. Consequently its geometry could
differ from the geometry used to certify the tissue/contact solve. The owning
Metal Stand prefix now computes `v_free` and `A0` in private storage without
advancing q or marking a step complete. The actual Stand step retains sole
authority to consume the staged reaction and advance the live Human state.
The old accepted-v checkpoint remains separate and unchanged for rollback.

The host borrowed pass is version 5 and binds the predictor buffer's identity,
GPU address, size, and non-aliasing with live/checkpoint/proof authority. The
pointer-free root/publication ABI remains version 4. Recompile native clients
of `MetalNumanXHumanMatterPass`; a version-4 pass is rejected. The public
`mrnx_bridge_v1` C interfaces and their Swift callers keep their existing ABI.

Support witnesses now obtain total point velocity from exact candidate body
linear/angular velocity. The Newton generalized vector contains a correction,
so applying J to that vector alone omits free Human motion. For residual
`J^T impulse - A0 delta_v`, the restoring contact term enters the Newton
operator with a positive `J^T D J` sign. The normal term includes the derivative
of positional stabilization. This is not a new friction/material calibration.

Every support evaluation within a Newton root uses the same accepted history.
The candidate history is written separately and becomes accepted only through
the existing commit/rollback transaction. The early nonlinear convergence test
now also satisfies the final-assembly residual scale; it no longer freezes an
iterate that the final certificate will reject. Publication tolerances were
not changed.

## Regression evidence

On macmini, Apple M4 Pro:

- The 160-DoF/161-coordinate probe exercises a nonzero source acceleration.
  Its zero-correction candidate and final Stand step have zero measured q/v
  difference. The old `v0` candidate is a failing negative control at the
  probe's 1 ms timestep. Arbitrary-correction kinematics, analytic attachment
  Jacobians, frozen A0, malformed passes, and alias rejection are also checked.
- The support probe uses total velocity -1 m/s with delta velocity -0.25 m/s,
  a nonzero accepted impulse history, finite-difference force derivatives,
  and exact history rollback. Its Newton action is 2.4, with finite-difference
  error below 3e-4.
- Owner, root publication, adapter, moving-attachment, and both legacy/authored
  full-body tests exercise the native transaction. Mixed MPM/FEM and monolithic
  multiphysics tests cover the shared convergence change.
- Both NumiBrain full-body tests pass through the existing C bridge: eight
  accepted roots at 100 microseconds, with rejection/retry and learning checks.

Detailed logs, immutable source revisions and binary/input SHA-256 values are
recorded in the Human repository's `Docs/NUMANX_PREDICTOR_CLOSURE.md` receipt.
Earlier package-admission evidence does not establish the corrected physical
candidate relation retroactively. The source joint-equality operator remains
unimplemented in the coupled path. Standing, walking, anatomical calibration,
and performance remain unqualified.
