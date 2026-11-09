# Compensated reacting body and surface integration: incomplete work

This is a research branch checkpoint, not a qualified implementation. No
accuracy-adjusted cost result or spatial convergence claim follows from it.
The prior results and failed calibrations remain historical evidence.

## What is present

* The old synthetic slider effort and its matching contact-wrench subtraction
  are removed together. Admission of the slider midpoint owner is explicitly
  disabled while the physical compensated transaction is incomplete. A
  requested low-word stream is rejected rather than silently ignored.
* The reference-area mesh retains triangle node IDs and reference areas, and
  a positive degree-five seven-point triangle rule has been added. Independent
  FP64 reference implementations can use that rule and an opt-in graded mesh.
  The diagnostic fixture can generate the same grading. The previously
  retained force peaks, impulse curves and thresholds are unchanged.
* The native-trace supervisor and analyzer stream the step array, and the
  supervisor, recovery and cost scripts accept an explicit evidence directory.
  Evidence and system-volume headroom are both checked; existing evidence is
  never overwritten. The historical cost runner rejects graded or quadrature
  input until a matching new reference is qualified.

## Follow-up checkpoint: vertex clearance and transaction scaffolding

Both FP64 quadrature references reject nonfinite or nonpositive triangle
vertex gaps before evaluating each triangle. The regression uses vertex gaps
-1, 10, 10 micrometres and confirms all seven interior samples are positive;
both energy and force paths in the direct reference and the compact force
path reject the penetrated face. This is an endpoint check, not a swept check.
The Metal quadrature path is not implemented and therefore is not yet covered.

The host can build seven compensated reference rows per triangle and a sorted,
node-major incidence table. These buffers are not yet uploaded to or consumed
by the nonlinear solver. The Metal law exposes unrounded force and curvature
pairs internally, but the existing caller still rounds and the reductions are
unchanged. The guarded slider owner has A/B, candidate and checkpoint low-word
buffers, with prepare/commit/late-rejection copy operations and a borrowed
source-low pointer. Its candidate lows, endpoint ABA and Matter plane geometry
remain unconnected. The guard has not been removed. Offline shader and host
compilation and CPU layout checks passed; no native transaction or rollback
was executed against this revision.

## Required before any new full qualification

1. Finish connecting accepted, candidate and checkpoint high/low slider
   position and velocity. Pass them through MetalWorld and Matter, including
   candidate kinematics, and independently advance ABA from the actual
   compensated physical contact force and gravity. Certify its endpoint
   against the Newton stage increment before accepting a complete transaction.
2. Carry the paired plane position through accepted, trial and endpoint
   projections. Use it in gap, force, tangent, energy, clearance and final
   residual; checkpoint, commit and rollback must include both words.
3. Implement the same surface quadrature and triangle vertex clearance in
   Metal contact and its full cross-node matrix-free tangent, line-search energy
   and reacting-body wrench.
   The retained triangles and FP64 reference are not a GPU implementation.
4. Generate new independent references for the exact graded mesh and law;
   establish temporal and spatial convergence, including bulk-wave resolution.
   Qualify force peaks separately from binned force and cumulative impulse.
5. Run a genuine two-way reacting-body transaction including impulse, torque,
   work, whole-environment rejection/retry and exact physical restoration.
   Compare midpoint and backward Euler at their largest equally qualified
   timesteps, with untimed diagnostics outside complete rollout benchmarks.

The directional CCD and augmented-Lagrangian research options remain disabled.
No acceptance limit has been relaxed. Source, binary and reference fingerprints
must be regenerated for the new discretization; do not reuse old references.

## Local checks on this checkpoint

* The streamed reader visits 200,000 synthetic steps with less than 1 MiB of
  Python-tracked peak memory and rejects truncated and malformed input. Four
  retained successful and failing analyzer comparisons matched previous output.
* A fake failed producer verifies chosen-directory routing, failure recording
  and no-overwrite behavior. The cost and recovery alignment unit checks pass.
* CPU tests check the quadrature rule through degree five, graded mass and area,
  nonuniform-contact force conservation, and cross-node tangent symmetry.
  The compact and direct FP64 references agree on tested graded meshes.
* The affected Objective-C++ sources and shaders compile with the selected
  strict floating-point options. This is not native rollout qualification.

Only the Mac's system data volume was mounted while this checkpoint was made;
no separate larger evidence volume was established. The prior frozen control
rollout was still completing its replay, with its follow-on held. Its result
must be inspected on completion and not relabeled as this source revision.
