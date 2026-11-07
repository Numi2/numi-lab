# Hybrid CCD thin-impact candidate

This candidate builds on the fail-closed legacy acceptance repair. The original
unresponded-crossing evidence and the earlier failed directional experiments
remain unchanged. Neither directional advancement option nor AL contact is
enabled by this change.

## Diagnosis

The retained production-cap rotating plate exposes two independent problems.
At the first 250 rad/s event, the evaluated normal target is +0.255528 m/s,
while the reconstructed outgoing witness velocity is -1.897063 m/s; the
solver's normal residual is 2.152589 m/s. Both local witness reconstructions
agree with the recorded world midpoint and separation within 1e-8 m at that
event. The earlier -0.00794 m/s figure described only the source speculative
target and omitted the evaluated restitution term.

The wave32 contact update multiplied all three row residuals by the inverse
coupled block, then clipped the tangential impulse to the friction limit. A
sliding contact cannot satisfy the sticking equation; the unfulfilled tangent
residual therefore contaminated the normal fixed point. An independent FP64
rigid-box response model reproduces the retained first-event residual to
approximately 0.025 m/s and shows that additional iterations alone do not
remove the defect. A separate closed-form two-row sliding counterexample
establishes the same failure without the fixture's geometry.

Candidate 001 uses a normal complementarity update followed by the conditional
2-by-2 tangential block and friction projection in the wave32 and distributed
wave32 paths. At the first 250 rad/s event, the normal residual becomes
0.002448 m/s after three sweeps; the independent FP64 model predicts
0.002448 m/s. The 177 rad/s impacts then complete at both scales, but the
250 rad/s cases still reject: the post-response separating sweep exhausts the
16-iteration box-plane certificate. Candidate 001 evidence is retained.

## Certificate

Candidate 002 augments the existing fixed-plane box Bernstein certificate
with a whole-node monotonicity proof. For an unnormalized quaternion
q(a)=(1-a)q0+a q1, the numerator of the rotated vertex projection and
D(a)=q(a)·q(a) are quadratic. The physical-time map has derivative K/D,
where shortest-arc endpoints bound K between q0·q1 and |q0||q1|. Bernstein
convex hulls bound both polynomial values and their derivatives over the
entire node; outward interval operations bound the derivative after
multiplication by positive D squared. A strictly positive starting clearance
and strictly positive lower bound on that derivative certify the node.
Uncertain nodes retain the previous subdivision and fail-closed behavior.

This is a certificate for the existing normalized-quaternion/physical-time
path, not an endpoint angular-rate shortcut. The FP64 reference samples
1,000 seeded unused quaternion pairs and checks monotonicity on certified
nodes. Sampling is supporting evidence, not a proof of every floating-point
execution or every supported collision shape.

## Retained native result

On the remote Apple host, candidate 002 completed the existing 288-case
single-step hybrid box-plane matrix. At cap 16, 96/96 cases completed; at
cap 128, 96/96 completed; cap 1 rejected and restored 90/96. Every run passed
the recorded repeat and restoration checks; no run accepted an unresponded
analytic swept hit. The 250 and 177 rad/s thin impacts complete at both
scales with nonzero normal impulse, and their paired near-misses complete
with zero normal impulse. This does not establish force or torque accuracy
for all impacts, sustained rollouts, moving planes, meshes, rods, articulated
contact, deformable contact, or every solver path. In particular the ordered
serial contact replay path still uses the previous block update and has not
been qualified by this experiment.

The control paths remain experimental research changes until broader
validation. No performance comparison follows from these runs. Raw traces,
prior candidates, input identities, patches, SHA-256 hashes and results are
retained on the host in
`/Users/numi/numi-lab-ccd-thin-evidence-20261107/`.

## Subsequent analytic qualification: candidate held

The broad wave32 correction is **not qualified for integration**. Additional
closed-form checks uncovered three limitations:

* A direct Metal helper test of the existing radial anisotropic ellipse
  projection does not produce the maximum-dissipation sliding direction for
  one admissible positive-coefficient ellipse. Standard collision compilation
  currently emits equal U/V coefficients; reachability from a production
  MetalWorld scene is not established. A helper test with only one zero
  coefficient lies outside the validated ConstraintIR contract and is not a
  production defect.
* Distributed wave32 uses unrelaxed simultaneous row updates. For two
  identical coupled normal rows with W=[[1,1],[1,1]], the update alternates
  between impulses (1,1) and (0,0). This is an analytic source-level
  counterexample and has not yet been reproduced on the device.
* Candidate 002's normal update floors the raw normal diagonal at 1e-10,
  whereas its 3-by-3 inverse adds 1% scale-relative regularization. With
  diagonal response (1e-12,1,1) and unit normal residual, those two choices
  yield normal increments of 1e10 and approximately 100, respectively.
  This is a candidate-specific conditioning risk requiring correction and
  native qualification.

The reference also exercises frictionless and interior sticking cases and a
finite normal impulse cap. Its successful execution preserves these analytic
counterexamples; it does not mean that the corresponding physical behaviors
have all passed. Full shared-solver coverage and the distributed native test
remain outstanding.

Candidate 003 replaces the normal denominator with the same scale-relative,
symmetrized and regularized normal diagonal used by the actual inverse.
Independent FP64 checks against direct conditional solves cover a singular
block and 100 seeded positive-definite blocks. A test-only Metal wrapper
including the actual helper returns a normal increment of 100 for the
previous 1e-12 diagonal counterexample; frictionless, sticking and bounded
normal cases agree with their analytic results. Its anisotropic case
reproduces the helper-level maximum-dissipation discrepancy. A two-row
unrelaxed *model* retains a large normal residual; it is not a full
distributed-world result. The existing contact-owner probe passes its
257/513-contact queue and accounting checks, and the 288-case hybrid
box-plane regression still passes. Those checks do not qualify the broad
solver update.

Later noncoincident physical-velocity owner-shaped probes exposed unresolved responses in both 33-contact non-distributed and 257-contact distributed paths. The earlier contact-owner pass is only a queue/accounting result and does not supersede that finding. The thin-impact candidate remains unpromoted. The preserved cross-candidate evidence archive [ccd-research-evidence-20261107.tar.gz](evidence/ccd-research-evidence-20261107.tar.gz) has SHA-256 66071749f28860da9b8b2e6005eb17ea6c495420068b3f899120f4361affc594; binaries and Metal libraries remain host-only with identities recorded in the archive.
