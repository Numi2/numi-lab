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
