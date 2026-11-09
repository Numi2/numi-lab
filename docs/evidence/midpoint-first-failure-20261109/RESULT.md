# First rejected midpoint transaction: diagnosis

This is a diagnostic result, not a new accepted integration method. On the
8-division fixture, h=3.90625 microseconds, step 203 rejects before contact
with Matter status 10 and the whole recorded physical state restored exactly.
The GPU full-step momentum residual is 1.39946323e-13 N s against its unchanged
1.37214662e-13 N s limit. An independent FP64 force and momentum evaluation
of the same discrete equations gives 1.39986633e-13 N s. The 0.0288%
difference is consistent with the retained calculations, and both exceed the
existing limit. This is not merely a misleading status or the old
backward-Euler residual applied to the reflected endpoint.

The [boundary capture](capture-002.json) and [offline analysis](capture-002-analysis.json)
separate twice the actual midpoint-stage residual (norm 7.25943913e-14 N s)
from the FP32 endpoint-velocity reflection contribution (norm 1.07335419e-13 N s).
Their vector sum has norm 1.40058120e-13 N s. These contributions are vectors:
their norms cannot be added to infer the combined error. The residual is
therefore sensitive to endpoint representation as well as the stage solve;
the evidence does not establish that either contribution alone is the only
cause. The independently reconstructed net residual is approximately
[8.37e-17, 2.64e-16, -8.90e-13] N s; the independent maximum individual-node
residual is 2.20818e-14 N s. Aggregate momentum and local errors remain
separate checks.

Capture 001 was made too late, after scratch was restored. Its stage and
reflection decomposition is **invalid**; its files are retained, not used as
supporting evidence. Capture 002 copied the candidate and residual on the
existing borrowed command buffer immediately after certification and before
rollback. The copied stage differs from the accepted state, the reflected
endpoint and independent equations agree with the failure, and the selected
step restores the same accepted physical digest as capture 001. It does not
establish a general precision fix, contact onset convergence, spatial accuracy,
or two-way dynamic coupling. The relevant executable and shader hashes are
in [inputs-002.sha256](inputs-002.sha256).

The earlier independent per-node diagnostic used a fixed 1e-10 N s denominator
floor. On the failing state its maximum is 2.91074e-5 with that floor and
3.36305e-5 using only each node's gravity impulse as the scale. This difference
is not an authorization to change the actual acceptance certificate. The
fixture-specific no-floor follow-up is preregistered in
[PHYSICAL_SCALE_PLAN.md](PHYSICAL_SCALE_PLAN.md); it is not a general scale
for zero gravity or constraints. All historical results remain as measured.

## Exploratory frozen-geometry representability check

After the boundary capture, an independent calculation rounded each component
of the force-balanced endpoint velocity to its nearest FP32 value while
holding the captured force, geometry, contact state and accepted state fixed.
The resulting minimum componentwise residual norm is 8.63501e-14 N s, below
the unchanged 1.37215e-13 N s limit; 538 of 2187 endpoint velocity components
were already at the nearest value under that frozen-force calculation. This
is an exploratory conditioned calculation, not a candidate transaction: it
does not update positions, recompute the force, enforce the midpoint
kinematic equation or demonstrate that a coupled solve can reach it. It rules
out claiming from this capture alone that the endpoint FP32 representation
makes passing the current limit impossible. See
[capture-002-representability.json](capture-002-representability.json).

## Independent coupled-rounding experiments

The [unrestricted CPU iteration](coupled-rounding-002.json) recomputed the
midpoint force after each velocity and position correction. It converged to a
full-step residual of 8.63512e-14 N s, below the unchanged limit, with a
3.50e-21 m midpoint kinematic defect and 0.000196885 m minimum gap. However,
875 velocity components were not reproduced by the nearest three stage floats
under the runtime reflection operation. The first file
[coupled-rounding-001.json](coupled-rounding-001.json) is retained; its
`unreachable` label overstated what that finite search establishes and was
corrected in the second file.

The [stage-reflection constrained follow-up](stage-reflection-001.json) found
a repeated candidate with residual 1.21144e-13 N s and kinematic defect
3.39e-21 m, again above clearance and below the unchanged residual limit. All
endpoint velocity components were reproduced by the nearby tested stage
floats. This rejects the preregistered prediction that the local reflection
restriction would necessarily exceed the limit. These are independent CPU
candidates for one precontact state, not accepted Matter steps: they do not
prove the native stage solver can reach the state or that contact trajectories,
whole-environment retry and accuracy-qualified performance pass.

## First native precontact correction: incomplete

A bounded precontact correction candidate was tested inside the midpoint
transaction, followed by force and residual reassembly and the same final
certificate. The correction is now separately disabled by default because the
full run failed; the midpoint option also remains disabled by default. In the
[204-step native trial](native-polish-001.json) the first 203 steps match the
original trace and the previously rejected step 203 was accepted. Its
independently recomputed full-step residual is 1.21143e-13 N s, below the
unchanged 1.37215e-13 N s limit; it is still before contact. The physical
state replay and forced rejection restoration checks passed.

The requested [10 ms run](native-polish-n8-full-001.json) did not complete:
it accepted 407 steps, then rejected step 407 with status 10 and residual
1.67921e-13 N s against the same 1.37215e-13 N s limit. Its restored state,
forced rollback, replay and continuation replay checks passed. No nonzero
contact was reached, so it is not contact-onset or force qualification. The
correction remains an experimental candidate; no larger-step, spatial,
performance or coupled-body claim follows. The native input hashes are in
[native-polish-001-inputs.sha256](native-polish-001-inputs.sha256); the new
failure is the subject of [NATIVE_REPAIR_407_PLAN.md](NATIVE_REPAIR_407_PLAN.md).

The subsequent [boundary capture](capture-407.json) reproduced exact
restoration and gave an independent full-step residual of 1.67879e-13 N s,
matching the GPU. The stage-reflection-constrained [nearby search](stage-reflection-407.json)
settled at 1.67879e-13 N s; even the [unrestricted FP32 endpoint search](unrestricted-rounding-407.json)
settled at 1.58225e-13 N s, both above the unchanged 1.37215e-13 N s limit.
These are local, force-recomputed searches on a single captured state, not a
proof of global impossibility. They provide no basis for simply increasing
iterations or declaring the prototype ready. The historical trial enabled the
candidate implicitly under midpoint; the current code requires the explicit,
fingerprinted `NM_MATTER_MIDPOINT_PRECONTACT_POLISH=1` opt-in solely for
reproducing and testing that incomplete research candidate. Normal midpoint
runs do not apply it. [Input hashes](capture-407-inputs.sha256) and the
[rounding input hashes](rounding-407-inputs.sha256) are retained.

## Follow-up without the unit-scale floor

Both midpoint rollouts completed 640 of 640 steps and replayed, with unchanged
physical and execution fingerprints and the same final physical digests as
the earlier measurements. Their recorded force, impulse, gap, work, momentum,
contact-count and old local-residual traces match the originals at every step.
With no fixed floor, the maximum node residual divided by its own gravity
impulse is 0.000658385 at n=2 and 0.000155196 at n=4, each below the
preregistered 0.005 fixture-specific limit. The old floored values were
0.000653732 and 0.000146836 respectively. The independent global momentum
checks and rollback/replay results remain separate. The [comparison](scaled-analysis.json),
[original](scaled-n2.json) and [four-division](scaled-n4.json) reruns, and
[binary and shader hashes](scaled-inputs.sha256) are retained. These results
do not establish convergence across meshes or a backward-Euler cost partner.

After rebuilding the contact targets against the current owner, the native
contact suite passed [12/12](ctest-current.log). The historical binaries used
for the earlier measurements were first checked against their recorded hashes
and preserved byte-for-byte in the build directory under
`evidence-1dd807964/`. Neither the experimental augmented-Lagrangian contact
nor the interval-local CCD option was enabled. The restricted midpoint and
area-IPC modes remain opt-in.

The rebuilt suite with the precontact candidate disabled by default passed
[14/14 tests](ctest-polish-disabled.log). Its dedicated step-203 test enables
the candidate explicitly; this test does not hide the retained step-407
rejection or qualify a complete rollout. The diagnostic-scope test checks
that the opt-in changes the execution fingerprint and rejects invalid or
out-of-scope activation.
