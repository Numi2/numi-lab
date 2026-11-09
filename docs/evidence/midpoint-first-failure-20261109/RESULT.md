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
