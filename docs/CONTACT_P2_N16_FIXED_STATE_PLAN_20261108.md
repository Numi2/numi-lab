# P2 n=8/n=16 onset fixed-state transfer plan

This plan is sealed before any new capture or transfer run. The original
force and impulse qualification gates remain 0.25%; this analysis cannot
qualify the complete 10 ms reference.

## Existing evidence and limitations

The independent P2 n=8 full fine run retains nodal displacements at 0.1 ms
intervals (SHA-256 b92ee66251b0c3f2bf6ca67927f9aa097b5b5b9d76fa72f65c1a93286e21b100).
The exact-freefall n=8 fine onset reconstruction
(SHA-256 998428bba53303b403ecfcbc120427061c1fefceb006d93d372d49c5d2e533b7)
reproduced its integrated first bin to 2.09e-9 of the n=8 full peak, but did
not retain displacement fields. The n=16 fine onset output
(SHA-256 81960388226b25a495f15dac4c14ee32f7abe27d143cf368daa0490d74a05a8b)
contains six 4.5–5.0 ms force and diagnostic samples, not nodal fields.
It used onset source SHA-256
5cc2ea81b4363827d0db304a05a74250b76a8d5abc170ad73abc804f3c666dae
and model SHA-256
726dbf72ff85ceec348df396b6a26c4db90583bc7843d8c8303035d6580f96bb.
The source, inputs, configuration and host evidence are in commit
5ce57ccda81bc3b0506c06af12c718723eb0f60c.

At 4.8 ms, the retained n=8 and n=16 order-18 normal forces are
0.0009889625 and 0.1064971570 N. At those six sampled n=16 states,
order-18 versus Duffy-64 differs by at most 2.81e-8 of the n=8 full peak
0.0874322633 N. This supports different sampled evolving displacement or
pressure fields as a contributor, while the quadrature checks argue against
sampled surface integration error. Six samples do not certify the integral,
establish a phase-error cause, or measure between-sample clearance.

## Next executable diagnostic

First make a capture-only copy of the n=16 fine onset instrument that saves
the displacement and velocity vectors at 4.5–5.0 ms, preserving its
equations, initialization, solver settings, evaluation points, checks and
contact law. Seal source and input hashes, package versions, command and
the pre-run plan in Git before executing. Compare its first-bin impulse and
all six sampled normal forces to the existing n=16 fine output. Require each
force and bin difference <= 1e-7 times n=8 peak, positive sampled gap and J,
and the existing 0.25% momentum, energy and work gates plus 0.05%
Duffy-48/64 and 0.25% order-18/64 gates. Failure invalidates the transfer
comparison; preserve the failure without rerunning automatically.

At the six matched times transfer the retained n=8 displacement field to the
n=16 surface and captured n=16 field to the n=8 surface, using P2
interpolation with an inward evaluation offset and independently reporting the
offset. At each surface evaluate the fixed regularized law with Duffy-48 and
Duffy-64. Test exact transfer of constant, affine and quadratic fields on
both meshes, verify sampled positive gaps, and compare force reproduction with
the saved untransferred Duffy-64 values within 1e-7 of n=8 peak. Require
48/64 agreement <= 0.05% of n=8 peak. Test a second, smaller inward offset
and require the force change <= 1e-7 of n=8 peak. If a transfer or control
fails, report it rather than interpreting force differences.

Report actual and transferred forces and gap ranges at every matched time,
including each direction's transfer-induced force change normalized by the
n=8 full peak. A small n=8-to-n16 change with a large reverse change would
identify sensitivity to the coarser representation for those states; it does
not explain whether the two trajectories differ due to bulk-wave dispersion,
contact dynamics, or their interaction. Since force is nonlinear in gap,
individual forces and signed changes will be reported rather than attributed
additively. These are instantaneous diagnostics, not time-integrated force
gates, certified clearance, or physical validation.

## Resources and execution boundary

The previous fine run used 448.35 s and peaked at 3,207,472 KiB sampled RSS;
wall time included uncontrolled background activity. The proposed rerun is
expected to need roughly 8–10 minutes and about 3.1 GiB sampled RSS; storing
six vectors adds little, but this is an estimate rather than a guaranteed
bound. Use the same single-thread BLAS, 1,800-second process-group timeout,
4 GiB sampled-RSS stop, 1 Hz monitoring, full logs and no GPU. Record host
load and any overlap. Stop on any failure or cap; do not start another job
automatically. Transfer will be separately bounded and monitored after capture,
with a separately sealed command and resource plan. Coordinate the host with
the parent before starting; other native gates take precedence. No full n=16
trajectory is proposed.

## Capture outcome

The capture and monitor exited successfully: 450.66 s elapsed and 3,177,216
KiB peak sampled RSS; neither cap was reached. The six captured force samples
and first-bin impulse reproduce the retained n=16 fine output exactly.
Displacement and velocity arrays have the declared dimensions and finite
values. All declared sampled clearance, Jacobian, quadrature, momentum,
energy and contact-work checks pass. Raw output, host and RSS records,
identities and analysis are retained in
[evidence](evidence/contact-p2-n16-capture-20261108/). This captures states
for the separate transfer diagnosis; it is not force qualification.

## Transfer execution registration

Run the retained fine n=8 full states (indices 45–50) and verified fine
n=16 onset capture through the P2 onset transfer script.
The transfer uses the same P2 tetrahedral mesh family and 32-point
interpolation chunks, separately evaluating Duffy-48 and Duffy-64 without
factoring a mass matrix. A polynomial transfer control requires each
constant, affine and quadratic displacement to agree within 1e-12 m.
All other reproduction, offset, quadrature and positivity gates above
remain unchanged. Retain exact source, state, environment and result hashes.

Use a single CPU thread and no GPU, 1 Hz sampled RSS, a 4 GiB hard RSS
stop and 600-second process-group wall stop. An estimated 1–2 GiB and
1–3 minutes is uncertain, especially for Duffy-64 allocation; do not infer
throughput or safety from the smaller preceding n4/n8 transfer. Stop on
failure, timeout or cap and retain the result. Coordinate with the parent
before execution and do not overlap queued native correctness checks.

## First transfer attempt

The first registered transfer exited 1 after 1.02 s before any force result or
resource limit. The quadratic-control interpolation failed in scikit-fem
element finding with “Point is outside of the mesh.” The manifest and complete
stderr are retained in the transfer evidence. No numerical comparison is
valid from this attempt. Diagnose the point ownership before preparing a
separately identified corrected attempt; do not silently repeat it.
