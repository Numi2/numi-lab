# Larger equality blocks in the existing projected-contact solve

All compilation, tests, native runs and geometry audits ran through SSH on
the Mac mini (Apple M4 Pro). The manifest retains exact source, command,
configuration, device, binary, trace and unretimed recording identities.
Large geometry packs and recordings remain at the listed Mini paths.

Run 043 completed six seconds in 86.194 wall seconds (0.06961 times real
time). Measured GPU time was 29.98% below source020/run037. The existing
projected-contact algorithm replaces the prior interleaved fallback for the
91-row block. This changes the finite-iteration body trajectory: maximum
terminal q/v differences were 0.03385/0.00433 in their mixed coordinate units.
Physiological trace columns were identical; contact columns were not.
Digit q/reference errors and velocities were zero, root assistance was
disabled, and rejected-step/retry checks passed. Full-skin exact audits at
steps 639, 2783 and 2999 each found zero forbidden pairs.

The production bilateral-kernel test passed 5,068 checks, including independent
reduced-mass, momentum, energy, contact and tangent-residual checks with 64,
65 and 91 equality rows. Its initial launch exposed a stale test function
specialization; `test.log` retains that failure, and `test2.log` records the
correctly specialized production-kernel result. The original 51-row native
configuration preserved both traces, initial/accepted packs and terminal
state bit for bit in run044.

Run040 is the preceding source020 24-second stability diagnostic. Its full
skin has zero forbidden pairs at 10, 18 and 24 s; numerical, volume, contact
and accepted-history checks pass. Slow body settling remains measured in
`engineering-stability-analysis.json`. Its recording has 376 frames over
476.14 wall seconds, with no retiming.

Run041 is a rejected performance experiment: reusing the 16 KiB factor cache
for packed lower/upper triangles made no meaningful difference. Its source,
build and profile remain here; it is not part of the published implementation.

These runs retain the old thoracic input and its known lung/rib and ventricular
wall defects. They do not satisfy the full anatomical, five-minute paired
intervention or real-time acceptance target. They are engineering evidence,
not clinical validation.
