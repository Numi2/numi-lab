# Saturated implicit motors: native defect confirmed, endpoint-row alternative rejected

The production drive path under-realizes saturated torque in an isolated native
crow-scale hinge. It clamps the pre-integration drive command, then retains the
unconditional `h*D + h*h*K` augmentation of the physical mass plus authored
armature. The correct saturated backward-Euler solution clips the solved velocity
between `v - h*T/M` and `v + h*T/M`; independent scalar bisection agrees with
that reference within 4.45e-16 rad/s over 256 calibration conditions. Native FP64
forward dynamics separately verifies the physical hinge inertia.

All 96 production-path observations match the existing clamp-then-augment
formula (maximum 4.86e-8 rad/s error). The 48 unsaturated controls also match the
correct reference. At saturated rest, the realized 10 mN m command is only
2.379/3.873/5.599/7.185 mN m at equivalent 16/32/64/128-substep durations.
The minimum saturated-rest deficit is 28.15%; the maximum final-velocity
reference discrepancy is 1.588 rad/s. These deterministic numerical conditions
establish the isolated mechanism, not its causal contribution to the full crow
release/landing discrepancy.

## Bounded constitutive row and the failed hybrid prediction

A canonical generalized row with `J=1`, target velocity
`K*(target-q)/(D+h*K)`, dissipation `1/(D+h*K)` and impulse bounds `+/-h*T`
implements the desired scalar constitutive equation with physical mass and
ordinary effort mode. Native code owns the solve and all physical state.

The first invocation under seal `9fa4c2ce` was rejected by canonical IR validation
before any physics: a required zero warm impulse was missing for each row.
Its binary, source, header-only TSV and logs are preserved. The metadata-only
amendment sealed at `d0604716` adds the zero spans; all states, equations,
solver settings and quantitative gates remain unchanged.

The corrected invocation completes all 144 one-microstep conditions: 16 states,
four durations, CCD disabled/hybrid, plus two scalar unilateral velocity-stop
conditions. Half the free states repeat the earlier diagnosis, half were unused
before the bounded study. This is one deterministic engineering study, not
independent learning or biological replication. The stop is a canonical
mechanism constraint, not geometric beak contact.

The 72 CCD-disabled conditions meet every predeclared gate: velocity error
<=4.09e-7 rad/s, position integration error <=4.51e-9 rad, motor torque residual
<=1.25e-9 N m and momentum residual <=2.45e-9 N m. Impulse bounds and both
stop directions pass. Hybrid final velocity, impulse bounds, momentum and stops
also pass, but its position error reaches .00260417 rad and constitutive torque
residual 8.2947e-5 N m. The original 2e-7 rad and 1e-7 N m gates fail.
**The bounded alternative is rejected for production; criteria are unchanged.**

Source tracing explains the placement error in this prototype. Hybrid CCD
predicts and materializes position before endpoint contact/constraint impulses,
then publishes that position and post-impulse velocity. That ordering preserves
position at instantaneous impact. A motor row inserted only at the endpoint
omits motor force from the free-motion predictor, so it cannot serve as a
force acting throughout the segment. Reintegrating every post-impact velocity
across the completed segment would also be wrong: it moves the impact geometry.
This is a failed motor placement, not evidence that all contact positions should
be recomputed from endpoint velocity.

## Production obligations before another crow release trial

The next implementation must include bounded motor forces in native prediction
and refresh them for actual remaining/selected CCD durations. It must preserve
authored physical mass/armature, gains, friction, controller scaling, actuator
limits and torque-speed envelopes. Smooth force accumulation and instantaneous
contact impulses must not be applied twice or share phantom implicit inertia.
Coupled articulations, real geometry, saturation changes, checkpoint/reset,
submission partition invariance and end-to-end throughput need owning checks.
Only then re-run the real crow fixture at the unchanged mechanical and 5 mm
trajectory convergence gates. The earlier 64/128 comparison remains failed.

No production source, task profile or policy changes were made for this study.
Protected deterministic deployment97 retains SHA256
`6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389`.
Full lifelong embodied capability and release convergence remain unfinished.
The user's optimizer checks remain completed in
`../crow-rl-post-step-rate-20261108/README.md`; the failed adaptive-rate policy
has not been promoted.

## Recovery and storage

Plans, calibration, actual outcomes, failed invocations, exact source, executables,
dylib/metallib and raw native TSVs are retained through `payload/manifest.json`.
Capture is content-deduplicated against actual committed Git objects. Scoped
Git-restored native replay and cleanup receipts are recorded separately in
`../crow-rl-git-recovery-implicit-drive-20261108/`. No raw evidence is removed
until remote parity, byte verification, executable recovery and closed-handle
checks succeed. These replays verify numerical reproducibility and recovery;
they do not add independent scientific observations.

Executed Git-restored recovery passed: calibration stdout and all three native
TSVs are byte-exact, including the original pre-physics metadata rejection.
The two successful native invocations reproduce all 240 microstep conditions.
The 54-file temporary tree (11,151,073 bytes) was inventory/hash/mode/mtime
validated with zero open handles, then removed. The 43 original source files
(10,761,036 bytes) are verified from actual committed Git objects; new
compressed payload is 173,961 bytes through reuse of existing runtime blobs.

After actual Git verification, remote HEAD parity and zero open handles,
all 43 raw files (10,761,036 logical bytes) were removed under the exact
manifest. The current production build and protected crow policy are retained.
