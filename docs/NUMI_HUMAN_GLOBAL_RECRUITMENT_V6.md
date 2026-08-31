# Numi Human global recruitment v6

The 2026-09-01 checkpoint adds a deterministic simultaneous activation polish
after the accepted whole-body posture. It addresses finite-sweep ordering error
without adding anatomy, hidden generalized forces, or a second force-authority
path.

The existing source-ordered bounded coordinate solve remains the robust
initializer. At the final posture, a bound-projected diagonal Gauss-Newton
proposal updates all 416 activations together. Every candidate is reevaluated
with the exact compliant muscle force law and accepted only after backtracking
reduces the same acceleration-space objective reported by the certificate.
Activations remain in `[0, 1]`; support remains unilateral; joint equalities,
position limits, and source passive tissue remain in the owning solve.

Global polishing is deliberately performed once after pose search. An
exploratory implementation that polished every discarded pose trial reached a
lower static residual, but took `242.48 s` at 960 sweeps. The accepted design
takes `62.22 s` for the same certificate and replay workload.

## Apple M4 Pro results

Both current rows use the passive-aware NHMYO2 416-muscle artifact, NHCNT1
support, NHEQ1 equalities, 40 source passive-coordinate rows, a `0.0001 s`
response step, all 128 residuals, and 24 global iterations.

| Metric | 240 sweeps | 960 sweeps | Previous 960-sweep v5 |
| --- | ---: | ---: | ---: |
| Internal normalized residual RMS | 1.93793 | 0.693309 | 0.980016 |
| Maximum acceleration (rad/s2) | 30.9833 | 19.1793 | 20.0873 |
| Coordinates above 100 rad/s2 | 0 | 0 | 0 |
| Coordinates above 10 rad/s2 | 5 | 2 | 2 |
| Coordinates above 1 rad/s2 | 48 | 27 | 34 |
| Maximum root acceleration residual | 1.51499 | 0.362014 | 0.607936 |
| Relative body-weight error | 2.02818e-9 | 2.02639e-9 | 2.32887e-9 |
| Accepted global steps | 11 / 12 tried | 24 / 24 tried | n/a |
| Replay | bitwise | bitwise | bitwise |
| Wall time (certificate plus replay) | 98.78 s | 62.22 s | not recorded |

At 960 sweeps, the final-only polish reduces normalized RMS by `29.26%`, root
acceleration residual by `40.45%`, and coordinates above `1 rad/s2` from 34 to
27 relative to v5. The non-monotonic wall time is expected: the 240-sweep case
accepts 12 posture steps while the 960-sweep case accepts two. The 240-sweep
result improves aggregate RMS but slightly worsens its maximum coordinate, so
960 remains the offline qualification setting rather than treating polish as a
substitute for coordinate convergence.

Metal API validation was enabled for the independent reference probe on an
Apple M4 Pro. CPU source parity, articulated inverse/forward dynamics,
kinematics/Jacobians, muscle route force, activation integration, joint
equality projection, and the bounded equilibrium compile pass. The global
polish accepts all 24 exact-objective steps in both the reference and 960-sweep
whole-body runs.

## Remaining measured mechanics gaps

This checkpoint changes the diagnosis, not the scientific boundary. Internal
balance is still false. The largest residuals remain bilateral wrist flexion
at `19.18/15.78 rad/s2`, followed by left third-MCP flexion at `-8.56`, thumb
CMC flexion/abduction, bilateral third-MCP abduction, and fourth-MCP flexion.
These are now better candidates for sourced pulley, extensor-hood, retinacular,
and passive joint mechanics because an additional global recruitment pass no
longer removes most of their error.

The result is a deterministic static unilateral support and muscle-recruitment
certificate. It is not proof of internal equilibrium, dynamic balance,
sustained standing, gait, deformable fascia, cartilage, ligament mechanics, or
clinical validity.

Machine-readable all-DoF evidence, raw 240/960 replay traces, timing, source
manifest, and Metal reference transcript are in
`docs/media/numi-human-global-recruitment-v6/`.
