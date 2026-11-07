# Interval-local CCD experiment, 7 November 2026

Run through **ssh numi on Apple M6**, from `9c5e8fb`. Keep both directional
options disabled. The new interval-local primitive bound is slower and rejects
all tested complete physical steps. It also exposed an existing rotational
speculative-fallback failure at the 16-iteration cap.

[Raw evidence and identities](evidence/ccd-interval-local-20261107/README.md)
retain earlier candidates, corrected sources, calibration and the final tests.
This follows the [first directional experiment](CCD_DIRECTIONAL_EXPERIMENT_20261107.md);
its older measurements have not been relabelled.

## Bound over the whole proposed interval

After the same hemisphere correction as the actual quaternion interpolation,
let u(a)=q0+a*d, d=q1-q0 and Q(a)=u(a)/|u(a)|. For the general float endpoint
values (without assuming their lengths are exactly one),

```
|omega(a)| = 2 * |q0 wedge q1| / |u(a)|²
|u(a)|² = |q0|² + 2*a*(q0·d) + a²*|d|²
```

For [a0,a1], its minimum occurs at
`clamp(-(q0·d)/|d|², a0, a1)`, including a possible interior minimum. The
implementation encloses that minimizer, the squared normalization denominator
and the six wedge components using outward scalar interval operations. The
numerator upper bound divided by the denominator lower bound bounds the rate
throughout the interval. If no positive finite bound is obtained, the advance
is rejected. Equal endpoint quaternions have zero rotational rate.

The point rate only proposes an interval. A second calculation bounds the
**entire** proposal, then shortens the advance using that bound. It never
certifies an interval using only its endpoint angular velocities. The fixed
support-plane gap is re-evaluated geometrically, combined with projected
relative translation and radius times angular-rate bounds. Strict Metal
arithmetic and the retained FP32 guard are used; this is not a general formal
proof of all floating-point geometry operations.

A strict certificate over the whole remaining sweep uses the resolver's
beyond-end sentinel. Candidate 001 instead advanced to exactly alpha=1 and
then spuriously rejected its zero next horizon; its observations are retained.
Separate body anchors and production default flags are preserved. This bound
is restricted to eligible primitive casts, not a rod or mesh qualification.

## Final isolated-kernel comparison

There are 16 distinct sweeps, including unused 73°, 137° and 169° thin-object
rotations and certified close misses, at scales 0.3 and 2.5. Each is replicated
across 1024 environments with 9 timing samples. Replicas are not independent
geometry units. CPU FP64 OBB SAT/plane sampling and analytic miss cases provide
an independent reference; sampling alone is not a universal CCD proof.

The 16 cap is decisive. The 128 cap is diagnostic. The ratios below are sums
of case GPU medians, **not accepted simulated time**.

| Cap | Scale | Unresolved, existing/global/local | Local / existing GPU cost |
|---:|---:|---|---:|
| **16** | 0.3 | 9 / 10 / 10 | **1.559×** |
| **16** | 2.5 | 9 / 10 / 10 | **1.597×** |
| 128 | 0.3 | 2 / 8 / 8 | 2.382× |
| 128 | 2.5 | 2 / 8 / 8 | 2.356× |

All local unresolved sweeps are typed failures (10/16 at cap16, 8/16 at128).
The focused two CTests pass their **geometry-or-rejection** gate; this allows
unresolved rejection and cannot establish a resolved physical interaction.
The cost prediction is refuted for this selected kernel workload, and the
physical promotion gate fails independently.

## Complete MetalWorld steps and fallback failure

The second instrument builds the actual MetalWorld owner and its production
shader recipe. It runs a 1 kg thin box with correct box inertia against a plane,
10 ms step, no gravity or damping, one substep. The mode is **hybrid CCD with
the production cap of 16** (also128); the engine's default CCD mode is
speculative, so these observations must not be described as qualification of
all default settings. Every case is repeated and compared byte for byte.

Fixtures are a downward plate impact and force-free principal-axis rotations
of 250 and an additional unused 177 rad/s. The rotating box has half extents
(0.2,0.002,0.03) m and centre height 0.1998 m. Both rotations pass pi/2 during
the step. The exact unforced minimum plane gap is
`0.1998-hypot(0.2,0.002) = -0.00020999975 m`, while the final poses are clear.

At cap16 the existing arm **accepts both rotations with zero normal impulse**,
`unresolvedCCDCount=1`, and clear endpoints. The older global-directional arm
does the same. This is a swept impact without response, despite success status;
positive final separation does not certify the interval. Candidate002 also
reached that same fallback. Its failure is retained separately.

Candidate003 forbids the legacy speculative fallback for unresolved
interval-local primitive casts. It now rejects all three complete-step fixtures
at16 and128 with status11, restores initial q/v exactly, and produces zero
accepted unresponded swept hits. All18 arm/cap/fixture results repeat identically.
The existing arm resolves the 250 rad/s rotation at128 with impulse0.2101 N·s;
the177 rad/s case instead rejects with status9. Raising the cap does not
establish broad physical correctness.

Rejecting the sweep closes the candidate's fallback path. It does not fix or
qualify the existing fallback, and it does not turn rejection into a successful
collision simulation. The legacy behavior remains unchanged as the control;
the explicit failure report should guide a separately bounded owner-level fix.
The complete-step wall observations include initialization and are not timings
for a performance claim.

## Decision and next prediction

The interval-local normalization-denominator calculation addresses the proposed
mathematical issue, but it is not an effective engine improvement in this test.
Leave `directionalCCDAdvancement` and `intervalLocalCCDAdvancement` false.
No accepted simulation-throughput, force, friction, torque or control
qualification follows from these kernel tests.

The next collision question is whether an unresolved rotational sweep can ever
be safely admitted by the existing speculative fallback. A bounded fix should
reject or resolve the retained250/177 rad/s hits at cap16, with correct impulse
and exact rollback, before evaluating timing. The contact-convergence experiment
remains the main numerical research priority.

## Reproduction

```sh
cmake -S tests/ccd -B /tmp/numi-ccd -G Ninja
cmake --build /tmp/numi-ccd -j 4
ctest --test-dir /tmp/numi-ccd --output-on-failure
/tmp/numi-ccd/bin/metalrobo_ccd_advancement_probe /tmp/numi-ccd/shaders/NumiCCDProbe.metallib 0.3 1024 9 16 > /tmp/ccd.json
python3 tests/ccd/analyze.py /tmp/ccd.json
cmake -S tests/ccd/world -B /tmp/numi-ccd-world -G Ninja
cmake --build /tmp/numi-ccd-world -j 4
/tmp/numi-ccd-world/metalrobo_ccd_world_step_probe /tmp/numi-ccd-world/shaders/MetalRobo.metallib > /tmp/ccd-world.json
python3 tests/ccd/world/analyze.py /tmp/ccd-world.json
```

The complete-step observer exit indicates collection, not physical success;
its analyzer records legacy misses and requires the local arm to reject them.
