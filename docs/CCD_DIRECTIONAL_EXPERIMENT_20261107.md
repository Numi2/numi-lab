# Directional CCD experiment on ssh numi, 7 November 2026

The candidate improves translation-heavy collision search, but does not improve
the selected mixed workload overall. Keep it opt-in and disabled by default.
The first candidate also exposed a real silent miss; its source and observations
are retained. The final candidate rejects that uncertified advance.

This work ran on the Apple M6 Mac mini (16 GB) through ssh numi, starting from
main@efde73484b24189a8ba83a79be2bff30437d6ba7. Source, native executable,
shader, compiler/OS, plans and observations are bound in the evidence manifest.
No earlier retained measurements were relabelled as this run.

## What changed

Collision.metal can use the relative translation along a separating axis plus
a bound on rotation for its actual shortest-normalized-quaternion interpolation.
For hemisphere-corrected unit endpoints, put
c = dot(q0,q1) and q(a) = normalize((1-a)q0+a*q1). The peak angular rate per
unit alpha is 4*tan(acos(c)/2), equivalently
4*length(q1-q0)/length(q1+q0). It can exceed the average angle and the
endpoint rotational chord; those are not interchangeable advancement bounds.

The candidate re-evaluates support extrema on the query axis and advances using
their projected separation. If that plane is uncertified, or its safe advance
falls below the minimum work budget, it records an unresolved failure. It does
not raise a certified step beyond that interval. Existing collision anchors,
normal-velocity targets, meshes, changing-length capsules, rod sweeps, and the
default advancement path retain their existing behavior.

Enable only for an explicit experiment:

~~~cpp
config.ccdMode = metalrobo::MetalWorldCCDMode::hybrid;
config.directionalCCDAdvancement = true;
~~~

Shapes still need their existing exact-CCD flags. Finite convex shape support
is implemented, but this experiment evaluates only the listed box cases.

## Final native observations

The final instrument ran ten paired sweeps at each unused geometric scale,
0.5 and 4, with 1,024 repeated environments and nine post-warmup GPU timings per
arm. This is 20 distinct sweep conditions, not 20,480 independent experiments.
The advancement cap was 128; the engine default remains 16.

| Sweep | Existing iterations | Directional iterations | Directional result |
| --- | ---: | ---: | --- |
| Thin plate impact | 25 | 3 | Impact reported |
| Tangential plane miss | 128 | 1 | Miss certified; existing arm falls back |
| Tangential box miss | 20 | 2 | Miss reported |
| Coherent translation impact | 128 | 2 | Impact reported; existing arm falls back |
| Moving catch | 25 | 2 | Typed unresolved failure |
| Rotating edge hit | 4 / 2 | 7 / 6 | Typed unresolved failure |
| Rotating edge miss | 6 | 7 | Miss reported |
| Near-tangent rotating plane hit | 2 | 128 | Unresolved, speculative fallback |
| Near-tangent rotating plane miss | 58 | 128 | Unresolved, speculative fallback |
| Antipodal representation of rotating hit | 2 | 128 | Unresolved, speculative fallback |

Both final invocations observed zero directional geometry-gate failures and
byte-identical record/status replay. This gate allows an explicit unresolved
or rejected result. Each final scale had five directional unresolved sweeps,
including two typed failures, versus two unresolved sweeps in the existing arm.
It does not mean all contacts were successfully resolved.

The sum of the ten per-case GPU medians was **1.016x and 1.053x** the existing
arm's sum at scales 0.5 and 4. This descriptive kernel comparison includes
slower rotations and rejected cases. It is neither an accepted-state
simulation speedup nor a fixed-physical-accuracy comparison. Iteration
reductions in selected sweeps are insufficient to promote this method.

## Failures and revisions retained

- Calibration 001 used an overly ample speculative margin. Its performance
  observations are unrepresentative; the instrument was corrected, and the
  output was retained.
- Candidate 002's scale-0.1 rotating-edge sweep silently missed a reference
  impact after an uncertified support axis reverted to existing advancement.
  The native probe exited 1. This contradicted the no-silent-miss prediction.
- Candidate 003 fails closed on that disagreement. Replication on the exposed
  condition and confirmation at unused scales 0.25 and 1.5 retained zero
  directional geometry-gate failures, with unresolved cases still present.
- Candidate 004 also fails closed below the minimum advancement budget.
  Confirmation used unused scales 0.75 and 3.
- Instrument 005 retains every timing sample and checks status-buffer replay,
  in addition to record replay. Its final confirmation used unused scales
  0.5 and 4. Earlier timings retain their median-only limitation.
- Existing advancement produced a late rotating-edge event bracket in exposed
  scale-1 and scale-0.1 conditions. This is bounded geometric evidence of an
  existing path weakness, not a whole-scene outcome.

## Verification and next distinguishing test

The standalone CMake build compiled the actual production shader and native
probe on the M6. Both CTests passed: default-scale geometry/replay and the
small-scale exposed regression. MetalWorld.mm passed Objective-C++ syntax
checking with the new configuration field. A complete MetalWorld rollout was
not built or qualified; no transaction or physical-fidelity claim follows.

The current global rotation-rate bound remains too loose near tangency.
The next numerical candidate should preserve support-plane certification while
using interval-local rotational bounds. Before collecting that candidate's
results, seal a new prediction on unused rotations, scales and geometries.
Include the same no-silent-miss gate, unresolved/rejected incidence, torque and
friction from actual coupled rollouts, and accepted physical seconds per wall
second including rejected work. Obstructed speculative contacts and motion
introduced during a solve remain separate unimplemented experiments.

Build/run instructions are in [tests/ccd](../tests/ccd/README.md).
All plans, raw JSON, failed observations and prior source snapshots are in
[evidence](evidence/ccd-directional-20261107/). The complete remote evidence
directory is /Users/numi/numi-lab-ccd-evidence-20261107; old binary/shader
payloads remain there and are identified by hashes in the manifest.

## Bounded follow-up

The [interval-local experiment](CCD_INTERVAL_LOCAL_EXPERIMENT_20261107.md)
adds whole-interval quaternion bounds, decisive cap16 tests and complete
MetalWorld steps. It remains disabled, and retains a legacy rotating-sweep
fallback failure. Those are separate new observations, not revised evidence
for the original candidate.
