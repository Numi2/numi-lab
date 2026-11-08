# Native FEM-plane midpoint transaction

Apple M6, 8 November 2026. The retained full rollout used runtime and
instrument revision `7dff38b7a` and the binary/shader hashes in
[inputs-certified.sha256](inputs-certified.sha256). The latest harness also
separates rejected-state replay from ordering of failure diagnostics; that
change does not alter the solver or physical fixture.

## Result and scope

A native, explicitly opt-in Matter transaction now solves the constant-mass
FEM fixture at its midpoint, reflects the endpoint in candidate storage,
checks the endpoint against the plane and validates its elements, and publishes
only through the existing GPU transaction. The public physical timestep is
`h`; the internal stage is `h/2`. The same authored reference-area IPC
potential, material, plane, physical clearance and coefficient are used in
the backward-Euler control and midpoint arm. The half-stage impulse is
doubled only after the final midpoint residual has been assembled. The
physical certificate checks `x1-x0=h*vm` and `M*(v1-v0)=h*f(mid)`; it
does not evaluate a backward-Euler equation at the endpoint. Runtime work
uses the borrowed command buffer. Unsupported owners and settings fail
closed. Midpoint and the earlier AL and directional CCD experiments are
all disabled by default.

The native qualification is **one environment, one stateless neo-Hookean FEM
object and one immutable frictionless plane**, with reference-displacement
coordinates and the explicitly opted-in reference-area IPC research law. It
is not a general midpoint integrator, a coupled-rigid/rod/friction result, a
mesh-convergence result, or a performance claim.

## Fixed-mesh force and impulse

The fixture is a 20 mm cube, 27 nodes (2 divisions), density 1100 kg/m^3,
`mu=2000 Pa`, `lambda=5000 Pa`, initial clearance 200 micrometres, zero
initial velocity, gravity 9.81 m/s^2, activation width 100 micrometres,
physical area IPC coefficient 1e16 N/m^5, and a 10 ms duration. The same
mesh was evaluated by an independent FP64 midpoint implementation at the same
step and compared against the retained 0.5 microsecond FP64 reference. The
predeclared force metric is the maximum absolute difference between 1 ms force
bins divided by peak reference force; the impulse metric is relative error in
total normal impulse. Each gate is 0.25%.

| Native timestep | Accepted | Force error vs refined reference | Impulse error vs refined reference | Result |
| --- | ---: | ---: | ---: | --- |
| 15.625 microseconds | 640/640 | 0.08545% | 0.01081% | both gates pass |
| 62.5 microseconds | 160/160 | 1.5654% | 0.23594% | force gate fails |

The coarse run also fails the subsequently added 10 nm nodal-position
comparison against matching-step CPU midpoint: its maximum is 11.85 nm.
The fine run's maximum nodal position difference is 1.13 nm and velocity
difference is 0.707 micrometres/s. At the fine timestep, native force and
impulse differences against matching-step CPU midpoint are 0.000595% and
0.000188%, respectively. Fine-run total contact work is -3.03219e-10 J
versus -3.03920e-10 J in the matching-step CPU calculation. These are fixed
mesh results, not validation against a physical material or spatially
converged reference.

For the fine run the minimum recorded gap is 87.10 micrometres, the maximum
independently reconstructed full-step local residual L2 norm is
3.06e-11 N s, maximum summed momentum-balance error is 3.66e-11 N s, and
maximum position/midpoint-velocity defect is 2.92e-14 m. The maximum relative
error in independently evaluating the contact force law and physical
impulse is 9.91e-8. A separate initially active 90-micrometre-gap step also
passes the law and momentum checks; its positive contact work is not by
itself an energy-creation verdict because the initial barrier stores energy.

See [fine comparison](comparison-h15us-certified.json),
[coarse comparison](comparison-h62us-certified.json), the corresponding
`native-*-certified.json` traces and `cpu-*-final.json` inputs. The coarse
comparison intentionally exits 1 because it fails its gates.

## Failure, restoration and replay

The native regression exercises freefall, active contact, same-source
backward-Euler control and distinct runtime fingerprints; a deliberately weak
contact and fast downward start makes the midpoint remain clear while the
reflected endpoint crosses the plane. That transaction reports contact
failure and restores the physical state. A separate deliberately compressed
checkpoint yields an inverted reflected element and reports invalid
deformation with restoration. Its absolute tolerance is loosened only for
this failure-path test so element validation, rather than the prior momentum
certificate, catches it. The original tighter inversion attempt failed first
at the momentum certificate and is retained as `inversion-attempt-001`.

The exact-state comparison covers accepted FEM nodes/materials, contact
history, topology, scheduler, other persistent snapshot-owned state and
published rigid reactions. It compares completion cursors separately.
Replay from an already contacting state, deliberate late outer rejection and
restore are exercised. The projected rigid state of a static proxy is a
reconstructed diagnostic cache, not continuation authority: a rejected first
step can refresh that projection. It is not included in the physical
restoration claim. Status and contact-sample diagnostics may describe a
rejected attempt rather than the previous accepted step.

On a deliberately inverted fixture, parallel element checks can report
more than one valid first offending element. In repeated runs, the failure
category and restored physical state agreed, but the reported first element
and diagnostic bytes sometimes differed; see `inversion-replay-*.json` and
the earlier failed [suite run](ctest-certified.log). The test requires exact
status bytes on accepted replay, and on rejected replay requires exact
physical restoration and the same failure category, recording separately
whether diagnostic bytes matched. It does **not** claim deterministic
failure-detail ordering. The native regression passed eight consecutive
repetitions; see [repeat log](ctest-replay-repeat.log). The final native
suite passed 10/10 tests; see [suite log](ctest-final-replay.log). Final
harness build inputs are recorded in [inputs-final-harness.sha256](inputs-final-harness.sha256).

## Reproduction

From the test build configured at `tests/contact`, run:

```sh
cmake --build <build> --target metalrobo_plane_contact_probe numi_contact_fem_fp64_midpoint
ctest --test-dir <build> --output-on-failure -j 1
<build>/metalrobo_plane_contact_probe <build>/NumiMatter.metallib 0.000015625 640 2 ipc_area_midpoint 0.0002 physical contact reference
<build>/numi_contact_fem_fp64_midpoint 0.000015625 0.01
python3 tests/contact/midpoint_native_compare.py <native.json> <cpu.json> docs/evidence/contact-area-reference-20261107/fp64-fem-h0.5us.json
```

The 640-step probe itself also restores, reruns and checks late rejection.
It takes snapshots for verification; its timing is not a kernel or
production-throughput benchmark. The known coarse-step force failure and all
earlier attempt files are retained.
