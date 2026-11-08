# Rod position precision and local momentum

The retained eight-node, fixed-support x-axis case at 0.25 m has a structural
FP32 representation problem. Among **all** world-space FP32 free-node
positions within 1 µm of the FP64 implicit reference, with the captured support
positions fixed, the least achievable maximum of momentum residual, kinematic
velocity residual and reference-velocity error is 0.0770653 m/s. The required
limit is 0.001 m/s. The search even permits arbitrary real-valued free-node
velocities; restricting velocities to FP32 cannot improve that bound. This is
a result for the declared straight, no-contact scalar model and those support
coordinates, not a proof about all rods or representations.

The finite search uses a minimax dynamic program over all coordinate
combinations, with a brute-force calibration on a small chain. Its independent
analytical stretch residual agrees with the previous full-energy finite
difference to 6.3e-8 m/s; the FP64 reference equation residual is 8.5e-10 m/s.
The tested subset of single-FP32 coordinates relative to the origin achieves
0.00929 m/s, also above the gate. That subset is not exhaustive, so it does
not rule out every possible local-coordinate representation. The complete
search, witnesses, input hash and predeclared conditions are in `result.json`
and `PLAN.md`.

The follow-up tests the existing `mrCompensatedScalar` arithmetic from Numi's
geometry infrastructure. On Apple M6, the strict-math Metal kernel evaluated
42 static axial momentum rows, including six retained rows and 36 newly
constructed rows across 0.4, 1.25 and 100 m offsets. All 42 single-world-float
calculations missed the 0.001 m/s gate; none of the paired-FP32 calculations
did. The maximum paired result was 2.15e-6 m/s. Its largest discrepancy from
an independent 80-digit decimal evaluation of the exact split inputs was
1.95e-11 m/s. The two GPU output byte streams matched; the exact oracle also
detected a deliberately perturbed result. Raw inputs, outputs, code, hashes and
plans are retained. This is a static arithmetic test, not a rod solve, physical
calibration, timing result or qualified coupled simulation.

The host standalone rod path currently casts incoming double positions and
attachment targets to single world floats before dispatch and reconstructs
output doubles from single floats. MetalWorld's `MRRodNodeStateGPU` likewise
contains a single position vector, and contact code reads it directly. A
complete repair therefore needs owned position corrections that survive input,
solve, contact geometry, support reactions, publication, save/restore and
rollback. Projecting a compensated result back to a single world float before
force evaluation would reintroduce the demonstrated failure. Existing
compensated helpers provide an arithmetic building block, not that state or
transaction contract. General bend/twist, moving and articulated contact,
multiple steps, exact restoration and useful-motion coverage remain open.
No production dynamics or the disabled CCD and AL experimental options were
changed.

From a checkout with the recorded source and Apple Metal toolchain, the
standalone arithmetic probe can be rebuilt and run from the repository root:

```sh
E=docs/evidence/rod-fp32-representability-20261108
xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -O3 -I include -c "$E/force_probe.metal" -o /tmp/rod-force-probe.air
xcrun -sdk macosx metallib /tmp/rod-force-probe.air -o /tmp/rod-force-probe.metallib
clang++ -std=c++20 -O2 -fobjc-arc "$E/force_probe.mm" -framework Foundation -framework Metal -o /tmp/rod-force-probe
/tmp/rod-force-probe /tmp/rod-force-probe.metallib "$E/inputs.bin" /tmp/rod-force-probe-a.bin
/tmp/rod-force-probe /tmp/rod-force-probe.metallib "$E/inputs.bin" /tmp/rod-force-probe-b.bin
cmp /tmp/rod-force-probe-a.bin /tmp/rod-force-probe-b.bin
```

The retained `output-a.bin` and `output-b.bin` are the actual trial outputs.
`python3 "$E/analyze.py"`, `python3 "$E/analyze_v2.py"` and
`python3 "$E/analyze_exact.py"` independently recompute the analyses from the
retained artifacts; the first also reads the previous evidence directory.
