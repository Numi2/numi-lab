# Adhesive repair: bounded native milestone

The implicit TYPE43 crash and late-birth zero-area defect have targeted fixes.
An accepted-geometry bond can now be set during a stationary explicit step,
then resumed in implicit analysis without losing its reference or history.
This qualifies the recorded synthetic fixtures only. The full corrugated box
has not been physically assembled or crushed.

Parent evidence is commit `818e0a04`; upstream OpenCourant remains
`cb760c206e346eff12161a2268b134ef9d48a4ef`. The five-file patch, source hashes,
build flags, build receipts, exact inputs, restart files, native histories,
failed attempts, and postprocessing receipts are retained in
[`adhesive-fix-20261023`](adhesive-fix-20261023/).
Candidate 03 Engine SHA-256 is
`bf7d5947034c8357b70c0c3e7f023e6535491d0e868530bfb614d2777f30f3d6`.
The packaged Starter and runtime libraries are unchanged.

## What was repaired

Valgrind located the first invalid write in `SRCOORK`, called through
`S8ZKE3`: the zero-thickness TYPE43 connection was routed through ordinary
volume-brick stiffness storage. The later allocator failure was a consequence.
The repair assembles a surface quadrature initial iteration matrix for
TYPE43/LAW117. It uses the larger of the two elastic stiffnesses as an isotropic
positive initial matrix, not a claimed consistent damage/geometric tangent.
The native constitutive force and history routines still determine equilibrium.
Other element/material routes are unchanged.

The displacement-based LAW117 branch also saved an undefined `gc` restart
slot. It is now initialized; it is unused by that branch's force law.
Finally, TYPE43 reference areas were only initialized at time zero. A dormant
element skipped that call, so it could later develop stress but transmit no
force. The first active call now initializes its previously unset area and
local frame at the current geometry. Existing nonzero reference areas and
accumulated histories are preserved.

Time-triggered `/ACTIV` inside a single implicit run is still defective:
implicit trial rollback restores dormant element state while the activation
flag remains set. That failed run is retained. The admitted fixture instead
uses native checkpoints: implicit motion to 0.490 ms, explicit stationary
bond setting through 0.550 ms, then implicit loading to 1.000 ms. No substrate
coordinates, material history or restart contents are rewritten. Real assembly
must additionally gate activation on accepted physical overlap/contact.

## Measured results

| Check | Result and scope |
|---|---|
| Original force-controlled implicit cohesive fixture | 11 cycles; displacement 0.010 mm and work 0.005 N mm; existing strict measurement passes |
| Free-interface staged bond birth | Active state changes 0→1 at 0.500 ms; initial 0.050 mm gap becomes stress-free reference |
| Loaded staged bond | 9.900924 N versus 9.900990 N analytic prediction; final incremental gap 0.0990092 mm |
| Cohesive work | Integrated native traction work 0.49014108 N mm versus stored 0.4901415 N mm |
| Rotated frame held-out fixture | Maximum traction difference 1e-15 MPa; same gap and active history; energy difference 1e-26 N mm |
| Valgrind on birth and subsequent implicit restart | Zero errors in each run |
| Separate adapter resume into a new directory | Native history identical to uninterrupted Engine sequence except its wall-clock timestamp |
| Diagnostic-on/off histories | Every byte outside the validated 24-byte timestamp record is identical |

The separate [measurement report](adhesive-fix-20261023/bond-measurements.json)
retains failed/incomplete cases. Its two split-directory entries are not full
protocol measurements; their preservation check is the native-history
[checkpoint comparison](adhesive-fix-20261023/checkpoint-and-diagnostic-comparison.json).
The original single dormant element that jumped over the event has only one
sample and fails coverage despite normal termination. The original implicit
activation case has full time coverage but remains dormant and also fails.

An earlier switch exactly at the start of motion discarded nonzero kinetic
energy when entering static analysis. It is rejected and retained. The accepted
switch is inside the stationary hold; its energy is at the measured numerical
noise floor. Raw external-work counters reset at Engine restarts and are not
compared as if they were one continuous counter.

## Gate review

- Retain native force equilibrium, full protocol coverage, finite geometry,
  contact feasibility, irreversible history, activation at the accepted
  geometry, energy/momentum accounting and eventual release of all assembly
  fixtures. No tolerance relaxation, mass scaling or forced interior folding.
- Remove relative displacement-increment and relative increment-energy as
  mandatory termination criteria at a zero-motion, zero-work hold. Their
  denominators vanish. The staged fixture uses the native force criterion at
  the unchanged `1e-6` tolerance, plus separately measured absolute energy
  continuity. The rejected hold had force residual about `2.4e-8` while its
  relative energy ratio was dominated by work around `1e-15 N mm`.
- A prescribed material-point/element law check uses measured traction versus
  separation and work. Kinetic energy in prescribed moving nodes does not
  invalidate that constitutive check. Quasi-static specimen and box claims
  still require inertia and speed sensitivity checks.
- Full mesh/time convergence and the entire mass/drop matrix follow the first
  clearly provisional complete sequence. Their original prediction-qualification
  tolerances remain. A provisional peak can select a provisional scenario mass,
  but cannot be called a qualified box capacity.
- Missing failure-locus observations or unidentifiable local parameters do not
  prevent fitting an available whole-fixture observable. They remain unavailable
  verdicts and sensitivity dimensions; they cannot be invented or called validated.

## Remaining boundary

The 100 MPa/mm, 10 MPa, 1.5 N/mm adhesion coefficients are synthetic instrument
values. They are not a starch or hot-melt calibration. Pure opening, shear,
mixed loading, unloading and partial-damage closure histories are retained, but
this milestone does not establish contact after complete bond deletion,
nonuniform closure-patch birth, a released finite-mass impact, or full-box
contact. In particular, LAW117's asymmetric option still carries elastic
compression while intact; adding an independent contact penalty indiscriminately
would double-count compression. That integration must be resolved explicitly.

The next physical-model work is a contacting score/fold strip, source-derived
liner/medium hardening, conforming closure-patch mappings, fixture release and
a measured shipping-mesh resource estimate. Exact-recipe fracture and bare-box
impact validation remain unavailable from the retained public sources.
