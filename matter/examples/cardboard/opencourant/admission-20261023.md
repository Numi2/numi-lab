# Physical-box continuation: MUMPS works; implicit adhesion blocks admission

The unchanged, pinned OpenCourant source now builds and runs implicit elastic
mechanics on macmini. The required LAW117/TYPE43 adhesive interface still fails
before accepting a mechanics step. This continuation therefore stops at the
admission boundary in the approved plan. The full box has **not** been folded,
assembled or crushed. The native Matter solver remains frozen.

The priority is the first complete physical sequence. Its immediate prerequisite
is a working slow-stage adhesive interface, including stress-free activation at
the accepted overlap geometry. Meshing the shipping box or fitting more material
parameters cannot clear this prerequisite. No long campaign is scheduled.

## Measured result

Five short runs used one worker and the same MUMPS-enabled Engine. Each retained
receipt binds inputs, binaries, loaded libraries and outputs. Times include
Starter and Engine, and are not full-model performance estimates.

| Run | Native result | Elapsed, s |
| --- | --- | ---: |
| Original implicit elastic control | Completed, 11 cycles | 0.859 |
| Original implicit cohesive control | Cycle-zero `free(): invalid pointer`, Engine exit 3 | 0.785 |
| Instrumented implicit elastic control | Completed, 11 cycles | 0.840 |
| Explicit cohesive control | Completed; mechanics qualification not established | 0.931 |
| Positive-reference-height implicit cohesive diagnostic | Zero accepted cycles; nonlinear timestep limit, exit 2 | 0.957 |

For the synthetic elastic control, final mean displacement is 0.01007387 mm
against 0.01 mm and element internal energy is 0.005044616 N mm against
0.005 N mm: errors of 0.739% and 0.892%. These are instrument-control results,
not cardboard calibration. The titled converter differentiates impulses with
centered interior windows and one-sided endpoint windows. An endpoint-force
comparison at its unchanged time labels is invalid. The retained measurement
uses those windows and keeps applied-load work separate from reaction output.
The maximum reaction-curve relative error after this source-based alignment is
4.66e-7. Applied-load work, integrated using measured displacement increments,
is 0.005047992 N mm (0.960% above the target). All four elastic-control gates
pass at the unchanged 1% threshold; this is not mesh or time convergence.

The positive-height diagnostic changed only the initial top-node Z coordinates
from 0 to 1 mm, leaving the TYPE43 THICK field at zero. It did not test a THICK
card correction or establish equivalence to a zero-thickness interface. It
avoided the heap abort but failed to converge even after timestep reductions.
The debugger replay of the original failure reaches `free` from `resol_`;
the optimized backtrace does not establish the cause of the heap corruption.

The [implicit compatibility documentation](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/implicit_features_compatibility_intro_c.htm)
lists supported stiffness formulations separately from constitutive force
evaluation. TYPE43 is absent from its stiffness list. TYPE2 contact excludes
rupture and penalty modes in implicit analysis. Thus a material law being
present in source does not establish a supported implicit adhesive assembly.
No verified input-only replacement for the required combination has been found.

## What changed

The [build script](../../../tools/build_cardboard_opencourant_mumps.sh) compiles
source `cb760c206e346eff12161a2268b134ef9d48a4ef` with GNU Fortran, OpenMPI and
MUMPS, retaining the packaged Starter. No mechanics source was changed. The
Engine SHA-256 is
`97844136bf3f0c69b0c26ccb221eb1742d781c1bed7d7b09cf6237a6ec8dc31e`.
Distribution MUMPS/ScaLAPACK/LAPACK versions differ from the upstream example;
their identities and shared-library aliases are recorded explicitly.

The adapter now selects a pinned package or pinned MUMPS-build profile, verifies
the build receipt, and checks runtime identity on resume. New opt-in controls
request titled native histories. A provenance-checking reader measures the
elastic fixture. Paper shear, partial plastic reversal and rigid-rotation
instruments were added without changing the frozen parameter card; they have
deck-generation tests and **have not been solver-run**. Neither these fixtures
nor adapter resume tests establish native restart equivalence.

The first build exposed a Fortran module dependency-order error. Giving CMake's
scanner the same MPI/MUMPS definitions fixed build scheduling. A second attempt
was interrupted after measuring shared-filesystem I/O contention; its outputs
remain. Continuing the unchanged build on the guest filesystem completed in
42.66 s. That is incremental completion time, not total clean-build time.

The converter's first invocation with a long absolute input path also crashed.
A byte-identical input and title sidecar with the short basename `probeT01`
converted successfully. Both conversion attempts are retained; the filename
hypothesis has not been independently confirmed.

## Reproduction and resource limits

Use [reproduction instructions](evidence-20261023/reproduce.md) and
`python3 matter/examples/cardboard/opencourant/evidence-20261023/analyze.py`
from the repository root. The [evidence package](evidence-20261023/) retains all
five attempts, the debugger replay, build attempts, preregistered diagnostic,
conversion failures and machine-readable verdicts. The earlier
[packaged-runtime rejection](admission-report.md) remains valid for that binary.
All 18 scoped adapter/deck/measurement tests pass. The retained package manifest
verifies 202 files; tests and file integrity do not change the admission verdict.

The ARM64 Linux VM has a 12 GiB memory cap and eight virtual CPUs. Observed VM
RSS reached about 8.10 GiB plus 0.163 GiB host-agent overhead in a retained
snapshot; this was not continuous peak accounting. The largest per-process native RSS in
these runs was about 149.5 MiB. No representative shipping segment has run, so
the below-16-GiB shipping-model gate and credible full-campaign time forecast
remain unearned. No mass scaling or loading acceleration was introduced.

The task VM was stopped after collection, releasing its active memory. Its
installed dependencies, built Engine and run evidence remain available for
continuation. The verified, inactive interrupted host build cache was removed
(65.2 MiB allocated); its manifest and configuration are retained. The complete
successful build remains in the stopped guest. No research inputs or unrelated
processes were removed. See `setup/scratch-cleanup.json` for the filesystem
measurements and concurrent-activity caveat.

| Qualification axis | Verdict |
| --- | --- |
| Backend correctness | Elastic control measured; required implicit adhesion fails; not admitted |
| Numerical convergence | Not run |
| Component calibration | Incomplete; frozen analogue/assumed coefficients remain |
| Analogue validation | Not run |
| Physical assembly | Not run; activation/reference and released overlap remain unproven |
| Crush/impact prediction | Not run |

Resume only with a source-supported correction or a demonstrated free runtime
that passes the adhesive control. Then run one score–fold–bond–release–restart
strip, measure a representative segment, and proceed directly to the nominal
full sequence. Broad refinement and sensitivity campaigns follow that working
integration milestone. These prerequisites avoid another blind solver campaign
while retaining physical assembly and damage requirements.
