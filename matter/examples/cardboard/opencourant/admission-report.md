# ARM64 OpenCourant admission: stopped before migration

The packaged `latest-20261011` ARM64 solver cannot execute the required
implicit slow stages. Correctly parsed, force-controlled LAW1 and LAW117
nonlinear fixtures both stop at cycle zero with `Fatal error: MUMPS required`
and Engine exit 2. PCG selection is confirmed in the native listings. The
source also compile-gates the normal implicit dispatch behind `MUMPS5`.
This is a packaged-backend admission failure, not a constitutive calibration
result. See [source audit](backend-capability-audit.md).

The prescribed stopping rule was applied after these bounded controls. Native
Matter remains frozen at `bb02fe5010c761bd28c73b02c11815d04c2939e8`, with its
solver discrepancy open. No shipping mesh, score/fold fixture, bond activation,
assembly, compression or impact campaign was started. No solver source build,
paid tools, shell substitution, mass scaling or loading acceleration was used.

## What actually ran

The native ARM64 package digest matches the release asset. A native Ubuntu
24.04 ARM64 guest ran with 8 vCPUs and a 12 GiB memory cap. The exported decks
use solid bricks only. Fifteen short attempts preserve input repairs and
negative evidence. They were interleaved to repair instruments, rather than
executed as a completed confirmation sequence in the listed fixture order.

Three explicit executions terminated normally: one TYPE6/LAW112 brick (302
cycles), one TYPE43/LAW117 opening-to-failure brick (31 cycles), and the elastic
control (12 cycles). These are runtime completions only. The LAW117 deletion is
an expected fixture event, not a failed production box. Its coarse history and
missing title sidecar prevent reaction-force qualification: generic `varNN`
reaction channels must be treated as impulses. The derived CSV is retained as
such and is not used as a force or calibration result.

The paired implicit controls use 1 N total load and a manufactured 100 N/mm
stiffness, predicting 0.01 mm displacement. They never reach a mechanics cycle,
so no displacement accuracy is claimed. Earlier configurations with output
requests crash at cycle zero; removing title/TH/animation output exposes the
MUMPS fatal. Removing only the title card does not fix the crash. The same
elastic model executes in explicit mode with the title card present.

| Attempt | Disposition | Adapter stage elapsed, s | Stage exits |
| --- | --- | ---: | --- |
| adhesion-01 | invalid_fixture_input | 0.626 | 2 |
| adhesion-02 | runtime_execution_only_not_mechanics_qualified | 0.749 | 0/0 |
| explicit-with-title-01 | runtime_execution_only_not_mechanics_qualified | 0.659 | 0/0 |
| implicit-cohesive-01 | failed_cycle_zero_crash | 0.655 | 0/3 |
| implicit-cohesive-nonlinear-01 | invalid_engine_input | 0.664 | 0/2 |
| implicit-cohesive-nonlinear-02 | blocked_packaged_implicit_missing_MUMPS | 0.647 | 0/2 |
| implicit-elastic-01 | invalid_fixture_input | 0.599 | 2 |
| implicit-elastic-02 | invalid_fixture_input | 0.585 | 0 |
| implicit-elastic-03 | failed_cycle_zero_crash | 0.658 | 0/3 |
| implicit-elastic-nonlinear-01 | invalid_engine_input | 0.656 | 0/2 |
| implicit-elastic-nonlinear-02 | blocked_packaged_implicit_missing_MUMPS | 0.661 | 0/2 |
| implicit-without-th-animation-01 | blocked_packaged_implicit_missing_MUMPS | 0.658 | 0/2 |
| implicit-without-title-01 | failed_cycle_zero_crash | 0.660 | 0/3 |
| paper-01 | invalid_fixture_input | 0.588 | 2 |
| paper-02 | runtime_execution_only_not_mechanics_qualified | 0.720 | 0/0 |

All retained output hashes and both executed adapter versions verify. The first
adapter version was reconstructed after the run and its bytes match the
preregistered run-receipt hash exactly; both versions are retained. The full
runtime-file manifest, loaded-library and executable identities, inputs,
checkpoints, stdout/stderr and native listings are in
[evidence-20261016](evidence-20261016/). Run `python3 evidence-20261016/analyze.py`
from this directory to reverify the retained attempts and regenerate the
[machine-readable result](evidence-20261016/admission-result.json).

## Implementation and limits

The narrow adapter supports export/run/resume/collect, validates exact input
and runtime identity, records timeout/failure, and rejects altered evidence.
The final adapter snapshots its source and only resumes the immediately next
preregistered Engine stage from its predecessor checkpoint; a staged run uses
`--through-engine`. These final resume safeguards have integrity tests, but
native restart equivalence remains unexecuted. This is not a qualified resume
or history-preservation claim.

The final paper exporter fixes the velocity integral and transverse
constraints after paper-02. The final adhesive exporter adds `/TH/TITLE`,
a lower numerical timestep factor and denser history sampling, source-derived
synthetic force/work predictions, and dormant-activation/control decks. These
revised variants are export-checked but were not solver-run after the stop.
Paper shear, unloading and rotation fixtures remain missing. The parameter
sidecar retains the distinction between published values, transferred density,
source assumptions, solver defaults and synthetic instrument coefficients.
It does not identify a kraft/NSSC LAW112 model or starch cohesive parameters.

## Time and resource decision

The measured successful one-element adapter executions took roughly 0.66–0.75 s
including Starter; Engine native listings report about 0.02–0.05 s. These tiny
explicit models are not a representative shipping segment. The largest
observed process RSS across attempts is recorded in the JSON result. A host
snapshot before shutdown measured 3,336,928 KiB VM RSS plus 149,200 KiB Lima
host-agent RSS. This is a snapshot, not a continuous peak or shipping-memory
qualification. Guest memory was capped at 12 GiB; no full-model resource
admission was earned.

A credible full-campaign elapsed-time forecast cannot be produced from a
configuration that cannot execute the slow stages. No long campaign was
scheduled. The next prerequisite is a verified **free, prebuilt MUMPS-enabled
runtime** that passes these same tiny implicit controls; availability of such
an ARM64 package has not been established. Selecting PCG does not bypass the
compile guard. Obtaining a different runtime would be a new admission attempt,
not a claim that the present migration succeeded.

The temporary VM was stopped and deleted after all mounted run evidence was
copied and hash-verified. Its exact configuration and the image SHA-256 remain
saved. Only that task VM and its verified regenerable image cache were removed.
Measured remote available space rose from 60,952,268 to 63,953,288 KiB during
this cleanup: **2.862 GiB recovered**, with **60.991 GiB available**. Runtime
binaries and the original research trees remain. See
[scratch cleanup receipt](evidence-20261016/setup/scratch-cleanup.json).

| Qualification axis | Verdict |
| --- | --- |
| Backend correctness | Not admitted: implicit runtime blocked; explicit smoke only |
| Numerical convergence | Not run |
| Component calibration | Incomplete; analogue/assumed/synthetic fields retained |
| Analogue validation | Not run |
| Physical assembly | Not run |
| Crush/impact prediction | Not run |
