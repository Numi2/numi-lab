# OpenCourant cardboard admission

**Current milestone: repaired adhesive activation and LAW112 elastic iteration matrix; full-box admission remains open.**
Start with [the current integration status and gate review](integration-status-20261023.md).
The original packaged `MUMPS required` failure remains in the [October 16 report](admission-report.md); it is historical, not the current blocker.

This is a bounded external-backend admission experiment. Native Matter remains
frozen at `bb02fe5010c761bd28c73b02c11815d04c2939e8`; its solver discrepancy
is unresolved. None of these instrument fixtures establishes a calibrated
corrugated box, physical assembly, mesh convergence, or impact validation.

The pinned packaged backend is OpenCourant `latest-20261011`, source
`cb760c206e346eff12161a2268b134ef9d48a4ef`, Linux ARM64 double precision.
The package SHA-256 is
`ded9814e2460767353d591c89695ba1f76251291041943ab0ad73070a6c865f7`.
See [admission-plan.json](admission-plan.json) for the preregistered capabilities
and stop rule. Synthetic instrument constants are never recipe calibration.

## Runtime and adapter

On the designated macmini, run the repository's
`matter/tools/setup_cardboard_opencourant.sh ABSOLUTE_TASK_ROOT`. This verifies
both the downloaded archive and every extracted runtime file. It starts a
native ARM64 Lima VM with eight CPUs, 12 GiB guest memory, a 16 GiB sparse
virtual disk, no container runtime, and only the task directory mounted writable.
VM overhead must also fit inside the task's 16 GiB total ceiling. This was the
original package-only attempt. The October 23 candidate uses a separately
identified MUMPS source build and the narrow patches linked from the current
status; the original runtime remains unchanged. No paid solver or mass scaling
is used. Fast synthetic fixtures are instrument tests, not physical process predictions.

The stdlib-only `matter/tools/cardboard_opencourant.py` implements just `export`,
`run`, `resume`, and `collect`; use `--help` on each command. Export runs on macOS;
solver execution requires Linux ARM64. Each attempt needs a fresh output directory.
Run receipts bind exact input, adapter, executable, configuration/runtime and
loaded-library hashes. Final run receipts also retain the executed adapter source.
A staged run uses `--through-engine`; resume accepts only the next declared
Engine stage and its immediately preceding checkpoint. Restart equivalence has been measured for the small staged adhesive fixture;
full-board restart equivalence is still open. Resume verifies a completed parent, matching input and
binary identity, and the retained checkpoint hashes. It does not manufacture a
new stress-free material state.

`execution_complete` requires successful exits and native normal termination.
It is never a mechanics pass. Collection verifies retained output hashes and
reports measured solver RSS. Paper, cohesive, activation, restart and energy
checks require separate analysis of the actual histories. Parser errors,
mechanical failures and timeouts are retained separately.

The first integrated box sequence is provisional. Numerical refinement and
analogue validation follow on the frozen model. Correct mass, constitutive
history, contact, activation reference and release remain indispensable. A
failed physical strip prevents a credible full-box resource forecast; no long
box campaign should be launched behind that failure.

## LAW112 paper instrument decks

`python3 matter/tools/cardboard_opencourant_paper.py NEW_OUTPUT_DIRECTORY`
exports four one-brick MD/CD tension/compression cases. The exporter verifies
the source-data hashes in `paper-parameters.json` before writing. The decks use
the orthotropic `/PROP/TYPE6` property required by `/MAT/LAW112`, and request
brick stress/strain/internal energy and nodal displacement/reaction histories.
The small-strain gate compares the matching elastic stress component to the
published medium modulus times the declared signed strain (1% instrument
tolerance); it is not a fit or a same-lot validation.

The parameter sidecar enumerates the complete continuous-yield LAW112 card.
Published medium elastic/yield anchors are mapped across laws only for this
instrument. Plastic Poisson ratios, the missing hardening/surface coefficients,
and unidentified shear fields are explicit synthetic zeros or documented
large-value sentinels. The Popil density is from a separate coupon and is not
shown to be the same specimen as Haj-Ali's medium. These four coupons do not
exercise shear, plastic unloading, rigid rotation, flute geometry, adhesive,
crease, damage, or a box; those remain outside this exported subset and no
physical qualification is implied.

This section describes the original October 16 instruments. Later measured
exports, fitted tensile hardening, overlay bonds and physical strip attempts
are linked from the current integration status. Each historical run is bound
to its own input snapshot; regenerate neither its inputs nor its receipt.
