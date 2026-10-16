# OpenCourant cardboard admission

**Admission stopped: packaged ARM64 implicit controls fail with `MUMPS required`.**
See [measured report](admission-report.md) and [retained verdicts](evidence-20261016/admission-result.json).

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
VM overhead must also fit inside the task's 16 GiB total ceiling. No source
build, paid solver, mass scaling, or physical loading acceleration is introduced.

The stdlib-only `matter/tools/cardboard_opencourant.py` implements just `export`,
`run`, `resume`, and `collect`; use `--help` on each command. Export runs on macOS;
solver execution requires Linux ARM64. Each attempt needs a fresh output directory.
Run receipts bind exact input, adapter, executable, configuration/runtime and
loaded-library hashes. Final run receipts also retain the executed adapter source.
A staged run uses `--through-engine`; resume accepts only the next declared
Engine stage and its immediately preceding checkpoint. Native restart equivalence
has not been executed. Resume verifies a completed parent, matching input and
binary identity, and the retained checkpoint hashes. It does not manufacture a
new stress-free material state.

`execution_complete` requires successful exits and native normal termination.
It is never a mechanics pass. Collection verifies retained output hashes and
reports measured solver RSS. Paper, cohesive, activation, restart and energy
checks require separate analysis of the actual histories. Parser errors,
mechanical failures and timeouts are retained separately.

The first fully integrated box sequence may start only after the component,
assembly and resource gates pass. A failed indispensable capability ends this
migration attempt, without extending the solver or replacing solids with shells.

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

The final paper and adhesion exporters include corrections made after the
retained native probes. They are export-checked and remain unexecuted; use the
exact decks retained under `evidence-20261016/runs/` to reproduce those attempts.
