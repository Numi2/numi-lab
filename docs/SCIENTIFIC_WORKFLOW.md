# Scientific work in Numi Lab

The standard loop is **question → instrument → prediction → controlled
experiment → evidence → model revision → new prediction**. Codex builds and
revises instruments and models; native owners execute experiments. The
[researcher guide](../plugins/numi-lab/skills/numi-lab/references/scientific-workflow.md)
defines controls, independence, uncertainty, and completion across the suite.

`numi science` supplies a local scientific notebook. It seals an authored plan,
executes one declared owner command at a time, retains its raw observations,
and records the comparison and model lineage. It has no planner, daemon,
remote service, or scheduling policy. It never implements physics or learning.

```sh
numi science register plan.json studies/study-001
numi science run studies/study-001     # next trial, once; repeat for each trial
numi science analyze studies/study-001
numi science revise studies/study-001 revision.json
numi science register followup.json studies/study-002 --parent studies/study-001
numi science verify studies/study-001
```

## Plan contract

Plans are JSON with `schema: "numi.science.plan.v1"`. See
[`prepare_neuron_study.py`](../examples/science/prepare_neuron_study.py) for a
complete authoring example. Required fields are:

| Field | Meaning |
| --- | --- |
| `question`, `hypothesis` | Observable question and falsifiable explanation |
| `model` | `version`, `statement`, plus authored parameters/structure |
| `owner`, `repository`, `backend`, `evidence_level` | Owning Git repository and actual execution scope |
| `instrument` | `description` and `calibration` evidence |
| `artifacts` | Absolute file paths to instrument, calibration, model, inputs, native binary, libraries and shaders |
| `design` | `intervention`, `controls`, `experimental_unit`, `allocation` |
| `observable` | `name`, `unit`, and JSON `path` array into stdout |
| `prediction` | `estimand: "paired_difference_mean"`, finite inclusive `minimum`, `maximum` |
| `validity` | Nonempty list of `{ "path": ["key"], "equals": value }` gates |
| `paired_equal` | JSON paths whose values must match within each pair |
| `trials` | Fixed execution order; each has unique `id`, `pair`, `arm`, `argv`, `env`, `timeout_seconds` |
| `limitations` | Calibration, inference, independence, and qualification limits |

An arm is `control` or `treatment`, exactly one of each per pair. `argv` is an
array, not a shell string. Its executable must be an absolute file path.
`{run}` in an argument expands to a fresh trial output directory, which is also
the process working directory. The owner must emit exactly one JSON document
on stdout; use a calibrated adapter if its interface differs, and retain its
original output alongside the extracted measurement. Extra artifacts written
under the output directory are hashed. Symlinks are rejected.

Only a minimal `PATH`, `HOME`, and `TMPDIR` are inherited; add required settings
explicitly to `env`. Never place credentials there: this is a retained record.
Bind transitively loaded model/runtime/input files in `artifacts`. The recorder
hashes the invoked executable automatically but cannot discover every library,
configuration, driver, or external dataset that an arbitrary owner may load.
Git revision, worktree status/diff hash, OS, architecture and host are retained.
An old prebuilt binary plus a new source checkout is not source-to-binary proof.

The local runner accepts software, reference and simulation evidence. Physical
instrument operation belongs to its owner's explicit arming/approval boundary;
do not bypass that boundary with a simulation label.

## Evidence and decisions

Registration happens before process launch. Inputs are checked immediately
before and after each trial. A kernel lock serializes mutations of a study.
Completed trials cannot rerun or overwrite observations. Timeout or interruption
kills the trial process group and records failure; it does not substitute zeros
or delete the partial output. A missing completion receipt is unfinished and
cannot be automatically restarted. Inspect `process.json`, the host/process,
and retained output before recovery; preserve the study and register an amended
one if it cannot be completed. Studies on the same GPU still need researcher
coordination; the notebook lock is not a device scheduler.

Analysis requires all declared receipts, including failures. A failed process,
invalid gate, nonfinite/nonnumeric metric, missing value, or mismatched control
makes the comparison inconclusive. Every declared trial remains in the table.
For valid data it computes treatment minus control per pair, their mean, range
and sample SD; the fixed inclusive interval determines supported/contradicted.
These are descriptive statistics, not a significance test, confidence interval,
causal proof, or guarantee that the selected experimental units are independent.
Use a preregistered domain analysis for designs this summary cannot express.

When an instrument fault makes the remaining trials pointless, use
`numi science stop STUDY --reason 'specific fault'`. It preserves the completed
receipts and blocks further runs. Analysis then retains each missing trial and
returns inconclusive. An unfinished process must be resolved before a stop.

Revision JSON contains `decision`, `model` (new version), `reason`, `evidence`
(an array containing the exact analysis SHA-256), `next_test`, and `limitations`.
An inconclusive comparison only permits an inconclusive revision; a contradicted
prediction needs revision or rejection. The next plan registered with `--parent`
must use the recorded revised model, with fresh observations and a new prediction.
Research judgment remains with Codex; the recorder cannot validate prose or
decide whether a mechanism is scientifically correct.

Files are exclusive, atomically published JSON records with SHA-256 bindings.
`verify` recomputes analysis from retained raw output, so a modified summary
does not pass merely because its own hash was replaced. This is local integrity
and provenance, not a signed external timestamp or protection against an owner
rewriting the entire evidence chain. Commit/share the registration hash before
collection when independent preregistration is required.

## Worked native experiment

Keep durable evidence outside a source checkout when changing sparse-checkout
patterns: Git may remove ignored directories outside the selected paths. Archive
the raw records before changing repository layout or cleaning builds.

If the prebuilt probe lacks the measurements required by the instrument, build
the unmodified native owner sources in an isolated directory:

```sh
python3 examples/science/build_neuron_instrument.py \
  --runtime /path/to/native/NumiLab --directory /path/to/evidence/native-instrument
```

Pass that directory to the authoring script's `--instrument-build`. The focused
build retains compiler/SDK, exact commands, source/header hashes and binary/
shader hashes. It is not the whole Lab integration build. Run the built probe's
native qualification before interpreting its measurements.

```sh
python3 examples/science/prepare_neuron_study.py \
  --runtime /path/to/native/NumiLab \
  --directory .numi/runs/science/my-study --seeds 3101 3102 3103
numi science register .numi/runs/science/my-study/plan.json \
  .numi/runs/science/my-study/study
```

The example authors a paired STDP study, calibrates its JSON reader against a
known value and invalid inputs, binds the prebuilt native runtime, and leaves
execution to the researcher. Check native help, GPU ownership, binary/library
paths, and unused seeds first. `--quick` is a bounded synthetic protocol and
cannot establish the full learning/biological qualification gates.
