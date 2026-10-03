# Scientific work in Numi Lab

The standard loop is **question → instrument → prediction → controlled
experiment → evidence → model revision → new prediction**. Codex builds and
revises instruments and models; native owners execute experiments. The plugin's
`references/scientific-workflow.md` defines the research standard across the suite.

`numi science` records bounded scalar paired studies. It does not simulate the
owner, choose experiments, schedule GPU work, or infer a scientific mechanism.
The paths to this contract and the examples appear in `numi science --help`.
Installed packages include them under `share/numi/docs` and `share/numi/examples`.

```sh
numi science register plan.json /path/outside/checkout/study-001
numi science run /path/outside/checkout/study-001  # next declared trial, once
numi science status /path/outside/checkout/study-001
numi science analyze /path/outside/checkout/study-001
numi science revise /path/outside/checkout/study-001 revision.json
numi science register followup.json /path/outside/checkout/study-002 --parent /path/outside/checkout/study-001
numi science archive /path/outside/checkout/study-002 /path/to/new/archive
numi science verify /path/to/new/archive
```

## Plan contract

New plans use `schema: "numi.science.plan.v2"`. Existing v1 records remain
readable and verifiable, with their weaker guarantees explicitly reported.
Already registered v1 studies can finish through the retained legacy runner,
whose original file and hash remain intact. Their execution does not acquire v2
guarantees. New registration requires v2; v1 cannot be an enforced v2 parent.

| Field | Meaning |
| --- | --- |
| `question`, `hypothesis` | Observable question and falsifiable explanation |
| `purpose` | `exploration`, `confirmation`, or `replication` |
| `model` | Version, statement, parameters/structure and `training_units` |
| `model_file` | Absolute JSON file matching the declared model exactly |
| `predictor` | Executable command generating the model's quantitative prediction |
| `owner`, `repository`, `backend`, `evidence_level` | Owning Git repository and execution scope |
| `instrument` | Description, absolute calibration-report path, and instrument artifact paths |
| `artifacts` | Absolute files for code, model, calibration, inputs, native binary, libraries and shaders |
| `design` | Intervention, controls, experimental unit, allocation, and `unit_paths` mapping identity keys to observed JSON paths |
| `observable` | Name, unit, and JSON `path` into stdout |
| `prediction` | `estimand: "paired_difference_mean"`, finite inclusive minimum/maximum |
| `validity` | Nonempty list of `{ "path": ["key"], "equals": value }` gates |
| `paired_equal` | JSON paths that must match within each pair |
| `trials` | Fixed execution order: unique ID, pair, arm, `unit` identity, argv, env, timeout |
| `limitations` | Calibration, inference, independence, and qualification limits |

Each pair contains one control and one treatment with the same declared unit.
Different pairs must use distinct units. Analysis extracts the unit from actual
stdout and compares it with the plan; renaming a pair cannot hide seed reuse.
Confirmation rejects units already present in the model's training list or the
retained parent lineage, including calibration observations. Parent/child unit
identity paths must remain consistent. Replication may reuse conditions when
explicitly labelled. Honest unit definitions and complete training history remain
the researcher's responsibility; the recorder cannot infer biological independence.

Commands use absolute executables and argv arrays, without shell interpolation.
`{run}` expands to the fresh output directory and that directory is the process
working directory. Commands receive a recorded minimal PATH, HOME and TMPDIR,
plus explicit `env` settings. Never include credentials in retained settings.
Timeouts must be positive and at most one day. Instrument stdout must be one
JSON document; a calibrated adapter may retain and extract native output.

A predictor must emit:

```json
{
  "schema": "numi.science.prediction.v1",
  "model_sha256": "SHA-256 of the exact model file bytes",
  "prediction": {"estimand": "paired_difference_mean", "minimum": -0.001, "maximum": 0.001}
}
```

Registration executes and retains this prediction **before** sealing the study,
and rejects a model hash or interval mismatch. The executable predictor and all
its input files must be bound artifacts. A model statement alone is insufficient.

Calibration reports use `schema: "numi.science.calibration.v1"`, `status: "passed"`,
a nonempty `checks` array of `{ "id": "check-name", "passed": true }`, `scope`,
and `bindings` mapping every `instrument.artifacts` path to its SHA-256.
Optional `evidence` maps retained qualification files to hashes; every such file
must also be in the plan's artifacts. Optional `observed_units` excludes the
calibration conditions from study observations. A report is a claim by its owner;
inspect the actual checks. Parser calibration does not establish sensor accuracy
or biological validity. The native example executes qualification and a separate
frozen-weight control before it produces this report.

## Evidence, interruption and analysis

Registration retains content-addressed copies of every bound input, the predictor
run, environment, Git revision/status/diff hash, OS and host. A follow-up includes
a full parent notebook snapshot. Verification therefore works after relocation
or removal of the original source inputs and parent directory. Launching a new
trial still requires the original inputs to match their registered hashes.
Snapshots support inspection; restoring a runnable native installation may require
rebuilding its original layout and dependencies.

Bind transitively loaded files explicitly. The runner cannot discover arbitrary
dynamic libraries, drivers, implicit configuration, remote data, or code that
reads undeclared inputs. Source hashes plus an old executable do not establish
source-to-binary provenance; retain the owner build receipt and verify its inputs.

A kernel lock serializes study mutations. Inputs are checked before and after
execution. An attempt ledger outside the trial directory prevents a lost trial
folder from becoming a new trial. Finished attempts cannot be rerun. Timeouts,
SIGINT, SIGTERM and SIGHUP terminate the process group and retain failed receipts.
A leader that leaves child processes behind is invalid; the group is terminated
before output is sealed. Symlink and special-file outputs are described without
following or opening them and yield failed receipts. Owner commands must keep
writers in this process group and must not daemonize or write outside `{run}`.

After a killed runner, `status` reports an unfinished attempt. Inspect its
`process.json` and live host processes. When the original runner and entire native
process group have exited, use:

```sh
numi science recover STUDY --reason 'specific interruption and observation'
```

Recovery preserves partial output and seals an **invalid** receipt. It never
restarts the attempt or converts partial data into a successful trial. It refuses
live workers, another host, missing process identity, or lost evidence. A crash
between launch and process-record publication cannot be automatically resolved;
retain that notebook and preregister an amended study after inspecting the owner.
For an instrument fault, `stop STUDY --reason ...` retains completed receipts and
missing trials. Resolve unfinished processes before stopping. Stopped studies
analyze as inconclusive. Notebook locks do not coordinate different GPU studies.

Analysis requires every declared receipt unless explicitly stopped. Process
failure, invalid gates, missing/nonfinite metrics, unit mismatch, or mismatched
controls makes the result inconclusive. Every trial remains in the table. Valid
pairs yield treatment minus control, mean, range and sample SD. The fixed interval
determines supported/contradicted. These descriptive statistics provide no
significance test, confidence interval, coverage guarantee, or causal proof.
Preregister a suitable owner analysis for designs this scalar summary cannot express.

## Executable model revision

Revision JSON contains decision, new-version model, matching `model_file`, reason,
`evidence` containing the exact analysis hash, next_test and limitations. Decisions
are retain/revise/reject/inconclusive. Invalid evidence only permits inconclusive;
a contradicted prediction requires revision or rejection. Retain/inconclusive
cannot change scientific parameters. All prior and declared units remain in the
revised model's training history, conservatively including missing/invalid arms.
The notebook snapshots the revised model and binds its hash to the analysis.

The next plan with `--parent` must use this exact model. Its predictor executes
again before registration, so revised parameters must produce the new declared
prediction. Freeze that prediction before collecting unused confirmation data.

These are local integrity checks, not signatures or independent timestamps.
An owner who rewrites the entire chain can replace its hashes. Share the sealed
registration hash before collection when external preregistration is required.
`verify` means retained records agree; inspect its state, scientific verdict and
limitations before claiming a completed experiment.

## Native worked example

Keep durable evidence outside the checkout. Git sparse-checkout expansion and
cleanup may remove ignored outputs; registration rejects ignored destinations.
Archive completed notebooks before changing layouts. Archive destinations must
be new separate directories; the command copies and verifies, leaving the source.

Check GPU ownership first, then build the unmodified native owner sources:

```sh
python3 examples/science/build_neuron_instrument.py \
  --runtime /path/to/native/NumiLab --directory /path/to/evidence/native-instrument
python3 examples/science/prepare_neuron_study.py \
  --runtime /path/to/native/NumiLab --instrument-build /path/to/evidence/native-instrument \
  --directory /path/to/evidence/discovery --seeds 4101 4102 4103 --calibration-seed 4100
numi science register /path/to/evidence/discovery/plan.json /path/to/evidence/discovery/study
```

The authoring script verifies the build receipt, executes native CPU/Metal parity,
replay and transaction checks, and verifies a separate STDP-off control with the
actual observable. It aborts before registration if an old binary lacks the metric.
Execute all six declared trials, then analyze and fit:

```sh
numi science analyze /path/to/evidence/discovery/study
python3 examples/science/weight_effect_model.py fit \
  --study /path/to/evidence/discovery/study --output /path/to/evidence/model-v2.json \
  --revision /path/to/evidence/revision.json
numi science revise /path/to/evidence/discovery/study /path/to/evidence/revision.json
python3 examples/science/prepare_neuron_study.py \
  --runtime /path/to/native/NumiLab --instrument-build /path/to/evidence/native-instrument \
  --directory /path/to/evidence/confirmation --seeds 4201 4202 4203 --calibration-seed 4200 \
  --model /path/to/evidence/model-v2.json --purpose confirmation
numi science register /path/to/evidence/confirmation/plan.json /path/to/evidence/confirmation/study \
  --parent /path/to/evidence/discovery/study
```

The fit uses mean ± three discovery sample SDs, an explicitly heuristic predictive
range. After all follow-up trials and analysis, `weight_effect_model.py retain`
with the same output/revision options records supported unchanged parameters;
contradiction requires another explicit revision or rejection. Choose actually
unused seeds after inspecting prior evidence; these command examples do not
reserve or certify unused conditions. Read `examples/science/README.md` for the
executed results and limits. Quick synthetic weight change is not biological or
full learning qualification.
