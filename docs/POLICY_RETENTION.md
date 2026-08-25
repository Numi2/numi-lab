# Policy retention and consolidation

Numi Lab retains learned capability, not every generated training tensor.
Raw rollout payloads are transient once a candidate has passed matched
selection and the protected deployment PolicyPack has been published.

## Retained boundary

For a promoted run, retain:

- `deployment.policypack`, the immutable deterministic actor;
- `learner.safetensors` when optimizer continuation is required;
- `selection/selection.json` or `selection.log` and compact evaluation evidence;
- `arguments.txt`, `revision.txt`, `runtime.sha256`, and artifact hashes.

Do not commit raw `*.rolloutpack`, `*.policyrolloutpack`, intermediate
checkpoints, state traces, profiler captures, or repeated logs to Git. Git may
carry a selected deployment PolicyPack when it remains comfortably below the
host's ordinary file limit, together with a manifest that binds its source
revision, world, task, observation, action, and runtime fingerprints. Larger
promoted artifacts belong in immutable object storage referenced by content
hash, not in Git history.

Use the fail-closed retention command on individual completed run directories:

```sh
numi artifact-retain .numi/runs/example
numi artifact-retain --apply --receipt /path/to/receipt.tsv \
  .numi/runs/example
```

The first command is a dry run. Application requires a protected deployment
PolicyPack, revision, selection evidence, and no open rollout payloads.

## One learned model per robot

The production target is one multitask student artifact for each robot, not a
directory of independently selected task or experiment specialists. Vastly
different embodiments naturally retain separate networks. A G1 student, crow
student, and dove student are different production artifacts because their
mechanics, sensors, observation semantics, and action spaces differ.

Within one robot lineage, existing PolicyPacks remain bound to exact world,
task, observation, and action contracts. Task specialists cannot be merged by
averaging or concatenating weights. The consolidated robot policy must define
one stable superset observation contract, one action contract, and explicit
task conditioning.

PolicyPack v5 supplies the multitask artifact boundary: one actor can authorize
an explicit set of exact world/task pairs when they share observation and
action semantics. It does not manufacture a multitask actor or prove that one
is physically competent. Promoted specialists therefore remain frozen teachers
until consolidation trains a shared student from fresh, deterministic teacher
queries using explicit task conditioning. Schema-specific input and output
adapters may surround one shared learned trunk, but they are part of the same
fingerprinted artifact and may not select independent hidden policies.

Each robot's single student replaces that robot's teachers only when matched
held-out evaluation demonstrates, for every retained capability:

- deterministic replay and fixed observation/action semantics;
- physical outcomes no worse than the corresponding promoted teacher;
- zero failed physics steps;
- no regression in safety, reset, contact, cadence, or transaction behavior;
- measured device throughput and memory within the authored promotion bounds.

Deleting teachers before those gates pass would discard learned progress, not
merge it. Raw rollout payloads can be regenerated from retained teachers and
the bound runtime, so they are not part of the long-lived model boundary.
