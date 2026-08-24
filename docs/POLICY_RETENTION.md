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

## One learned model

The production target is one multitask student artifact, not a directory of
independently selected specialists. Existing PolicyPacks are bound to exact
world, task, observation, and action contracts, so policies with incompatible
contracts cannot be merged by averaging or concatenating weights.

Until the multitask artifact format and executor exist, promoted specialists
remain frozen teachers. Consolidation must train a shared student from fresh,
deterministic teacher queries using explicit embodiment and task conditioning.
Schema-specific input and output adapters may surround one shared learned
trunk, but they are part of the same fingerprinted artifact and may not select
independent hidden policies.

The single student replaces its teachers only when matched held-out evaluation
demonstrates, for every retained capability:

- deterministic replay and fixed observation/action semantics;
- physical outcomes no worse than the corresponding promoted teacher;
- zero failed physics steps;
- no regression in safety, reset, contact, cadence, or transaction behavior;
- measured device throughput and memory within the authored promotion bounds.

Deleting teachers before those gates pass would discard learned progress, not
merge it. Raw rollout payloads can be regenerated from retained teachers and
the bound runtime, so they are not part of the long-lived model boundary.
