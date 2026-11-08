# American crow: first unbiased motor learning and amplitude diagnosis

Native execution: macmini, Apple M4 Pro, macOS 26.6. The exporter/study source
is committed at `49fcdc564` (initial exporter at `4b4341fd`); the research record
and profile are committed at `76b778c84` on `agent/american-crow-rl-20261107`.
Exact source, toolchain/runtime hashes and worktree state appear in each plan.

The first unbiased PPO trial is a **valid negative result**. A separate causal
output-amplitude intervention supports a narrower mechanism. No policy is
promoted, no adaptive learning rate is selected, and learned manipulation remains
unqualified. The existing deployment97 feeding policy is protected.

## Accepted endpoint measurement

`metalrobo_task_rollout --final-state-json NEW_FILE` optionally publishes all-lane
accepted world-coordinate body/q/v state, names, dimensions, field order and
run/world/task fingerprints. Native state readback is enabled only for the final
submission; preceding chunks remain resident. The file must be new and its
directory must exist. This observation does not alter actor inputs or introduce
a per-step host simulation.

The first calibration failed after the untraced chunk-8 run because final
configuration readback had not been enabled. The trace had enabled it in chunk1,
masking the missing endpoint request. That failure, exact executable, source and
all successful/failed outputs are retained. An earlier shell build invocation
also missed Homebrew's CMake PATH; its logs remain. The corrected full incremental
Release build passes. Calibration repeats use the same development conditions
and are not presented as new independent scientific units.

The corrected 33-lane, 160-tick neutral chunk1/8 and positive controls pass 15,840
accepted transitions. Neutral final states and rollout bytes are exact across
chunks. Lane0 q/scene values agree with the existing trace within 1e-7, and all
accepted endpoint goal scores agree with independent FP64 world-coordinate
distance calculations within 2e-6. Constant head .6 moves all 33 objects by
.2417–.2778 m versus matched neutral positions. Existing-file output rejects
before native execution and preserves its original hash. Raw calibration is
102,780,739 B, within its 128 MiB prospective bound.

## First learning prediction: contradicted

The committed study runner is `python/metalrobo/crow_object_learning.py`:

```sh
PYTHONPATH=python python/.venv/bin/python -m metalrobo.crow_object_learning \
  --root . --output NEW_DIRECTORY calibration
PYTHONPATH=python python/.venv/bin/python -m metalrobo.crow_object_learning \
  --root . --output THAT_DIRECTORY training
```

The learner begins with fresh GRU32/encoder32 parameters, zero actor output and
latent standard deviation .2. It imports only task fingerprints from an archived
policy; no scripted weights or positive-control bias enter training. All 18
actions retain native authority. Object truth remains critic-only; the actor
receives proprioception/history and genuine delayed RGB.

One process performs 16 updates ×160 ticks ×128 lanes, fixed LR .0003, gamma .999,
lambda 1, two full-batch epochs and unchanged joint clipping. Complete coupled
checkpoints at updates 8 and 16 preserve physics, contact, object/RNG/GRU, pending
camera, next policy, model and Adam. Intermediate generating rollouts are
intentionally overwritten by the ordinary trainer as prospectively declared;
the two checkpoint-generating batches and final artifacts are retained. The
exact seeded inputs allow continuation/reproduction.

The preregistered prediction requires both ≥20 mm paired mean endpoint-distance
improvement and ≥20 percentage-point success gain within 20 mm. Frozen initial
and final mean policies run in a fixed balanced order on 64 unused placement
draws from two seeds. Every paired initial actor/critic observation is exact.

| Frozen policy | Mean endpoint goal distance | Within 20 mm |
|---|---:|---:|
| Initial mean | 218.023739 mm | 0/64 |
| Final mean, update16 | 218.023739 mm | 0/64 |

Every endpoint object position is unchanged. The distance gain is exactly zero;
the prediction is contradicted. Increasing training reward is insufficient.
The unit is a counter-RNG placement draw conditional on one learner and one
apparatus, not an independent learner or geometry.

Retained diagnosis recomputes exactly from native records. All joint clip scales
are 1, so critic clipping does not suppress these updates. Actual actor Adam
displacement per update is .03391–.08465. Final per-head KL grows to .00210449;
the mean-policy head latent remains .00827–.08610. Neither retained batch8 nor
batch16 visits the 20 mm goal region. The closest score-derived distances are
70.50 and 25.38 mm; these derived diagnostics do not replace endpoint coordinates.

## Revised amplitude model: bounded prediction supported

`object-amplitude-01/model-revision.json` binds the parent plan, failed analysis
and retained diagnostics by hash. The revised hypothesis says the current mean
head command is insufficient for useful object displacement. Its alternative
is a coupled-posture or temporal-command problem.

A separate prospective study adds .2 only to the frozen learned actor's final
head-output bias15, preserving other parameters, GRU, normalization, action map
and exact task binding. The intervention can change future proprioception and
therefore other feedback action outputs. It is not a trained/deployed policy.
Two 32-lane, 160-tick native arms use matched unused placements, with no learning,
parameter sweep, retry or KL-target selection. Both declared thresholds must pass:
≥50 mm mean goal-distance reduction and ≥75% of objects moving more than 1 cm.

| Arm | Mean endpoint goal distance | Within 20 mm |
|---|---:|---:|
| Unchanged frozen mean | 218.520169 mm | 0/32 |
| Head bias +.2 | 56.978830 mm | 8/32 |

All 32 objects move more than 1 cm; mean goal-distance reduction is 161.541339 mm.
Both thresholds pass over 10,240 accepted transitions with zero physics rejection.
Every paired initial actor/critic observation is exact and all endpoint FP64
identities pass. This supports head-output amplitude as an immediate practical
bottleneck in this apparatus. It does not demonstrate an optimal learning rate,
learned visual manipulation, generalized contact, feeding or biological behavior.

The next comparison should continue fixed-rate learning on fresh native episodes
and test the frozen policy on fresh placement streams. These observed streams
are now development evidence; an offset-control result cannot qualify learning.

## Performance, storage and actual recovery

The learning process accepts 327,680 samples at 720.876 transitions/s in
454.558 s, zero failed native steps. MLX peak is 580,382,152 B, native retained
buffers 104,109,362 B and transient private buffers 76,476,188 B. These counters
are distinct and do not measure whole-system peak. A during-run snapshot and
post-run query report zero swap; no immediately pre-run reading was captured.
This is a bounded learning measurement, not sustained production qualification.
The raw learning study including calibration is 615,565,252 B before its summary,
within the declared 1 GiB bound; the amplitude follow-up is 66,228,086 B before
its summary, within 128 MiB. No video or image captures are produced.

`manifest.json` binds the complete selected raw evidence and retained runtime by
path, hash, mode and mtime, including failures. Compressed content deduplicates
against previous Git archives. `runtime-recovery.json` records exact binaries,
native library, three shaders, committed Python, canonical assets/curriculum,
coupled checkpoint8, expected checkpoint16 and all six frozen evaluation arms.

`restore_runtime.py` reads actual committed Git objects. `verify_recovery.py`
replays six evaluations with relocated assets/curriculum/policies, requiring
canonical rollout bytes and every accepted endpoint field to be exact. It resumes
checkpoint8 for eight updates and compares final policy/candidate/rollout bytes,
all model/Adam arrays and metadata, causal native world/retina/GRU components and
scientific learner metrics. Process-dependent time/allocator counters are recorded
separately. Results appear in `verification.json` only after actual execution.

Raw deletion requires Git-object content verification, successful restored-runtime
execution, remote HEAD agreement, unchanged hashes/modes/mtimes and closed handles.
Exact cleanup receipts preserve deletion sets. Archive checkout duplicates may
then be omitted while all evidence remains in Git. Shared Git providers and
unrelated work are protected. Full lifelong ecology, solid-food handling,
generalization and social learning remain unfinished.

## Completed committed-object recovery

Git content verification passes all158 captured files (127 unique contents),
729996028 raw bytes, adding273710558 compressed bytes. Actual Git restoration
recovers79 files /162030498B in a separate temporary path. All six native frozen
evaluations reproduce their original rollout bytes and every accepted endpoint
field exactly, including relocated canonical assets/curriculum. Checkpoint8
resumes eight updates and reproduces all41 model/Adam arrays, metadata, three
canonical policy/candidate/rollout files, all causal native components and
scientific learner metrics at checkpoint16. Process time and allocator counters
are recorded separately. No recovery reanalysis or retry is needed.
The exact temporary recovery set is hash/mode/mtime/closed-handle checked and
removed; its file manifest is in `verification.json`.

The full update16 checkpoint remains recoverable using the generic archive
restore tool with prefix
`.numi/runs/american-crow/object-learning-02/training/checkpoints/revision-00000017.trainingcheckpoint`.
Use the recorded exact runtime and learner recipe for further fixed-rate
continuation; it begins at lifetime tick2560/episode16, with new native episode
draws. The source/current feeding deployment remains separate.
