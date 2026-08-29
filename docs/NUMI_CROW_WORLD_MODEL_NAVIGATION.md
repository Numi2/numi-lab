# Numi Crow v10 world-model navigation

V10 is the first Numi Crow development track that joins the qualified visual
flight actor to an authored obstacle/perch course, accepted-state RGB-D replay,
an MLX latent dynamics model, cross-entropy planning, and a deployable native
PolicyPack. The bird remains an estimated American-crow hybrid. The evidence
below is Apple-native simulation evidence, not measured animal or hardware
flight.

## Runtime boundary

- Task: `birdflow_american_crow_navigation_v10_world_model`
- Current development actor contract: 697 observations and 15 actions. It
  preserves v9's 684 inputs, inserts the original eight route values at actor
  index 84, and inserts a five-way stage one-hot at index 92 before the 600
  visual inputs.
- The current V10 task-fingerprint revision keeps every authored waypoint,
  reach radius, reward, and policy action unchanged, but previews the following
  route segment in the native yaw command inside 0.70 m of the first slalom.
  It is designed to make the measured alternating-slalom bottleneck bankable
  rather than changing what counts as five-waypoint completion. The earlier
  broad-preview screen uses a different task fingerprint and is retained as
  rejected diagnostic evidence. All prior V10 PolicyPacks must be explicitly
  transferred because this command semantic is fingerprinted.
- Sensor contract: four 16x9 instance-masked depth frames at history offsets
  0, 3, 8, and 18, plus 24 derived features
- V9/V10 training and rollout require an explicit
  `--visual-observation-config`. The runtime now fails closed when that config
  or its masked-depth sensor is absent; it cannot silently substitute 600
  zero-valued visual/history inputs.
- Course: two gate bodies, two slalom bodies, and a final perch, all ordinary
  static collision bodies in the Temporal Cone world
- Splits: `training`, `held-out-a`, and `held-out-b`; the same PolicyPack runs
  on every split. `development-reference` is a fourth invocation-only paired
  regression fixture and is never qualification evidence.
- Replay: canonical `numi.crow-replay.v1`, now including the full actor
  observation, course identity, scheduled-reset provenance, and RGB-D metadata

The course geometry is part of native physics. It is not a render-only prop.
Held-out splits transactionally change its default poses while retaining one
deployment ABI, allowing a single controller to be compared on unseen layouts.

### Turn-preview development status (29 August 2026)

Commit `ccfe463` adds a fingerprinted yaw-only preview at the first slalom
exit: position, reach radius, reward, action authority, and all five authored
waypoints are unchanged. A 3,964,928-sample, 32-lane PPO continuation reached
revision 122 with zero failed environment steps, but its deterministic
selection screen completed 0/8 lanes and did not reach waypoint one. A second
31-update protected pilot trained only actor columns 92 through 96 (the
five-way stage mode); it also completed 0/8 lanes with zero failed environment
steps. The earlier broad preview was separately rejected after it regressed
gate acquisition.

These artifacts are retained under the remote `numi-runs` workspace, but none
are candidates and no three-seed qualification was launched. Inspection of the
owning Metal RNG corrected the earlier fingerprint hypothesis: reset samples
are keyed by seed, environment, episode, control step, and channel, not by the
TaskPack fingerprint. The unstable comparison came from episode-local course
randomization and from confusing a one-repeat smoke with the earlier 20-repeat
development horizon.

`development-reference` now pins the five accepted course-body poses captured
from selection seed `2650817001`, environment 4, band 5, where the inherited
V33-r11 actor reached waypoint one. The invocation skips only the 15 course
position-offset operators. Root state, sensors, controllers, dynamics,
terminations, action authority, rewards, and all other authored randomization
remain live. The runtime labels this mode `paired-development-only`; the
qualification script still admits only the training and two held-out splits.
Use band 5 and identical seed/environment/horizon settings when comparing two
policies on this fixture. A pass here is a regression signal, not promotion.

## Model and deployment path

`numi crow navigation` exposes the complete path:

```sh
# Accepted native flight data. Collection disables scheduled resets by default.
numi crow navigation collect --milestone full-journey \
  --policy-pack /path/to/incumbent.policypack \
  --visual-observation-config \
    assets/crow_navigation_course/crow-navigation.sensor-fast.visual-observation.json \
  --envs 32 --steps 1600 --chunk 1 \
  --crow-replay-pack /path/to/training.crowreplay.json

NUMI_CROW_MLX_PYTHON=/path/to/python-with-mlx \
numi crow navigation model-train \
  --replay /path/to/training.crowreplay.json \
  --output /path/to/world-model

NUMI_CROW_MLX_PYTHON=/path/to/python-with-mlx \
numi crow navigation plan \
  --model /path/to/world-model \
  --replay /path/to/training.crowreplay.json \
  --maximum-action-delta 0.08 \
  --output /path/to/demonstrations.json

NUMI_CROW_MLX_PYTHON=/path/to/python-with-mlx \
numi crow navigation distill \
  --model /path/to/world-model \
  --demonstrations /path/to/demonstrations.json \
  --replay /path/to/training.crowreplay.json \
  --base-policy-pack /path/to/incumbent.policypack \
  --policy-pack /path/to/candidate.policypack \
  --output /path/to/student
```

The loader rejects mixed fingerprints, reset discontinuities, scheduled-reset
data, and ground-only datasets before training. Planning uses CEM inside an
explicit action trust region around the incumbent. Distillation initializes
the exact incumbent topology, retains all replayed incumbent actions, and
blends only a bounded planner correction. Model-predicted return is diagnostic;
it cannot promote a controller.

## 29 August 2026 five-waypoint development

The current development reference uses the authored 0.42 m waypoint reach
sphere, band 10, one-step submission chunks, 1,600 steps, and no scheduled
resets. Its strongest retained parent, revision 14, completes waypoints one and
two in all 32 lanes and waypoint three in 5/32 lanes on selection seed
`2650817001`; it completes neither waypoint four nor waypoint five. This is a
paired development result, not held-out qualification.

The continuation fixed four training-contract defects without easing the
autonomous task:

- V10 route extensions initialize at a zero physical/task mean instead of the
  masked-depth empty-pixel mean of one.
- The 697-input actor publishes an explicit stage one-hot. Transfer inserts its
  five zero-connected columns at `[92, 97)`, preserving inherited route and
  visual behavior. `--train-actor-observation-extension-only` with
  `--train-actor-observation-extension-count 5` can freeze the inherited actor
  and train only this adapter from fresh optimizer state while leaving the
  critic trainable.
- Curriculum stage three samples three nonterminal waypoint-three arrivals
  collected from the retained parent. Each reset restores accepted `q`, `v`,
  previous action and drive history, previous generalized joint velocity,
  navigation command, wing cyclic phase, and normalized journey phase. The
  first 84 actor inputs at reset match their replay source with maximum absolute
  errors of `0`, `0`, and `4.47e-8` across the three templates.
- V9/V10 visual execution now requires the authored sensor config. This closes
  a real evidence defect: recent route-parent training had omitted the config,
  so actor indices `[97, 697)` were identically zero despite the run being
  described as visual.

To reconnect vision without disturbing the retained controller, the MLX
worker can zero-connect an arbitrary actor-observation range while folding the
normalized raw-zero contribution into the first-layer bias. The transformed
revision-14 pack exactly preserved the paired 8-lane waypoint counts
`[8, 7, 1, 0, 0]` both with the old route-only input and with live masked-depth
input. A 131,072-sample live-vision continuation subsequently reached waypoint
three in 2/8 lanes at its best sparse screen, but its paired 16-lane result was
identical to the parent at `[16, 16, 1, 0, 0]`. It is rejected, not promoted.

Exact stage-three reset training was also bounded and stopped. A single-template
run and two three-template variants—route/stage columns and stage-one-hot
only—produced no waypoint-four completion. Their best sparse screen reached
waypoint three in 2/8 lanes, while the paired 32-lane route/stage candidate
regressed from 5/32 to 1/32 waypoint-three completions. These runs demonstrate
that replay-exact resets alone do not solve transfer back to the autonomous
route distribution.

The fixed promotion gate is three untouched randomized-course seeds with 32
environments and 1,600 steps per seed: at least 28/32 full five-waypoint
completions on every seed, at least 90% aggregate completion, and zero failed
environment steps. Selection seed `2650817001` is excluded from qualification.

No 29 August candidate is promoted and no three-seed qualification was
launched. The honest current milestone is reliable waypoint two, occasional
waypoint three, and zero waypoint-four/five completions. The next useful
controller experiment must improve autonomous transfer across a broader set of
waypoint-three arrivals; simply extending the rejected PPO runs is not a
pre-registered development plan.

## Retained 28 August 2026 artifact

This section describes the historical 684-input artifact. Its hashes remain
valid for that contract, but it is not compatible with the current 697-input
development ABI and is not evidence of five-waypoint reliability.

Three training replays supplied 4,796 contiguous transitions. Across those
replays, 66.40% of frames were above 0.35 m and the maximum replayed root
height was 1.4332 m. The retained latent model uses 128 hidden units and 32
latent units. Its held-out next-observation MSE was 0.63084 in normalized
observation space; this is a model-fit statistic, not a physical accuracy
claim.

The retained planner evaluated 512 start states over a 20-step horizon with
256 candidates, 32 elites, five CEM iterations, and a maximum per-action
deviation of 0.08. The student blended 10% of that correction and deviated by
at most 0.006513 from the incumbent on planner states. It preserves the
incumbent `684-512-256-128-15` actor topology.

Native qualification compared incumbent and candidate across three seeds,
32 environments, 1,600 steps, and all three course layouts: 18 runs in total.
All 15 predeclared gates passed with no failed Metal environment step or
contract failure.

| Course | Forbidden contacts | Total terminations | Mean height | Mean tracking |
|---|---:|---:|---:|---:|
| Training | 58 -> 28 | 105 -> 96 | 0.6256 -> 0.6713 m | 0.7998 -> 0.7250 |
| Held-out A | 51 -> 30 | 96 -> 96 | 0.6752 -> 0.6912 m | 0.7359 -> 0.6892 |
| Held-out B | 39 -> 26 | 97 -> 96 | 0.6349 -> 0.6651 m | 0.7908 -> 0.7163 |

The tracking regression is retained explicitly; it stayed inside the
predeclared 0.10 absolute retention bound. The candidate is promoted for this
simulated navigation contract, not declared universally superior.

Retained hashes:

- latent model: `a1eca4fabe38d9355c3a15032120ef47f7c72094344779ee6e6c0321b59cab3e`
- deployment PolicyPack: `d4eef1080a77203fa0b4e86394dcd46a99a374df119690a6413ea00792b0f50f`
- qualification payload: `f07f74cf06f83b67041d4396dd44cbbef1d9e500f7f05b7803de55308d3780e5`

The retained bundle is in
`artifacts/crow-v10-world-model-20260828/`. Full per-run native evidence and
training replays remain in the ignored `.numi/runs` workspace; their hashes
are locked by the retained qualification artifact.

## Qualification

Run the fail-closed matrix with explicit artifacts:

```sh
NUMI_CROW_WORLD_MODEL_ROOT="$PWD" \
NUMI_CROW_WORLD_MODEL_BUILD="$PWD/build-crow-world-model" \
NUMI_CROW_WORLD_MODEL_BASELINE=/path/to/incumbent.policypack \
NUMI_CROW_WORLD_MODEL_CANDIDATE=/path/to/candidate.policypack \
NUMI_CROW_WORLD_MODEL_RUNS="$PWD/.numi/runs/crow-v10-qualification" \
./tools/crow_world_model_qualification.sh
```

This qualifies control and collision outcomes. High-quality feather rendering,
multi-angle presentation, and movie quality remain BirdFlow responsibilities
and require separate inspection of real rendered frames.
