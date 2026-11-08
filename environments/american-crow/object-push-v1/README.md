# American crow: native object push v14

This is a bounded engineering fixture for developing physical manipulation.
The object is an initially resting rigid2g sphere, radius8mm, on an ordinary
colliding support. The articulated beak/head moves it through solved native
contacts. There is no grasp latch, launch velocity, host restaging, teacher,
food tracer or hidden object position in the actor. All18 native action channels
remain available. This is an estimated crow and contact model, not biological
calibration or a learned manipulation result.

## Runtime and observation contract

The ordinary Swift trainer/rollout selects `v14-object-push`. It requires the
retinal configuration beside this file and the explicit admitted32-substep
hybrid-CCD profile:50Hz control,1600 physics substeps/s,4 velocity iterations,
2 final iterations. The C API rejects a different substep count. An explicit
rollout `--physics-substeps` override remains visible and is validated.
The minimum supported timestep is not established for arbitrary scenes.

The native actor has1247 features:93 proprioceptive/history features and1154
retinal RGB/validity/age features. Two160x120 parallel engineering cameras
look10degrees down at the beak workspace;16x12 RGB features per eye update at
15Hz with2-control-tick latency. Object identities are instrumentation only.
The critic has100 features, including7 privileged object visibility, relative
position and relative velocity channels. No object orientation is supplied.

Reward is native accepted-state `exp(-squared_distance/0.01)` to
`[0.19,0.25,0.257]m`, with mechanical-power and action-rate costs, integrated
by the50Hz timestep. The target and support were developed on this apparatus;
they are not held-out geometry. This task currently supplies no persistent
nutrition/water/energy or social state. Ordinary coupled life/learner checkpoints
retain native world/contact/object/RNG/GRU and pending/visible camera state,
plus model and Adam state, at update boundaries.

## Use on macmini

From the source checkout:

```sh
cmake --build build --parallel 4
tools/numi crow-env object-push-admission --output NEW_DIRECTORY
tools/numi crow-env object-push-life-admission --output NEW_LIFE_DIRECTORY
tools/numi crow-env prepare-object-push --output ANOTHER_NEW_DIRECTORY
```

Preparation only authors scene assets and the sensor configuration; no physics
steps run. Existing output directories reject. The canonical configuration is
`environments/american-crow/object-push-v1/retinal-observation.json`.
For a fresh recurrent policy use `build/bin/metalrobo_task_train` with:

```sh
--birdflow-american-crow-journey --birdflow-journey-variant v14-object-push
--visual-observation-config environments/american-crow/object-push-v1/retinal-observation.json
--envs 8 --steps 160 --updates 2 --chunk 8
--minimum-difficulty-band 0 --maximum-difficulty-band 0
--initialize-policy UNIQUE_ID --zero-actor-output
--recurrent-hidden-size 32 --recurrent-encoder-sizes 32
--recurrent-memory-timescale-ticks 64 --recurrent-sequence-length 160
--initial-log-standard-deviation -2.995732273553991
--fixed-learning-rate --learning-rate 0.0003 --discount 0.999 --gae-lambda 1
--update-epochs 2 --minibatch-size 1280
```

Supply fresh policy, updated/deployment/incumbent policy, rollout and learner
output paths, native library/metallib and the locked MLX environment as shown in
the exact archived command. These are engineering admission settings, not a
selected learning recipe. Add `--checkpoint-life --checkpoint-interval 1` and a
fresh `--checkpoint-directory` to save complete coupled state. Resume in a fresh
process with `--resume-training-checkpoint PATH_TO_COMMITTED_DIRECTORY`, the
exact runtime, sensor inputs and learner configuration. The restored accepted
lifetime clock must match the manifest before any transition. A changed chunk
size is supported; a changed learner recipe is rejected.

For fixed-policy continuation, use `--save-life-checkpoint NEW_FILE` and then
`--resume-life-checkpoint FILE`, with the exact policy, `--no-scheduled-resets`
and `--repeats 1`. Saving/restoring life skips benchmark warmup. The common
`NUMILF01` envelope keeps an explicitly empty resource component for this
resource-free task; it does not invent a nutrition ledger.

## Evidence and limits

See `artifacts/crow-rl-object-push-20261108` for all failed development attempts,
prospective plans, exact source/runtime, native controls, ordinary learning,
performance measurements and Git recovery checks. Native controls use8 identical
lanes to verify batching; they do not establish8 independent physical cases.
A scripted head pulse is contrasted with opposite, neutral and contact-disabled
controls. Solved beak impulse samples, object movement, FP64 reward agreement,
initial real RGB visibility, chunk1/8 equality and fresh world+retina continuation
must all pass. Impulse evidence samples the final physics substep of each control
tick; it is not the total impulse over all substeps.

Resting penetration at4/8/16/32 substeps is2.217/.562/.148/.0444mm.
The4-substep push has a substantially different endpoint;8 substeps rejects a
CCD step. The16/32/64 endpoints are6.43/5.54/10.35mm from the original target.
Maximum32-versus64 trajectory difference is4.86mm and is not monotonic convergence.
The32-substep profile passes this fixture's0.5mm resting-depth control; broader
contact calibration, material sensitivity and held-out layouts remain required.

A two-update128-lane run accepts40960 training samples at734.69 transitions/s
in55.75s, with579605392B MLX peak and zero reported swap or physics rejection.
The8-lane run reaches70.96 transitions/s. Both retain32 physics substeps and
actual RGB, but the full-batch learner size changes, so this is not a controlled
learning-algorithm speedup. MLX peak is not whole-system memory.

No learned manipulation, transfer, sustainable production performance, or full
lifelong ecology qualification is claimed. Judge future policies by held-out
physical task outcomes, using causal perception/memory/contact interventions.
Do not select a policy or learning rate by KL target attainment.

Coupled continuation evidence is in `artifacts/crow-rl-object-life-20261108`.
The two-lane engineering probe checks five shared optimizer boundaries, each
with all41 model/Adam arrays exact, and fresh fixed-policy future streams at
chunks1/8 from tick47 (including pending camera state and a moving object).
Six invalid restores reject; the v13 feeding regression retains its food ledger
and all nine rejection controls. The initial checker treated flat feature
arrays as sample rows; its failure and source are preserved, and corrected
analysis reused the original successful runs without repeating them.
