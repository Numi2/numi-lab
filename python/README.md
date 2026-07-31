# MetalRobo Python and MLX

Python is the learning and data boundary, not the production simulator
runtime. The current locomotion architecture is:

```text
Swift rollout scheduler
        |
        v
native persistent MetalWorld
  + compiled TaskPack
  + compiled PolicyPack
        |
        v
compact actor/critic observations and transitions
        |
        v
MLX policy update
        |
        v
next fingerprinted PolicyPack revision
```

Physics, contact caches, actuator state, observation history, reward,
termination, reset, randomization, and counter-based RNG remain owned by the
native runtime. A policy update never imports those opaque simulator
internals.

## Install

Build the native library and Metal shaders first:

```sh
cmake -S .. -B ../build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ../build --target \
  metalrobo_task_program_check \
  metalrobo_task_rollout
python3 -m pip install -e .
```

The package pins `mlx>=0.32,<0.33`. It no longer builds a Python physics
extension: native and shared rollout records carry an ABI fingerprint, and
the learner rejects stale artifacts before an update.

## Generic MLX policy learner

`MLXPolicyLearner` owns only actor/critic parameters, optimizer state, and PPO
updates. It consumes a compact batch collected by Swift:

```python
from pathlib import Path

from metalrobo import (
    MLXPPOConfiguration,
    MLXPolicyBatch,
    MLXPolicyLearner,
)

learner = MLXPolicyLearner(
    actor_observation_count=480,
    critic_observation_count=495,
    action_count=29,
    configuration=MLXPPOConfiguration(
        hidden_sizes=(512, 256, 128),
    ),
)

batch = MLXPolicyBatch.from_numpy(
    actor_observations=actor_observations,
    critic_observations=critic_observations,
    latents=sampled_latents,
    old_log_probabilities=old_log_probabilities,
    old_values=old_values,
    advantages=advantages,
    returns=returns,
)
metrics = learner.update(batch)
learner.write_policy_pack(
    Path("runs/policy.policypack"),
    policy_id="locomotion_actor",
    library_path="../build/lib/libmetalrobo.dylib",
)
```

The canonical writer is implemented by the native library. It validates the
network and writes a fingerprinted artifact transactionally. The Swift
executable loads it without translating weights:

```sh
../build/bin/metalrobo_task_rollout \
  --metallib ../build/shaders/MetalRobo.metallib \
  --policy-pack runs/policy.policypack \
  --envs 32 --steps 48 --chunk 8 --scene terrain
```

For production collection, Swift launches the long-lived MLX learner, retains
the native simulator, and owns every rollout/update boundary. The first run
can initialize its PolicyPack directly from the compiled TaskPack dimensions:

```sh
mkdir -p runs/g1
../build/bin/metalrobo_task_train \
  --metallib ../build/shaders/MetalRobo.metallib \
  --native-library ../build/lib/libmetalrobo.dylib \
  --mlx-python .venv/bin/python \
  --python-root . \
  --initialize-policy unitree_g1_native_locomotion \
  --policy-pack runs/g1/initial.policypack \
  --updated-policy-pack runs/g1/policy.policypack \
  --deployment-policy-pack runs/g1/deployment.policypack \
  --rollout-pack runs/g1/latest.rolloutpack \
  --learner-state runs/g1/learner.safetensors \
  --envs 1024 --steps 24 --chunk 8 --updates 1000 \
  --learner-timeout-seconds 120 \
  --scene ground --verbose
```

The final critic value is evaluated against the accepted post-rollout state
inside the last Metal submission. It does not consume or discard a physics
transition. The training PolicyPack retains the diagonal-Gaussian behavior
distribution; the deployment PolicyPack contains the same actor revision
without exploration. Re-running without `--initialize-policy` restores the
model, Adam state, and native task-wide curriculum level from the learner
sidecar. The worker derives curriculum state from the fingerprinted rollout;
it does not trust a duplicate scheduler value. The current sidecar schema is
strict; a new simulator context begins a fresh synchronized curriculum window
because its environment episodes are also new.

PolicyPack v3 is the raw-Gaussian behavior boundary. Historical v2 packs remain
readable by the independent deployment evaluator with their original tanh
action semantics, but cannot resume PPO because that stochastic distribution
has no exact v3 migration.

The learner compiles each PPO minibatch as one forward/backward/clip/Adam
graph. Swift appends native chunks directly into a single preallocated rollout
arena and passes borrowed buffers to the native writer, so chunking bounds
Metal submissions without duplicating the complete rollout on the host. The
worker memory-maps the resulting artifact and validates its content hash in
native C++ instead of copying and hashing tens of millions of bytes in Python.

Run the focused handoff check:

```sh
python3 probes/mlx_policy_learning_check.py \
  --library ../build/lib/libmetalrobo.dylib \
  --output /tmp/metalrobo-learner.policypack
```

The check performs a real PPO update and publishes a PolicyPack. It does not
construct a world, schedule a transition, or submit physics.

## Tactile-dataset learning

The tactile dataset path remains Python/MLX because it is model and data work,
not simulator ownership. `metalrobo-tactile` ingests pinned LeRobot 3 seasons
through Arrow and PyAV, keeps train-only normalization, and seals EMA and
optimizer artifacts at checkpoint boundaries.

```sh
python3 -m pip install -e '.[tactile-dataset]'

metalrobo-tactile prepare \
  --dataset-root /path/to/origami \
  --output /path/to/contracts

metalrobo-tactile train \
  --dataset-root /path/to/origami \
  --stream-contract /path/to/contracts/tactile-stream.json \
  --manifest /path/to/contracts/dataset-manifest.json \
  --video-key observation.images.head_left \
  --video-key observation.images.wrist_left \
  --steps 100000 \
  --output runs/origami
```

The public Origami dataset card does not establish hardware action semantics
or fingertip wrench units/frames, so hardware execution stays blocked until a
deployment contract supplies them.

## Policy export and sim2sim

The bundled G1 policy utilities retain Core ML, MLX, and ONNX export plus the
independent official Unitree MuJoCo comparator. They operate at the
PolicyPack/deployment boundary and do not schedule MetalRobo physics.
Core ML preserves the actor's rank-1 input/output contract. Export performs
strict CPU/GPU parity and separately records the bounded all-compute-units
error caused by Neural Engine reduced precision; deployment should select
`MLModelConfiguration.computeUnits = .cpuAndGPU` for the validated
float32 path.

```sh
python3 -m pip install -e '.[g1-deployment]'

metalrobo g1 import-unitree \
  --official-repo /path/to/unitree_rl_lab \
  --library ../build/lib/libmetalrobo.dylib \
  --output-policy-pack /tmp/unitree-g1-velocity.policypack

metalrobo g1 export \
  --policy-pack /path/to/policy.policypack \
  --library ../build/lib/libmetalrobo.dylib \
  --output-dir /tmp/g1-export

metalrobo g1 sim2sim \
  --policy-pack /path/to/policy.policypack \
  --library ../build/lib/libmetalrobo.dylib \
  --official-model \
    /path/to/unitree_mujoco/unitree_robots/g1/scene_29dof.xml

metalrobo g1 sim2sim \
  --promotion-suite --seconds 20 \
  --policy-pack /path/to/policy.policypack \
  --library ../build/lib/libmetalrobo.dylib \
  --official-model \
    /path/to/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
```

`import-unitree` accepts only the clean RL Lab revision pinned by the native
G1 contract and verifies the official ONNX digest and deployment joint map.
It folds Unitree's term-major history and interleaved joint ordering into the
first and last dense layers once. The resulting PolicyPack therefore runs
through the ordinary generic Metal policy engine with no runtime adapter.

The promotion suite recomputes the low-level PD drive at every MuJoCo physics
substep and reports realized local velocity, displacement, saturation and
termination across idle, forward, reverse, lateral and yaw commands. Balance
without command response cannot pass it.

The old MLX-owned physics, world-state, contact, reset, reward, and rollout
adapters have been removed. New robot mechanics enter through the native
`EngineModel`/URDF route; new task semantics enter through a TaskPack; new
networks enter through a PolicyPack. Only a genuinely new physics primitive,
sensor modality, or task operator justifies new native code.
