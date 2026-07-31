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

The package pins `mlx>=0.32,<0.33`. Native and shared Metal records carry an
ABI fingerprint; stale compiled extensions are rejected before GPU work.

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
    critic_observation_count=696,
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
network and writes a deterministic, fingerprinted artifact transactionally.
The Swift executable loads it without translating weights:

```sh
../build/bin/metalrobo_task_rollout \
  --metallib ../build/shaders/MetalRobo.metallib \
  --policy-pack runs/policy.policypack \
  --envs 32 --steps 48 --chunk 8 --scene terrain
```

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

```sh
metalrobo g1-locomotion export \
  --checkpoint /path/to/checkpoint \
  --output-dir /tmp/g1-export

metalrobo g1-locomotion sim2sim \
  --checkpoint /path/to/checkpoint \
  --official-model /path/to/unitree_mujoco/g1_29dof_rev_1_0.xml
```

## Research adapters

The source tree still contains isolated active-encoder adapters for numerical
oracles and unfinished manipulation/perception migration work. They are not
the public locomotion architecture, are not imported by
`mlx_policy_learning.py`, and must not be copied when adding a robot. New
robot mechanics enter through the native `EngineModel`/URDF route; new task
semantics enter through a TaskPack; new networks enter through a PolicyPack.
Only a genuinely new physics primitive, sensor modality, or task operator
justifies new native code.
