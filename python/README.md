# MetalRobo Python data and deployment tools

Python is optional tooling for tactile datasets, policy export, artifact
inspection, and independent MuJoCo comparison. It is not part of production
simulation or policy learning. The current locomotion architecture is:

```text
Swift rollout + PPO scheduler
        |
        v
native persistent MetalWorld
  + compiled TaskPack
  + compiled PolicyPack
        |
        v
compact learning tensors
        |
        v
in-process MLX Swift update
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
  metalrobo_task_rollout \
  metalrobo_task_train
python3 -m pip install -e .
```

The Python package remains optional. Its MLX dependency supports deployment
export and offline model/data work; the native learner pins MLX Swift in CMake.

## Native MLX Swift policy learner

Production PPO is implemented in
`bindings/swift/MetalRoboMLXLearner.swift`. Swift owns GAE, deterministic
minibatch scheduling, bias-corrected Adam, checkpoints, and PolicyPack
publication; MLX Swift supplies GPU tensor algebra and autodiff. Python is not
loaded or launched by the training runtime.

The canonical native writer validates each network and writes a fingerprinted
artifact transactionally. The Swift rollout executable loads the deterministic
deployment artifact without translating weights:

```sh
../build/bin/metalrobo_task_rollout \
  --metallib ../build/shaders/MetalRobo.metallib \
  --policy-pack runs/policy.policypack \
  --envs 32 --steps 48 --chunk 8 --scene terrain
```

The first training run initializes its PolicyPack directly from compiled
TaskPack dimensions:

```sh
mkdir -p runs/g1
../build/bin/metalrobo_task_train \
  --metallib ../build/shaders/MetalRobo.metallib \
  --initialize-policy unitree_g1_native_locomotion \
  --policy-pack runs/g1/initial.policypack \
  --updated-policy-pack runs/g1/policy.policypack \
  --deployment-policy-pack runs/g1/deployment.policypack \
  --rollout-pack runs/g1/latest.rolloutpack \
  --learner-state runs/g1/learner.safetensors \
  --envs 1024 --steps 24 --chunk 8 --updates 1000 \
  --scene ground --verbose
```

The final critic value is evaluated against the accepted post-rollout state
inside the last Metal submission. It does not consume or discard a physics
transition. The training PolicyPack retains the diagonal-Gaussian behavior
distribution; the deployment PolicyPack contains the same actor revision
without exploration. Re-running without `--initialize-policy` restores the
model, Adam state, and native task-wide curriculum level from safetensors. The
checkpoint is bound to the compiled task fingerprint and exact PPO contract. A
new simulator context begins a fresh synchronized curriculum window because
its environment episodes are also new.

PolicyPack v3 is the raw-Gaussian behavior boundary. Historical v2 packs remain
readable by the independent deployment evaluator with their original tanh
action semantics, but cannot resume PPO because that stochastic distribution
has no exact v3 migration.

Swift appends native chunks into one preallocated rollout arena, evaluates MLX
once per PPO minibatch, and installs the resulting weights directly into the
resident Metal context. The persisted rollout remains available for replay and
audit; it is not an inter-process transport.

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

The promotion suite recomputes the low-level PD drive at every MuJoCo physics
substep and reports realized local velocity, displacement, saturation and
termination across idle, forward, reverse, lateral and yaw commands. Balance
without command response cannot pass it.

The old MLX-owned physics, world-state, contact, reset, reward, and rollout
adapters have been removed. New robot mechanics enter through the native
`EngineModel`/URDF route; new task semantics enter through a TaskPack; new
networks enter through a PolicyPack. Only a genuinely new physics primitive,
sensor modality, or task operator justifies new native code.
