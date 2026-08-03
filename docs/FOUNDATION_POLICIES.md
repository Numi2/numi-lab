# Foundation policies on Apple Silicon

Numi Lab can execute a large vision-language-action model on Apple Silicon as
a provider of finite action chunks. The foundation model proposes intent;
Numi's native Metal runtime remains the authority for simulation, contact,
constraints, control, rollout, and physical outcome.

The first adapter supports NVIDIA's staged `GR00T-N1.7-ApplePnP-V1` ONNX
export for Unitree G1. Despite its name, ApplePnP is a real-robot apple-fruit
pick-and-place policy, not an Apple-platform build. Numi supplies the missing
Apple execution path through ONNX Runtime's Core ML execution provider with an
explicit CPU fallback for unsupported operators.

## Reproducible workflow

Keep the large model and its Python runtime outside the repository:

```sh
python3 -m venv /path/to/groot-runtime/.venv
/path/to/groot-runtime/.venv/bin/python -m pip install \
    numpy onnxruntime huggingface_hub PyYAML

NUMI_FOUNDATION_PYTHON=/path/to/groot-runtime/.venv/bin/python \
    numi foundation fetch \
    --model-directory /path/to/groot-runtime/model

NUMI_FOUNDATION_PYTHON=/path/to/groot-runtime/.venv/bin/python \
    numi foundation inspect \
    --model-directory /path/to/groot-runtime/model \
    --verify-hashes
```

`fetch` resolves the requested Hugging Face revision to an immutable commit
and records it in `.numi-foundation-source.json`. `inspect --verify-hashes`
checks every manifest-pinned ONNX graph and fingerprints the external weight
files.

One deterministic runtime probe is:

```sh
NUMI_FOUNDATION_PYTHON=/path/to/groot-runtime/.venv/bin/python \
    numi foundation infer \
    --model-directory /path/to/groot-runtime/model \
    --output-directory /path/to/run \
    --synthetic-observation \
    --provider coreml \
    --seed 1701 \
    --verify-hashes
```

For a real observation, replace `--synthetic-observation` with
`--observation observation.npz`. The archive must contain `ego_view` with
shape `[1, 480, 640, 3]` and the seven named G1 joint groups declared by
`exported_leapp.yaml`.

Each run writes:

- `action_chunk.npz`: 16 decoded steps of named joint-position, joint-effort,
  navigation, and base-height commands;
- `evidence.json`: source revision, model and adapter fingerprints,
  observation and noise fingerprints, providers, tensor shapes, stage timing,
  host information, and the action-chunk fingerprint.

## Camera-to-teacher pipeline

The production bridge uses the robot's torso-mounted policy camera, the exact
post-step G1 generalized coordinates, and the existing native motion-teacher
artifact. Foundation inference never blocks a Metal submission:

1. capture one synchronized policy-camera frame and state row;
2. infer a finite named action chunk outside the simulator hot loop;
3. compile the accepted named joints into a full, rate-limited G1
   `InteractionPack`;
4. replay the pack through ordinary Metal physics with a native student actor;
5. distill executed teacher actions in proportion to measured stability and
   progress.

The concrete commands after capture and inference are:

```sh
numi foundation observation \
    --camera-frame /path/to/capture/frame-000000.ppm \
    --state-trace /path/to/capture/state.tsv \
    --output /path/to/run/observation.npz \
    --evidence /path/to/run/observation.evidence.json

numi foundation compile-interaction \
    --action-chunk /path/to/run/action_chunk.npz \
    --observation /path/to/run/observation.npz \
    --native-library /path/to/libmetalrobo.dylib \
    --output /path/to/run/foundation-teacher.interactionpack \
    --evidence /path/to/run/interaction.evidence.json \
    --id foundation-teacher-v1 \
    --desired-outcome "move away from the visible projectile while balanced"

numi train \
    --task ball-dodge --scene ground \
    --interaction-pack /path/to/run/foundation-teacher.interactionpack \
    --interaction-clip foundation-teacher-v1 \
    --interaction-student-authority 0 \
    --g1-visual-pack-dir /path/to/g1-visual-packs \
    --ball-visual-pack-dir /path/to/ball-visual-packs \
    --envs 1024 --steps 27 --updates 1 --chunk 1
```

Pure teacher control is marked in every transition. Its sampled student action
receives zero PPO weight because the student did not cause the physics, while
the executed action remains available for behavior cloning. Failed or
terminated steps receive no teacher weight. Stable partial progress receives a
continuous weight instead of being discarded behind a binary success gate.
With nonzero student authority, the student owns a residual and ordinary PPO
attribution remains valid; the absolute generated pose is not imitated twice.

The current G1 bridge maps waist and both seven-joint arms. The bundled 29-DoF
robot has no dexterous-hand actuators, so hand outputs are retained in evidence
but not executed. Navigation, base-height, and effort outputs are also retained
but not yet mapped. Lower-body targets stay on Numi's native standing posture;
this is an explicit first slice, not a claim that the upstream manipulation
policy already knows whole-body dodge motion.

## Qualified Mac mini result

On 2026-08-03, the complete model executed on a 24 GB Apple M4 Pro Mac mini:

| Measurement | Result |
| --- | ---: |
| Resolved model revision | `1c956f1f622cc6496f120efff43012dddeb2ff5e` |
| External model weights | `12.58 GB` |
| Total staged runtime | `57.26 s` |
| Backbone inference | `1.19 s` |
| Action-head inference | `9.33 s` |
| Total inference across all five stages | `10.52 s` |
| Maximum resident set | `12,083,544,064 bytes` |
| Peak memory footprint | `21,601,764,848 bytes` |
| Swap operations | `0` |
| Failed stages or non-finite outputs | `0` |

Core ML accepted `1,629 / 2,421` backbone nodes and `3,164 / 5,300`
action-head nodes. ONNX Runtime executed the remaining nodes on CPU. All 12
decoded action arrays were finite and had their declared shapes. Three runs
with the same observation and seed produced the identical array fingerprint:

```text
1df8dd8c084e3f5621c140c7f317eb217fd659ff8815f195a3612f92a995b22b
```

The current latency is acceptable for teacher generation, offline
distillation, trajectory proposals, and capability research. It is not yet a
real-time control loop. Roughly 44 seconds of the measured run was staged
session loading and Core ML compilation, exposing a concrete optimization
frontier: reduce unsupported graph partitions, retain or cache compiled
stages within unified-memory limits, and selectively port dominant operators
to MLX or native Metal. The provider-neutral action-chunk boundary does not
change as Apple hardware and runtimes improve.

## Real observation and teacher qualification

On 2026-08-03, Numi captured a real 640x480 torso-camera view of the authored
ball-dodge scene and synchronized it with G1 state at step 1. The exact
observation drove the full GR00T graph on the Mac mini, producing a finite
16-frame chunk. Numi resampled it from 30 Hz to 50 Hz, enforced authored joint
position and velocity limits, and executed the resulting 27-frame
`InteractionPack` through Metal:

| Measurement | Result |
| --- | ---: |
| Failed Metal environment steps | `0 / 27` |
| Terminations | `0 / 27` |
| Mean root height | `0.77766 m` |
| Maximum tilt | `0.06260 rad` |
| Mean tracking score | `0.97485` |
| Visual observation active during teacher rollout | `true` |
| Teacher-attributed transitions | `27 / 27` |
| PPO weight on pure-teacher transitions | `0` |
| Mean continuous distillation weight | `0.99175` |

A one-update MLX smoke distillation consumed those 27 native visual samples,
advanced policy revision 1 to 2, reported `policy_loss = 0`, and reported an
imagination loss of `0.0218893`. This validates the data path and attribution;
it does not establish dodge competence because no projectile trial completed
inside the short 27-step chunk.

## Claim boundary and integration path

The synthetic replay proves runtime determinism. The real-observation run adds
camera/state wiring, controller mapping, native physical execution, and one
teacher update. Neither proves that this pick-and-place policy can dodge, get
up, or safely control a real G1.

Task use requires three additional physical steps:

1. assemble calibrated Numi camera frames and named G1 state into the model
   observation;
2. map the proposed named chunk through the authored G1 controller contract;
3. accept teacher samples only from solver-measured successful trajectories,
   then distill them into a fast native PolicyPack.

No foundation-model output may write simulator state, contact state, or solver
outcomes directly.
