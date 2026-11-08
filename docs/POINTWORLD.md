# PointWorld on Apple Silicon

PointWorld is an advisory visual world-model provider. MetalRobo remains the
authority for sensing, controllers, limits, collision, contact, task success,
and simulator state. A forecast can rank a finite candidate batch; it cannot
mutate a world or certify a physical outcome.

## Meaning of parity

Full parity means the Apple implementation matches the pinned release's input
ordering, model architecture, forecast semantics, uncertainty mapping, and
official DROID/BEHAVIOR metrics within the qualified oracle tolerances. It does
not mean translating CUDA kernels instruction-for-instruction. CUDA runs only
on the frozen RTX oracle; production uses Metal/MPSGraph inference and MLX
training with custom Metal sparse operators.

The immutable release identity is:

- PointWorld source `05484826dfef74cbe278a3974179a5a16705d35d`.
- Model repository `b9e2e19a4f2bd65922e1f6d70aa953fe70aa9dba`.
- Large DROID+BEHAVIOR checkpoint SHA-256
  `bb7a5b0d717b79363c75751531c2416b099b8fca28b2f799ff49ace7123af787`.
- DINOv3 source `54694f7627fd815f62a5dcc82944ffa6153bbb76`.
- Released DROID+BEHAVIOR normalization, PTv3 blueprint, and checkpoint
  contract hashes are embedded in `PointWorldModelPack`.

## Native ownership

```text
VisualFrameBatchV1 borrowed Metal buffers
    -> persistent Metal preprocessing and DINO/PTv3 inference session
    -> immutable PointFlowForecastPack
    -> task-owned target-region candidate score
    -> ordinary Numi controller and authoritative Metal physics
```

C++ validates and fingerprints model, observation, candidate, and forecast
artifacts. `MetalPointWorldPreprocessor` owns a persistent Apple GPU device,
queue, pipelines, and bounded staging for qualification. Its `encode()` method
borrows RGB-D and output buffers and neither commits nor waits. The full model
session must retain this composition behavior. MLX may own the differentiable
training graph and optimizer state but never the simulator or rollout schedule.

## Implemented qualification surface

- Strict released-model and DINO receipts, with exact large-checkpoint hash.
- Release-shaped 320x180 RGB-D observation compilation, rigid camera checks,
  world-frame normals, point identities, and deterministic 1.5 cm voxel/cap
  selection.
- Authored FP64 link-transform plus surface-geometry candidate compilation.
  It publishes robot points, normals, magenta release colors, midpoint velocity
  and acceleration, gripper state, and candidate-specific `dist2robot`.
- CUDA golden sealing with exact confidence derivation, named stage hashes,
  named timings, and complete observation/candidate/model provenance.
- DROID evaluation that requires the released expert-confidence mask and emits
  `full_eval/test/filtered_l2_moved/mean`; BEHAVIOR remains unfiltered.
- Target-region endpoint MPC ranking with an explicit uncertainty penalty.
- Live M4 Metal preprocessing probe with an output fingerprint and retained
  memory receipt.

## Frozen CUDA oracle workflow

The oracle is an evidence producer, not a deployable provider. Keep it in an
isolated environment on the RTX host and leave the pinned NVIDIA checkout
unchanged. After the released checkpoint, DINOv3 weights, and evaluation data
have passed their receipt checks, export a canonical batch with:

```sh
python tools/pointworld_cuda_oracle.py \
  --source-root /home/numi/pointworld-oracle-05484826/source \
  --checkpoint /home/numi/pointworld-oracle-05484826/assets/large-droid+behavior/model-best.pt \
  --output /home/numi/pointworld-oracle-05484826/golden/droid-test-000 \
  --split test --repeats 3 -- \
  --domains droid --data_dirs /path/to/released/droid
```

The exporter refuses a changed PointWorld or DINOv3 revision, records the
checkpoint SHA-256, captures canonical input and named stage arrays, measures
repeated CUDA output variation, and atomically writes a compressed NPZ plus a
JSON receipt. Metal tolerances are derived from these arrays; an absent array
or non-finite stage is a qualification failure, not permission to widen the
tolerance.

## Non-qualifying frontier

`numi world-model infer` and `train` remain fail-closed. Full completion still
requires converted DINOv3 ViT-L/16 weights, all large PTv3 sparse/serialization/
attention/pooling/heads operators and their MLX backward paths, the RTX golden
corpus, released dataset manifests and confidence artifact, official full
evaluations, warm-session profiling below the memory budget, and the held-out
Franka MPC comparison. Shader compilation or preprocessing success is not
released-model inference evidence.
