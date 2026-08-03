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

## Claim boundary and integration path

The qualified run used a deterministic synthetic observation. It proves that
the full foundation-policy graph executes on Apple Silicon and that Numi can
produce a reproducible typed action artifact. It does not prove that this
pick-and-place policy can dodge, get up, or safely control Numi's simulated
G1.

Task use requires three additional physical steps:

1. assemble calibrated Numi camera frames and named G1 state into the model
   observation;
2. map the proposed named chunk through the authored G1 controller contract;
3. accept teacher samples only from solver-measured successful trajectories,
   then distill them into a fast native PolicyPack.

No foundation-model output may write simulator state, contact state, or solver
outcomes directly.
