# Motion providers

Motion providers imagine kinematic trajectories. They are intentionally
separate from foundation action policies and from physical execution:

```text
prompt or constraints -> motion provider -> MotionProposal
                                             |
                                             v
                               retargeter / InteractionPack
                                             |
                                             v
                          controller -> Metal physics -> evidence
```

A motion proposal is useful teacher material, but it is not proof that a robot
can reproduce the motion. Joint limits, actuation, balance, collision, contact,
and the world remain authoritative in the native simulator.

## ARDY Core ONNX

The first provider runs
[`TREEIndustries/ARDY-Core-RP-20FPS-Horizon40-ONNX`](https://huggingface.co/TREEIndustries/ARDY-Core-RP-20FPS-Horizon40-ONNX)
at pinned revision `99da2ef6605784967141d3fa91754c0f99e8a65d`. It emits a
40-frame, 20 FPS `cskel27` human motion proposal from a 4096-element ARDY text
feature. The separately published
[`Llama-3-ARDY-Text-Encoder-ONNX`](https://huggingface.co/TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX)
can encode arbitrary prompts; upstream reference embeddings permit a smaller
exact runtime qualification without installing the text encoder.

Keep downloaded model graphs outside Git. `numi motion inspect` validates the
fixed graph contract and can recompute every SHA-256 before inference:

```sh
numi motion inspect \
  --model-directory /path/to/ardy-core-rp-onnx \
  --verify-hashes
```

Run a supplied text feature:

```sh
numi motion infer \
  --model-directory /path/to/ardy-core-rp-onnx \
  --output-directory /path/to/run \
  --text-feature /path/to/prompt-embedding.npy \
  --prompt 'a person raises the left hand' \
  --provider auto \
  --seed 7 \
  --verify-hashes
```

Or qualify with an exact upstream reference embedding:

```sh
numi motion infer \
  --model-directory /path/to/ardy-core-rp-onnx \
  --output-directory /path/to/run \
  --reference-text-features /path/to/text_encoder_reference.json \
  --prompt 'wave with the right hand while standing' \
  --provider cpu \
  --seed 20260803 \
  --verify-hashes
```

The versioned `numi.motion-proposal.v1` artifact contains:

- normalized source motion;
- root position and heading;
- 26 local joint positions and 27 global 6D joint rotations;
- joint velocities;
- four source-model foot-contact scores and thresholded contact hints;
- skeleton names and parents, model revision and graph hashes, prompt-feature
  fingerprint, seed, runtime provider, timings, output shapes, and output
  fingerprint.

The contact channels remain predictions, not measured forces. A retargeter
must map the human skeleton into robot-authored mechanics, and an
`InteractionPack` must preserve missing force/pressure masks rather than
inventing physical observations.

## Measured Apple Silicon qualification

On arm64 macOS, the prompt `wave with the right hand while standing` produced
40 finite frames. Ten denoising steps took `2.463 s`; decoding took `0.012 s`.
An exact repeat with the same seed produced arrays fingerprint
`18206200595415908c3a8ecbdfe83b75290d93760583ba745c68bc6664e7a3d6`.

ONNX Runtime's Core ML provider accepted only a partition of this export and
failed inside one delegated partition on the measured machine. `--provider
auto` therefore discarded that attempt and reran from the original seed on
the arm64 CPU provider. This is a working Apple Silicon inference path, not a
claim of current Apple GPU execution. The provider boundary allows a later
Core ML or Metal-native backend without changing motion-artifact semantics.

For presentation evidence, render the actual proposal rather than recreating
the motion manually:

```sh
python3 tools/render_ardy_motion.py /path/to/run /path/to/ardy-motion.mp4
```

The renderer labels the source prompt, skeleton, generated cadence, selected
runtime provider, and the explicit boundary that physics has not yet been
applied.
