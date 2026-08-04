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

## ARDY G1 ONNX

The default G1 provider is
[`TREEIndustries/ARDY-G1-RP-25FPS-Horizon52-ONNX`](https://huggingface.co/TREEIndustries/ARDY-G1-RP-25FPS-Horizon52-ONNX)
at pinned revision `d36d7069d514f9e6534c44c9fcf2463733298326`. It emits a
52-frame, 25 FPS `g1skel34` Unitree G1 proposal. Numi converts the generated
global rotations to local rotations through the model's authored hierarchy,
then applies NVIDIA ARDY's pinned G1-to-MuJoCo joint-frame and hinge-axis
contract. This is a native embodiment conversion, not human-skeleton IK.

The model is learned and can emit small position or velocity overshoots. Numi
retains the raw 29 joint coordinates, then computes the closest L2 trajectory
satisfying the authored Unitree position and velocity limits. The artifact
records every correction by joint. This mechanism projection does not splice
poses, smooth a landing, prescribe root dynamics, or change gravity/contact.

```sh
numi motion imagine-g1 --prompt 'do backflip' --seed 4
```

This command compiles ARDY's binary heel/toe predictions as predicted contact
modes with unknown force, wrench, CoP, area, and pressure fields. NumiSolver
then applies actuation, gravity, collision, and contact on every physical
substep. `--model-family core` selects the retained generic model and bounded
Core-to-G1 retargeter instead.

## ARDY Core ONNX

The first provider runs
[`TREEIndustries/ARDY-Core-RP-20FPS-Horizon40-ONNX`](https://huggingface.co/TREEIndustries/ARDY-Core-RP-20FPS-Horizon40-ONNX)
at pinned revision `99da2ef6605784967141d3fa91754c0f99e8a65d`. It emits a
40-frame, 20 FPS `cskel27` human motion proposal from a 4096-element ARDY text
feature. The separately published
[`Llama-3-ARDY-Text-Encoder-ONNX`](https://huggingface.co/TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX)
can encode arbitrary prompts; upstream reference embeddings permit a smaller
exact runtime qualification without installing the text encoder.

With the model cache and pinned Unitree description installed, the retained
Core-to-G1 path is:

```sh
numi motion imagine-g1 \
  --model-family core \
  --prompt 'perform a standing backflip, jump upward, rotate backward in the air, and land on both feet' \
  --seed 4
```

It encodes the prompt, runs ARDY Core, maps the resulting root and whole-body
motion onto the authored 29-DoF G1 within position and per-frame velocity
limits, applies the official Unitree visual meshes, and publishes a GIF, MP4,
source motion, retarget artifact, and provenance evidence. Optional paths and
render settings remain available through `numi motion imagine-g1 --help`; the
normal local cache locations require no repeated path arguments.

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

Provider selection is independent for the text encoder, denoiser, and decoder.
On the measured arm64 Mac, the INT4 text encoder retained its Core ML attempt:
its CPU comparison had cosine similarity `0.99999986`, relative L2 error
`0.000524`, and maximum absolute error `0.007064`. The current denoiser still
fails inside a delegated Core ML partition and therefore falls back by itself.
The decoder executes through Core ML but its strict CPU comparison currently
rejects a maximum absolute error of `0.018033`, so only that stage falls back.
Evidence records the selected provider, failures, timings, output shapes, and
CPU-parity result per stage. This is a mixed Apple Silicon inference path, not
a claim that every operation currently executes on the Apple GPU. A future
static-shape FP16/Core ML graph can replace either stage without changing the
motion-artifact contract or fallback boundary.

Merely replacing the public symbolic dimensions with `[1,64,148]`,
`[1,256]`, and their decoder equivalents is insufficient: the measured
specialized denoiser still failed in the same delegated Core ML partition.
The optimized export must also eliminate or specialize the internal
zero-length history `Slice`/dynamic selection path before FP16 MLProgram
qualification; Numi does not retain a static-labelled graph that only moves
the failure.

The reusable INT4 Llama 3 text encoder is a separate approximately 4.6 GiB
external-weight graph. On the measured arm64 Mac, it turns arbitrary prompt
text into the same `[1, 4096]` ARDY feature contract. Its Llama 3 license,
acceptable-use policy, source revision, tokenizer hash, graph hash, and
external-data hash remain part of the run evidence.

### G1 retargeting and presentation

`numi.motion-retarget.v1` uses the pinned official Unitree G1 29-DoF URDF and
bounded source endpoint IK. ARDY's continuous-6D root
orientation is transformed from its y-up frame into G1's z-up/x-forward
frame; limb direction and root translation drive the retarget while G1 keeps
its own proportions. Every joint target is constrained by authored position
and velocity limits, with torso/forearm self-clearance plus explicit left/right
knee, ankle, and shank separation objectives. The full 40-frame ARDY horizon
is retained in temporal order. The retargeter never replaces or extends its
tail, authors a ballistic arc, locks feet to the floor, or blends toward a
standing pose. The artifact records that zero dynamics frames were synthesized,
along with endpoint errors, self-clearance, maximum joint/root speed, link
transforms, source fingerprints, URDF hash, and joint order.

The Blender presentation consumes those exact link transforms and the
official Unitree meshes. It is deliberately downstream of the numeric
artifact, so a GIF cannot change or conceal the retarget. It may visualize the
raw kinematic proposal, but it cannot claim or cosmetically construct a
landing. Actuation, gravity, balance, collision, contact, and landing success
become evidence only after the joint-space intent is executed by NumiSolver;
the simulated root is never copied from the provider trajectory.

`numi motion imagine-g1` performs that physical execution by default. It
compiles the complete retargeted joint sequence into an InteractionPack with
ARDY's predicted foot-contact modes and per-sample confidence while keeping
force, wrench, CoP, area, and pressure fields explicitly unknown. It applies
zero student residual and records one native
solver configuration per control step. Rendering uses only forward kinematics
of those accepted configurations. The generated-motion root remains auditable
intent and an optional reward reference; after initialization it never writes
the simulated root. A physical fall therefore renders as a fall.

Interaction reset initializes the complete first-frame generalized state, not
only its configuration: root-link motion is converted to root-COM linear and
angular velocity, and joint velocities come from the next reference frame.
The implicit physical drives track both reference position and velocity while
retaining gravity, motor envelopes, effort limits, collision, and solved
contact. Joint/root references are interpolated at the control cadence, so a
25 Hz ARDY clip does not become a 50 Hz staircase. This prevents the controller
from damping every generated movement as if it were a sequence of static poses.

Generated fields are never replaced with plausible constants. Native ARDY-G1
and ARDY-Core retain the model's four foot-contact scores through embodiment
conversion and collapse them into left/right modes with confidence derived
from distance to the model's decision boundary. A provider with no contact
output, such as the current GR00T action-chunk adapter, authors no contact
tracks; it does not invent bilateral support. Unknown contact force fields use
empty validity masks, so their zero storage values cannot be consumed as
physical truth. Legacy binary-only imports default to zero confidence unless
the caller explicitly supplies confidence provenance.

Generated G1 motion is still kinematic intent, not a dynamically feasible
torque or contact-force trajectory. The action-level tracking policy therefore
observes root-link position, orientation, linear-velocity, and angular-velocity
errors in addition to joint, phase, and contact intent. Those signals let PPO
or distillation learn physical balance corrections, and the interaction reward
scores joint and floating-root velocities as well as pose. They never actuate the
floating root. Pipeline evidence reports achieved displacement and root/joint
tracking errors explicitly. A completed solver horizon means only that every
transaction remained valid, not that the requested motion succeeded.

`numi residual-teacher run` supplies the missing solver-in-the-loop bridge for
motions that are plausible but not dynamically feasible. ARDY remains the
nominal joint-space motion and the native stochastic policy supplies batched,
closed-loop residual candidates. Gravity, actuator limits, collision, and
contact remain active for every candidate. The evaluator publishes peak and
final forward displacement, tracking, mean/minimum root height, mean/maximum
tilt, termination, and physics-error evidence per environment. Numi reports
the non-dominated physical frontier, but does not use it as an admission gate.
Every physically valid trajectory contributes continuously according to
reference-relative tracking and its realized TaskPack reward. Smooth
rank-advantage weighting lets better physical realizations teach much more
strongly without an elite cutoff. A small actor-only Huber update moves toward
the weighted sampled residuals; critic gradient is zero in this update.
It writes both a stochastic full continuation PolicyPack and a deterministic
deployment candidate so matched evaluation never confuses exploration noise
with the actor that would ship.
The command never authors root state, support, contact, or a corrective pose.
Its output is only a candidate and must beat the source policy in a matched
physical rollout before deployment.

Generic InteractionPack tracking does not inherit standing-only upright,
height, tilt, or root-velocity penalties, nor height/tilt terminations. Those
contradict intentional rotation, get-up, crawling, crouching, jumping, and
other non-standing motion. The generated root trajectory supplies the
reference-relative position, orientation, linear-velocity, and
angular-velocity objective. Numerical solver failures still roll back
transactionally; mechanism limits, effort, slip, and forbidden contacts remain
physical costs; non-looping clips retain their finite timeout. Whole-body
generated motion uses a measured 64-manifold contact envelope (two Wave32
cohorts); this changes capacity only, not contact generation or acceptance.

For presentation evidence, render the actual proposal rather than recreating
the motion manually:

```sh
python3 tools/render_ardy_motion.py /path/to/run /path/to/ardy-motion.mp4
```

The renderer labels the source prompt, skeleton, generated cadence, selected
runtime provider, and the explicit boundary that physics has not yet been
applied.
