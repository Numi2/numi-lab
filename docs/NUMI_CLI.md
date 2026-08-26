# Numi CLI

`numi` is a deliberately small doorway from Codex into Numi Lab. Codex owns
reasoning and adaptation; the CLI discovers and dispatches capabilities from
the installed runtime and the user's overlays.

## Core

```sh
./tools/numi doctor
./tools/numi context
./tools/numi run train --help
./tools/numi version
```

`numi doctor` validates the Apple/Metal toolchain, native trainer and rollout
executables, MLX availability, and robot-catalog list-to-inspect coherence. A
catalog executable built against an incompatible runtime library fails the
doctor check instead of advertising robot inspection that cannot execute.

Make the local runtime available throughout Codex with one command:

```sh
./tools/numi codex install
```

This registers the repository marketplace, installs and enables the Numi Lab
plugin, and links the dispatcher into `${XDG_BIN_HOME:-~/.local/bin}` without
replacing an existing command. Start a new Codex task after installation so its
Numi Lab skill is loaded.

Use `numi codex status` to verify more than registration. It fails when the
plugin is disabled, Codex reports another source or version, the versioned cache
is absent, or any cached plugin file differs from this runtime's source. This
prevents an old cached skill from being reported as current after a source edit.

`numi train` and `numi evaluate` are discovered commands, not core CLI logic.
The dispatcher searches in this order:

1. `<workspace>/.numi/commands`
2. `NUMI_COMMAND_PATH`
3. `${XDG_CONFIG_HOME:-~/.config}/numi/commands`
4. the Numi Lab source tree
5. the installed `libexec/numi` directory

The first executable with the requested name wins. A custom capability only
needs to implement normal command-line behavior and may optionally answer
`--numi-describe` with a one-line description for `numi context`.

`numi window` is the workspace's one-command live renderer. It builds an
isolated presentation runtime, finds the saved authored scene, and opens an
`MTKView` preview of environment zero:

```sh
numi window
```

The first setup only needs an authored `numi.visual-observation.v1` file:
place or link it at `.numi/window.visual-observation.json`. Subsequent preview
runs need no executable, metallib, scene, or display flags. The preview uses a
GPU-private three-slot ring and drops stale frames rather than delaying the
native rollout; closing the window ends that preview run. See
`numi window --help` for scene discovery and optional rollout overrides.

Motion imagination is another discovered capability. It executes a provider
without giving that provider authority over robot actions or physical truth:

```sh
numi motion --help
numi motion inspect --model-directory /path/to/ardy --verify-hashes
numi motion infer --model-directory /path/to/ardy --output-directory /path/to/run \
  --text-feature /path/to/text-feature.npy --prompt 'raise the left hand'
```

See [Motion providers](MOTION_PROVIDERS.md) for the artifact boundary and the
qualified native ARDY G1 and generic ARDY Core ONNX paths.

## User-owned overlays

Codex may create commands, instructions, profiles, robots, tasks, evaluators,
or other transparent files beneath `.numi`. The core does not parse a global
robotics schema and does not reject configuration owned by a capability.

```text
.numi/
├── instructions.md
├── config.toml
├── commands/
├── profiles/
├── robots/
├── tasks/
└── evaluators/
```

Generated training and evaluation evidence defaults to `.numi/runs`, which is
ignored by Git. Set `NUMI_RUNS_DIR` or `NUMI_RUN_DIR` to place it elsewhere.
Each bundled command records its arguments, source revision, stdout, and native
artifacts without changing the underlying runtime interface.

Generated contact-first intent uses the same generic training and evaluation
capabilities. The InteractionPack and selected clip compile into the native
task before either command executes:

```sh
numi train \
  --interaction-pack runs/ardy.interactionpack \
  --interaction-clip ardy-g1 \
  --interaction-reset-phase-probability 0.20 \
  --interaction-reset-maximum-phase 0.85 \
  --initialize-policy g1_contact_first

numi evaluate \
  --interaction-pack runs/ardy.interactionpack \
  --interaction-clip ardy-g1 \
  --policy-pack runs/deployment.policypack
```

The two reset controls are deliberately independent. The probability controls
how often training begins away from frame zero; the maximum phase controls how
late those resets may begin. For example, `0.20` and `0.85` retain 80% canonical
starts while exposing the policy to continuation states across the first 85%
of the generated motion. The legacy
`--interaction-reset-phase-fraction` remains available as a coupled shorthand.
Held-out policy selection omits all reset-curriculum options and evaluates the
complete physical trajectory from its canonical initial state.

For the bundled G1 forward-running task, canonical evaluation publishes mean
and per-environment peak and final forward displacement in meters plus
per-environment tracking, mean/minimum root height, mean/maximum tilt,
termination, and physics-error evidence. Forward reach is the primary task
outcome. Held-out selection uses a continuous locomotion-progress score rather
than allowing any small height or tilt tradeoff to veto forward progress;
device-rejected physics steps remain invalid evidence. Every candidate is
retained, and all saved checkpoints are compared directly with the immutable
incumbent so the result does not depend on checkpoint ordering.

For generated motions that need physical correction without a hand-authored
pose, run the solver-residual teacher:

```sh
numi residual-teacher run \
  --policy-pack runs/source.policypack \
  --interaction-pack runs/ardy.interactionpack \
  --interaction-clip ardy-g1 \
  --output-directory runs/solver-residual-search
```

This searches native stochastic policy residuals under normal gravity and
contact. Every physically valid trajectory contributes continuously according
to reference-relative tracking and its realized TaskPack reward. Smooth
rank-advantage weighting emphasizes the better physical realizations without
an elite cutoff; the physical Pareto frontier is retained as evidence. The
command writes an actor-distilled full continuation plus deterministic
deployment candidate. It does not promote that candidate; use the same
held-out physical rollout to compare deployment actor with deployment actor.

## Codex bootstrap

The plugin under `plugins/numi-lab` intentionally contains a small skill. It
teaches Codex to begin with `numi context`; changing robotics knowledge remains
owned by the live installation, its commands, and its source.

Representative activation, incomplete-input, non-trigger, hardware, and
evidence-boundary requests live in the skill's `evals/evals.json`. The stdlib
contract test validates that corpus and the plugin install/cache checker; prompt
quality still requires evaluation in a fresh Codex task after reinstall.

Validate the plugin from the plugin-creator skill root:

```sh
python3 scripts/validate_plugin.py \
  /path/to/MetalRobo/plugins/numi-lab
```

The initial repository plugin is distribution-ready source, not a personal
marketplace installation. Release packaging can install or register it without
coupling the runtime to a particular Codex version.
