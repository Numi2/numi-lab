---
name: numi-lab
description: Use when a user wants Codex to discover, run, inspect, or improve Numi Lab workflows or this plugin on Apple silicon, including NumiVivo chemistry and MD, NumiTissue, NumiBrain and NumanX, Numi Human biomechanics, Matter and NumiSolver, robotics, BirdFlow, or Numi Automata. Do not use for generic science explanations or unrelated runtimes.
---

# Numi Lab

Treat the assistant as the researcher and Numi Lab as the user-owned local
laboratory. Route intent across the whole suite without forcing molecular,
cellular, neural, or embodied work into a robot-training schema. Use the owning
runtime and its native tools; do not invent a second planner or simulator.

This Codex plugin supplies local workflow guidance, not the sibling runtimes.
Use the tools and execution access actually available in the current Codex
environment; a remote or cloud thread may not have access to the user's Mac.

## Scientific work is a closed loop

For research, model fitting, capability claims, or mechanism comparisons, use
[scientific-workflow.md](references/scientific-workflow.md) as the standard:
**question → instrument → prediction → controlled experiment → evidence → model
revision → new prediction**. Codex authors and improves the instruments and
models; native owners execute the experiment. Read the reference before the
first research trial. Inspection and routine repairs need no study ceremony.

Preregister quantitative predictions, controls, experimental units, validity
checks, analysis, and a bounded stopping rule before seeing the trial results.
Use `numi science --help` when discovered for a transparent notebook over the
owner executable. Keep exploratory observations separate from confirmation;
record negative and inconclusive results. Bind passed calibration, executable
model predictions, actual unit identities and portable input evidence. Preserve
interrupted attempts and inspect live workers before recovery. A model revision must cite the actual
comparison, preserve the previous model, and lead to a new test on unused
conditions. Do not claim the loop is complete at a build, successful command,
report, or revised narrative without testing the revised prediction.

## First response

Translate the user's goal into the nearest owning tool below. If local execution
is available, inspect that owner's installed command and existing artifacts,
then act. If it is unavailable, say what can be established from supplied files
and name the single local setup or access step needed. Do not make the user
choose a CLI command or supply a path that live discovery can find.

For common tasks, use [local-workflows.md](references/local-workflows.md). It
gives a short path from intent to preflight, native execution, and a result the
user can verify. Load only the selected workflow. Keep the answer readable:
**what ran or was found; measured result; evidence and limits; next action**.

## Start from live truth

1. Run `numi doctor` when machine or installation readiness matters.
2. For a `numi` dispatcher task or an unknown owner, run `numi context --paths`
   to discover capabilities, overlays, paths, revision, and extension points.
   Resolve Numi source relative to its `Runtime root` and user overlays relative
   to its `Workspace`. For a request that names a sibling such as `numivivo` or
   `numitissue`, inspect that owner and its help directly; run Numi context only
   if an overlay or cross-suite integration is relevant. Bound discovery: if
   context stalls, inspect the selected owner instead of retrying it.
3. Run `numi robots list` or `numi robots inspect ROBOT_ID` before configuring
   a robot; skip robot discovery for non-robot workflows. Use its authored
   capabilities and semantic roles rather than assuming G1 joints, humanoid
   sensors, or locomotion outcomes.
4. Read [suite-tools.md](references/suite-tools.md) for the requested domain,
   resolve the owning checkout and executable, then read its help. Use
   `numi <capability> --help` for discovered dispatcher capabilities. A sibling
   executable such as `numivivo` is not automatically a `numi` subcommand.
5. For solver choice, start with the family view from `numi solvers list`
   (filter with `--target` for an intended runtime), then inspect the family.
   Use its safe default or an explicit `--variant`; use `--implementations`
   only when backend detail matters. Create or resolve a fingerprinted profile
   and respect its role, exact selector, targets, and evidence boundary. Never
   treat every variant as an interchangeable `train` or `evaluate` backend.
6. Inspect the owning repository code when the request needs behavior that the
   installed commands do not already provide.

Discover missing inputs from context, catalogs, capability help, and existing
artifacts. Ask only when live discovery cannot resolve a required robot, task,
artifact, outcome, or hardware-arming choice; never guess paths, fingerprints,
physical results, or approval.

Keep the user-facing flow concise: lead with the outcome or blocker, summarize
large catalog/help/JSON output, and give the next safe recovery command. On a
failed run, inspect its typed failure and retained artifacts before retrying;
never duplicate an expensive workload merely because it stopped.

## Route the whole suite

Load only the relevant section of [suite-tools.md](references/suite-tools.md):

| User intent | Owning tools |
| --- | --- |
| Molecules, MD, electronic structure, QM/MM, reactions, target engagement | NumiVivo / `numivivo` |
| Cells, neural tissue, growth, glia, electrophysiology, tissue campaigns | NumiTissue / `numitissue`, `numitissue-examples` |
| Perception, memory, learning, motor policy, accepted neural consequences | NumiBrain / `numi-brain-*` executables |
| Coupled brain, body, muscles, materials and sensors | NumanX integration contracts and owning Brain/Lab executables |
| Anatomy, muscle routes, tendon loads, standing and joint mechanics | Numi Human / `numilab-human`, discovered `numi human` overlay |
| Rigid/articulated/deformable physics and solver configuration | Numi Lab / `numi matter`, `numi solvers`, native probes |
| Synthetic LIF/STDP cultures, virtual MEA, growth and protocols | `numi neurons` and Numi Neuron Lab |
| Robot packs, training, evaluation, drones and action/motion teachers | `numi robots`, `train`, `evaluate`, `drone`, `sapiens`, `foundation`, `motion`, `hyper-policy`, `residual-teacher` (as discovered) |
| Bird aerodynamics, flight and accepted-state visualization | BirdFlowMetal and the owning Numi flight integration |
| Artificial life, evolving ecology, genomes and conservative chemistry | Numi Automata |
| Surgical robot studio or NVIDIA/Isaac execution | Dr.Anmar companion skill and its runtime |

Resolve paths from live context, overlays, integration manifests, executable
locations, package manifests, and repository remotes. The reference lists
repository identities, not mandatory absolute paths or installed-status claims.
Check a selected remote host explicitly; do not assume a local checkout or GPU
is available there. Do not scan all repositories or build every tool to answer
one domain request. If discovery stalls, retain partial output and inspect the
resolved command/source directly; report the incomplete readiness check.

For a cross-suite workflow, establish each participant's revision, backend,
units, time base, schema, resource identity, and authority before coupling.
Preserve accepted-state transactions, rollback, checkpoint identity, and causal
observation boundaries. Contracts or adjacent tools do not prove an executable
coupling. Missing integration belongs in the lowest owning layer, with an
explicit qualification boundary; do not silently replace it with host loops,
synthetic observations, or duplicated force/rate authority.

## Deeper guidance

Load [human-standing.md](references/human-standing.md) for whole-body Human
execution, [native-execution.md](references/native-execution.md) for Apple-native
runtime architecture or source changes, and [evidence.md](references/evidence.md)
for completion, provenance, and hardware-arming rules. These references extend
the routing map; they are not prerequisites for an unrelated domain.

## Codex plugin upkeep

When the task is to update this plugin, compare the repository source, the
configured marketplace path, the installed cache, and the live CLI before
editing. Change the owning plugin source, validate the skill and manifest,
update the cachebuster, reinstall through its marketplace, and confirm the
loaded source and version. A new Codex thread is needed to exercise the new
skill snapshot. Preserve unrelated changes in the source checkout.
