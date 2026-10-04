# Local Numi workflows

These are entry points for a session with local execution on an Apple Silicon
Mac. They are not proof that every sibling runtime is installed or that a run
is qualified. Select one workflow, read its owner's live help, and use the
owner's own command and artifact contract. Do not turn these into a fixed task
catalog or a general shell-command launcher.

For research questions within any workflow below, apply
[the standard scientific loop](scientific-workflow.md). These domain recipes
select the owner; the scientific loop determines how to make and revise claims.

## 1. Find what this Mac can do

- **Discover:** `command -v numi`, then `numi doctor` if readiness matters and
  `numi context --paths` for the selected workspace. Discover sibling
  executables separately with `command -v` and their package manifests.
- **Present:** a short table of available domains, resolved owner/command,
  readiness, and one useful next action. Say `source available`, `installed
  command`, and `execution verified` only when each has its own evidence.
- **Recover:** if `numi context --paths` stalls, stop waiting and inspect the
  selected owner directly. If local execution is unavailable, use supplied
  files or connected repositories and state the missing access.

## 2. Run prepared NumiVivo molecular dynamics

- **Discover:** resolve the `numivivo` owner and read `numivivo md-help` plus
  the preparation or protocol help for the supplied system. Locate its
  topology, parameters, coordinates, backend, and any restart receipt.
- **Preflight:** check units, atom identity, periodic geometry, numerical
  profile, output capacity, and supported CPU or Metal backend. Inspect an
  existing trajectory or checkpoint before starting a new run.
- **Execute:** invoke the current `md-run` or `md-protocol-run` syntax shown by
  that binary, using the user's prepared inputs and an isolated output path.
  If preparation is incomplete, use the owner's preparation workflow first.
- **Verify:** report accepted steps and simulated time, energies and numerical
  checks, backend/device, trajectory and restart identities, and exact artifact
  paths. MD output does not establish a reaction rate or biological effect.

## 3. Compile a NumiTissue campaign

- **Discover:** resolve `numitissue`, read `numitissue help` and the current
  `campaign compile` or `organoid compile` help, and inspect the authored
  experiment and intervention inputs.
- **Preflight:** check schema, units, fidelity, seeds, output location, and
  existing checkpoint or compiled campaign.
- **Execute:** use the owner's compile command with its current arguments.
  Only launch an execution step when the requested study and runtime are
  available and the compiled contract identifies that step.
- **Verify:** report the generated campaign/checkpoint identities, validation
  status, and any actual simulated observables. Compilation alone is not an
  executed study or biological calibration.

## 4. Diagnose Numi Human standing

- **Discover:** resolve the installed `numi human` overlay or the Human owner,
  the matching native binary, bone and muscle-surface payloads, and existing
  standing artifacts. Read the discovered `stand` help and
  [human-standing.md](human-standing.md).
- **Preflight:** verify source/binary/payload identity, active GPU work, and an
  isolated output path. Use a short native prefix only to locate a failure.
- **Execute:** run the existing whole-body scene with the requested horizon;
  for the documented source scene the ten-second request is `--execute --steps
  10000 --timestep 0.001 --muscle-path-feedback 10 1` plus matching payloads.
- **Verify:** report accepted simulated time, assistance and controller mode,
  muscle/tendon and support loads, contact, falls/rejected steps, and the first
  limiting failure. A diagnostic prefix is not sustained standing.

## 5. Evaluate a robot policy on held-out rollouts

- **Discover:** `numi robots list`, inspect the selected robot, then read
  `numi evaluate --help` and the task/policy contracts. Find candidate and
  incumbent fingerprints and any retained held-out evaluation.
- **Preflight:** verify world, task, observation and action compatibility,
  held-out seeds, device, active GPU ownership, and artifact capacity.
- **Execute:** use the native evaluation command with exact candidate and
  incumbent inputs and isolated outputs. Keep the production selection as a
  separate evidence-backed step.
- **Verify:** compare paired physical outcomes, failed environment steps,
  replay identity, throughput, memory, and limits. Reward and build success do
  not prove policy superiority or hardware behavior.

## 6. Choose and inspect a solver profile

- **Discover:** `numi solvers list` and `numi solvers inspect FAMILY`; filter
  by the intended target. Show conceptual families first and exact variants
  only when the target or backend needs them.
- **Preflight:** inspect target compatibility, role, descriptor source and
  fingerprint. Do not treat a standalone or reference solver as a training
  backend without an executable target contract.
- **Execute:** create or resolve a fingerprinted profile through the current
  `numi solvers --help` grammar. If the user requests a run, invoke the owning
  runtime using that profile only after its compatibility checks pass.
- **Verify:** report the family, exact implementation, target, profile path,
  fingerprint, and stale status. Configuration is not physical or performance
  qualification.

## Virtual Wet Lab

Select a tissue population, define an exact RNA objective with Codex, edit shared
experiment cards, compare interventions, seal, explicitly authorize reveal and
inspect discrepancies. Follow [virtual-wet-lab.md](virtual-wet-lab.md); begin with
`numi wet-lab context`. No manual assay paths or JSON authoring.
