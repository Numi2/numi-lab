# Numi suite tool routing

This is a navigation map, not an installed-tool inventory or qualification
certificate. Recheck the selected owner's current source and help. Repository
names below identify owners even when checkouts have different directory names.
Read only the section needed for the user's task.

## Contents

- Discovery and execution
- NumiVivo: molecular science and programmable biology
- NumiTissue: developing neural tissue
- NumiBrain and NumanX: embodied intelligence
- Numi Human: anatomy and biomechanics
- Numi Lab: physics, robotics, providers, and neural cultures
- BirdFlow, Numi Automata, and Dr.Anmar
- Cross-suite execution and completion

## Discovery and execution

1. Start with `numi context --paths` and, when readiness matters, `numi doctor`.
   These cover the selected Lab installation; a successful doctor does not
   establish that NumiVivo, NumiTissue, or other sibling packages are installed.
2. Resolve a sibling from explicit user paths, the active workspace, existing
   configuration, integration lockfiles, or `command -v`. Verify its repository
   remote, revision, dirty paths, package manifest, and owning documentation.
   Do not guess a checkout from a similarly named archive or old worktree.
3. Read the selected executable's help. Prefer an existing compatible build.
   `swift run` may build and resolve dependencies; it is not a read-only help
   operation when the package is unbuilt. Inspect `Package.swift` and source
   help first in that case, then build only the necessary target when needed.
4. A discovered `numi` overlay takes precedence according to the dispatcher.
   If a sibling has no overlay, invoke its actual executable from its owner.
   Do not invent `numi vivo`, `numi tissue`, or `numi brain`. For a requested
   persistent alias, use the existing `.numi/commands` extension mechanism and
   preserve executable identity, arguments, exit status, and provenance.
5. If the owner is unavailable locally, inspect its configured remote or the
   user's selected execution host. Distinguish source discovery, missing build,
   unsupported platform, and unavailable execution access. Installing this
   plugin alone does not install or qualify any sibling runtime.

## NumiVivo: molecular science and programmable biology

Owner: `Numi2/numiVivo`. Product: `numivivo` from
`Sources/NumiVivoCLI`; native libraries and shaders remain in this package.
Start with `README.md`, `Documentation/CAPABILITIES.md`,
`Documentation/README.md`, and the relevant example. Inspect `numivivo --help`
and the family's help command for exact inputs/options.

| Intent | Command/help family |
| --- | --- |
| Molecular structures and selections | `structure-help`: import, export, inspect, select |
| Alternate locations, topology and slicing | `structure-prep-help` |
| Force-field records and parameter assignment | `forcefield-help`: validate, assign, compile |
| Prepared AMBER systems | `amber-help`, `amber-md-help`: `amber-import`, `amber-import-md` |
| Explicit preparation and adaptive molecular sampling | `molecule-help`: `molecule-prepare-template`, `molecule-prepare`, `molecule-sampling-template`, `molecule-sampling-run`, `molecule-sampling-analyze`, `molecule-rate` |
| Metal molecular dynamics and minimization | `md-capabilities`, `md-help`: `md-run`, `md-minimize` |
| Staged MD, restart and coordinate archives | `md-protocol-help`: template, validate, run, resume, inspect; `md-trajectory-inspect` |
| Native electronic structure and embedded chemistry | `chemistry-help`: `chemistry-template`, `chemistry-run`, `chemistry-solve` |
| Shared-orbital ECC reaction paths | `chemistry-path-help`: template and run |
| Reaction qualification | `reaction-help`: `reaction-template`, `reaction-run` |
| QM/MM free energy, transmission and chemical-state kinetics | `qmmm-free-energy-help`: analyze, qualify, rates, populations and exchange-network commands |
| ProgramPack compilation and inspection | `validate`, `compile`, `inspect`, `analyze`, `synthesize-mechanisms` |
| Hybrid reaction execution | `hybrid-help`: `plan-hybrid`, `run-hybrid` |
| Target engagement and conditional rate evidence | `engagement-help`: validate, source, compile, run, batch, study, rate, apply-rate |
| Posterior kinetic inference | `posterior-help`: `engagement-fit`, `engagement-predict`, `engagement-sensitivity` |
| Metal-screened inference | `screening-help`: `engagement-screen-check`, `engagement-fit-screened` |
| Finite-drug and experiment-design research | `research-help`: `finite-drug-run`, `engagement-design`, occupancy benchmark inspection/comparison |
| Campaign, partition, population, surrogate and physiology packs | `compile-campaign`, `verify-campaign`, `compile-partition`, `compile-population`, `compile-surrogate`, `compile-physiology`, `compile-physiology-coupling` |
| Capacity and checkpoints | `plan-capacity`, `inspect-checkpoint` |
| Reproducible task/artifact workflows | `workflow-help`: catalog, template, plan, run, import, export, verify |

The prefixes above are routing aids, not wildcard executable commands. Read
`Sources/NumiVivoCLI/main.swift`, the selected `Vivo*CLICommands.swift`, and
`VivoCLICommandRouter.swift` when help and documentation differ. Do not expose
library-only quantum, embedding, population, or coupling APIs as imaginary CLI
commands. Inspect the owning module and its examples for those requests.

For an existing binary, a small supplied fixture can begin with:

```sh
numivivo engagement-help
numivivo engagement-validate Examples/target-engagement/synthetic-pulse.json
```

Run from the resolved owner so fixture paths resolve. Choose execution backend
and output directory from the actual task, then follow its example. A fixture
validation is not a simulation or measured pharmacology.

Preserve unit-explicit topology/parameters, charge and protonation assumptions,
atom maps, periodic geometry, numerical profile, seeds, artifact hashes, and
restart identity. Check backend support before running the prepared system.
Use bounded trajectory storage and resume receipts instead of restarting an
expensive protocol. Native FP64 CPU chemistry is a legitimate execution path;
Metal MD and kinetic cohorts have separate numerical qualification.

An electronic-energy difference is not automatically an activation Gibbs free
energy. Conversion to a rate requires the owner's context-bound kinetic
contract and uncertainty/convergence evidence. Occupancy does not establish
cellular response or organism efficacy. Use `Documentation/QMMMFreeEnergy.md`,
`QMMMDynamicalTransmission.md`, `QMMMChemicalStateKinetics.md`,
`PreparedMolecularWorkflows.md`, and `Design/ARTIFACTS_AND_PROVENANCE.md` for
those specific paths. Read current restrictions instead of freezing an old
capability map into a new scientific claim.

## NumiTissue: developing neural tissue

Owner: `Numi2/numitissue`. Products: `numitissue`, `numitissue-examples`.
Start with `README.md`, `Examples/README.md`, `Package.swift`, and
`Sources/NumiTissueCLI`. Use `numitissue help` for current command grammar.

Route cellular development, morphology, compartment electrophysiology,
synapses/plasticity, glia, extracellular fields, molecular microdomains,
adaptive fidelity, tissue observations, and biological-interface modeling here.
The main workflow entry points include:

- `validate-experiment`, `campaign compile`, and `screening compile` for
  versioned experiments, shards, and intervention campaigns.
- `organoid compile` for fitting studies; `wetware plan` and `wetware validate`
  for modeled stimulation protocols and declared safety envelopes.
- `checkpoint inspect` and `nmodl compile` for stored state and the supported
  mechanism-import subset; inspect the source for other import formats.
- `phase3 status`, `support`, `contract`, and `verify` for explicit execution
  contracts and evidence manifests.
- `numitissue-examples all OUTPUT_DIRECTORY` for supplied deterministic workflow
  generators; inspect their README before executing the generated campaigns.

Read `Docs/Verification/PHASE3.md` and `PHASE3_STATUS.md` for the selected
qualification path. Preserve fidelity projection, conservation, time cadences,
intervention overlays, random streams, and all-or-nothing publication.
A generated campaign is not an executed study; a CPU reference is not a Metal
qualification. Simulated MEA/protocol output is not evidence of physical
culture stimulation. Distinguish NumiTissue from both NumiBrain regional
computation and the Lab's smaller `numi neurons` LIF/STDP runtime.

## NumiBrain and NumanX: embodied intelligence

NumiBrain owner: `Numi2/numi-brain`. Inspect `Package.swift`, `README.md`,
`STATUS.md`, and `docs/NUMANX_STATE_OF_THE_ART_ROADMAP.md`.
The package exposes `numi-brain-scheduler`, `numi-brain-dispatch`,
`numi-brain-tissue`, `numi-brain-policy`, `numi-brain-numanx-interop`,
`numi-brain-gate-c`, and `numi-brain-gate-d`. Resolve the product needed and
read its actual help/source before execution. They cover different scheduler,
Metal, policy, integration, learning, and validation boundaries; one is not a
substitute for another.

NumanX owner: `Numi2/Numan-x`. Read `integration.lock.json`,
`docs/ARCHITECTURE.md`, and `docs/QUALIFICATION.md`, plus the Lab runtime's
`docs/NUMANX.md`. NumanX is the integration home; executable implementations
remain in Brain and Lab. Do not invent a standalone `numanx` CLI or copy owner
implementations into the integration repository.

Preserve the exact motor/sensor resource leases, physical timestamps, root
proposal/preflight/ACK/apply protocol, private consequences, joint publication,
and rollback/quarantine behavior. Rejected physical futures cannot publish
neural learning, memories, observations, or random-state advances. Gate C
learning evidence and policy promotion are distinct from root correctness;
Gate D physical/biological evidence and measured performance remain distinct.
Update integration locks only from the actual published and qualified owners.

## Numi Human: anatomy and biomechanics

Authoring owner: `Numi2/numilab-human`, product `numilab-human` from
`src/numilab_human/cli.py`. Native mechanics and runtime coupling are also owned
by the Lab. Discover a `numi human` overlay before using that spelling; it may
exist only in a configured Human workspace.

Read `Docs/ARCHITECTURE.md`, `Docs/IMPORT.md`, `Docs/HUMAN_STAND_V1.md`,
`Docs/HUMAN_TENDON_STEP_TRANSACTION.md`, and the relevant regional mechanics
record. Use the command help for supported import, preparation, inspection,
standing, and visual workflows. Trace native execution into the Lab owner.

Keep MyoSim/OpenSim mechanics separate from BodyParts3D or other visual anatomy.
Preserve source licenses, source endpoint provenance, authored muscle routes,
terminal load records, and accepted-step publication. A single force authority
must own each contribution: do not scatter `J^T` loads twice or use rendered
geometry as proof of mechanics. Load transfer, deformable response, sustained
standing, and anatomical validation need their own measured evidence.

## Numi Lab: physics, robotics, providers, and neural cultures

Owner: `Numi2/numi-lab` (often checked out as `MetalRobo`). Discover the current
commands, including user overlays, from `numi context --paths`.

| Capability | Route |
| --- | --- |
| `robots` | Catalogs, authored mechanics and semantic role inspection |
| `train`, `evaluate` | Exact policy contracts, native rollouts, learning and held-out selection |
| `solvers` | Family discovery, variants, target compatibility and fingerprinted profiles |
| `matter` | Coupled material/deformable workflows and native probes |
| `drone` | Authored drone and flight workflows; read `docs/PX4_X500.md` where applicable |
| `sapiens` | Official ROBOTIS AI Sapiens K1 source import and policy qualification |
| `foundation` | Foundation action providers and adapters; `docs/FOUNDATION_POLICIES.md` |
| `motion` | Motion imagination and physical realization; `docs/MOTION_PROVIDERS.md` |
| `hyper-policy` (when discovered) | ARDY move compilation and physical qualification; see the preserved workflow below |
| `residual-teacher` | Discover the current residual-teacher command and its evidence contract |
| `neurons` | Synthetic neural cultures and virtual MEA protocols |
| `codex` | `install` and `status` for the local plugin and cached-source agreement |

Core `doctor`, `context`, `run`, `version`, and `help` are dispatcher operations.
Workspace tools such as `window`, `coupled-profile`, or `franka-explore` are
optional overlays; read their resolved source/help, not an assumed bundled API.
All capability families remain extensible beyond this list.

For `numi neurons`, inspect `docs/NEURON_CULTURE.md` and `numi neurons --help`.
The capability includes compile, simulate, grow, protocol, qualify, embody,
benchmark, view, inspect, replay, and render. Preserve network seed, mapping,
ablation, checkpoint, accepted neural window, and embodied-root identities.
A quick functional run cannot substitute for the full promotion matrix.
The viewer represents synthetic networks, not living tissue or physical MEA
measurements. NumanX embodiment requires its authoritative assets.

For engine architecture, load the main skill's owner documentation map.
The standalone `Numi2/numi-solver` repository owns the extracted Temporal Cone
Metal solver and pointer-free GPU ABI. Read its `README.md` and owning source
for solver-specific development and probes; it does not contain the Lab's robot
models, trainers, tasks, or application runtime. Treat NumiSolver and Matter as
exact target-specific implementations. A standalone solver or profile does not
establish training integration, backend parity, or physical qualification.

### ARDY HyperPolicy moves


When `numi context --paths` discovers `hyper-policy`, use it when the user wants one concrete ARDY-imagined G1 move
realized as a device-resident policy. ARDY supplies motion intent; it never
supplies physics or deployable actions directly. The production path is:

1. Run `numi doctor`, `numi context`, `numi robots inspect unitree_g1`, and
   `numi hyper-policy --help` before selecting inputs.
2. Prefer an existing authenticated proposal directory when reproducing a
   known move. Use `--prompt` only when the installed ARDY model and text
   encoder are available. When the owner supplies prompt text, pass it
   verbatim rather than silently expanding its choreography. Native
   `g1skel34` proposals use exact G1 mechanism projection; `cskel27` proposals
   use the verified Core-to-G1 retarget path. Pin the G1 URDF, compiler
   checkpoint, evaluator, main metallib, native library, seeds, task, scene,
   and contact-group mapping.
3. If no HyperPolicy compiler checkpoint exists, use
   `numi hyper-policy canonicalize`, then `initialize-checkpoint` with a
   fingerprint-compatible PolicyPack and explicit adapter ranks. A checkpoint
   with zero training updates is integration-only; never describe its output
   as a learned move or pass `--allow-untrained` without saying so.
4. Use `numi hyper-policy create` for the production transaction. It
   canonicalizes the motion, generates deterministic low-rank candidates,
   writes authenticated HyperPolicyPacks, executes each through NumiSolver,
   optionally repairs coefficients from an independently produced exact
   solver-teacher rollout, and publishes a deployment only after its
   configured physical gates pass. Never use a candidate's own executed
   actions as its repair labels.
5. For direct replay or diagnosis, run `metalrobo_task_rollout` with both the
   matching `--interaction-pack`/`--interaction-clip` and
   `--hyper-policy-pack`. The InteractionPack remains the physical reference;
   HyperPolicy actions are residuals and must not apply the reference twice.
   Always publish a `--rollout-pack` so the exact `hyper_policy_phase`, teacher
   actions, policy revision, transition failures, and outcome schema can be
   inspected.

The live GPU dependency is accepted q/v plus the actor observation row and
solver-resolved compact contact metrics. Phase update, low-rank adapters, and
residual actions must remain in one command buffer before TaskProgram action
application. Reset masks reset phase canonically; host warmups must not advance
hidden phase state.

Report the move prompt/identity, source and checkpoint fingerprints, native
pack and rollout hashes, exact command, device, environment count, control
steps, phase progression, failed steps, terminations, tracking, root height,
tilt, contact outcomes, retained/peak memory, and artifact directory. A build
or synthetic pack proves integration only. A short simulator rollout proves
that execution path only. Neither is trained-motion quality, soak evidence, or
real-hardware proof.

## BirdFlow, Numi Automata, and Dr.Anmar

**BirdFlowMetal** — owner `Numi2/numi-gpu-physics`, commonly checked
out as `BirdFlowMetal`. Read `README.md`, `Package.swift`, `RESEARCH.md`, and
`Docs/NUMI_CROW_JOURNEY.md` where relevant. Route fluid/body coupling, measured
bird inputs, crow flight, scientific rendering, and accepted-state replay here.
Resolve the named package product or Lab integration from source. Preserve
measured/estimated geometry and motion provenance. A surface trail is not a
CFD streamline; an authored animation is not a physical flight rollout.

**Numi Automata** — owner `Numi2/numi-life-automata`, often the
`emergentnumilife` workspace. Read `README.md`, `Package.swift`, the Xcode
project, and the selected script under `Scripts/`. Existing workflow scripts
include GPU, causal, lifecycle, regeneration, crossbreeding, and replicate
experiments. Inspect script arguments before execution; app packaging is a
separate task. Preserve one evolving world, finite material/energy ledgers,
genome identity and interventions. Its dimensionless ecology and classical
quantum-walk field do not establish calibrated abiogenesis or quantum hardware.

**Dr.Anmar** — use the installed `dr-anmar` companion skill for its studio,
surgical workcells, and NVIDIA/Isaac execution. If unavailable, discover its
owning repository and report the access boundary. Preserve that runtime's
hardware, host, artifact, and physical-evidence contracts. A surgical image or
simulated result does not establish clinical validity.

## Cross-suite execution and completion

Turn the question into an explicit chain of observables and owners, for example:
molecular parameters → reaction evidence → tissue state → accepted neural
observation → embodied action → physical consequence. For each edge, verify
that the actual adapter exists and supports the requested quantity, units,
space/time mapping, resource lifetime, revision, and backend.

Use the existing transaction participant and artifact contracts. Keep proposed
and accepted states distinct across all participants. Do not synthesize a
missing participant and label the combined result as end-to-end execution.
Build missing integrations in their owners when requested, then qualify them
with a bounded causal case and rejection/replay checks.

Report which tools actually ran, their exact commands, input/model identities,
owner revisions, backend/device, output artifacts, failures, and relevant
observables. Separate source/build success, artifact integrity, numerical
agreement, causal coupling, biological calibration, learning advantage, and
performance. State what remains unavailable without replacing measured partial
progress with a blanket success or failure verdict.
