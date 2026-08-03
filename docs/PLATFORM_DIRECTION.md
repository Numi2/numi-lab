# Numi Lab platform direction

Numi Lab is a general physics and robotics laboratory built for Apple Silicon.
Individual tasks are workloads over the platform, not permanent architectural
centers. Ball dodge is one useful embodied-perception benchmark beside
standing, locomotion, manipulation, contact-rich interaction, recovery,
multi-robot work, and research surgical mechanisms.

## North star

A user should be able to bring a robot, a world, sensors, a task, and an
optional policy or demonstration; compile them once; and run deterministic,
batched, physically authoritative simulation, evaluation, and learning on the
Apple GPU. Adding a robot or task must not require a robot-specific shader,
host execution mode, or second simulator.

The shared path is:

```text
mechanics + WorldPack + TaskPack + sensors
                 +
InteractionPack or PolicyPack (optional)
                 |
                 v
compiled indices, topology, capacities, and fingerprints
                 |
                 v
persistent Metal physics + contact + sensing + inference
                 |
                 v
accepted state, exact replay, physical evidence, compact learning data
```

## Architectural invariants

1. **Physics owns truth.** Policies, demonstrations, foundation models, and
   cameras propose intent. Only accepted post-solver state establishes motion,
   contact, force, pressure, success, failure, or recovery.
2. **Robots are authored mechanics and artifacts.** Robot identity is resolved
   during compilation and disappears from the hot loop.
3. **Tasks are replaceable programs.** Observations, actions, rewards,
   outcomes, resets, and randomization compile from `TaskPack`; the core CLI
   does not become a catalog of hard-coded tasks.
4. **Sensors are composable.** RGB-D, tactile, plantar, proprioceptive, contact,
   and task-specific sensing share timestamp, reset, history, and publication
   semantics without host readback in production.
5. **Execution remains Apple-native.** Metal owns persistent simulation and
   device inference, Swift owns bounded asynchronous rollouts, and MLX owns
   batch learning. Unified memory is explicitly budgeted rather than copied
   opportunistically.
6. **Progress is evidence, not a binary gate.** Every run retains continuous
   physical metrics and its candidate artifacts. Deployment selection may
   protect an incumbent, but it must not erase useful trajectories or hide
   partial progress.

## Capability families

| Family | Shared platform responsibility | Example workloads |
| --- | --- | --- |
| Articulated robotics | ABA, actuation, limits, contacts, controllers | G1, Franka, PSM |
| Rigid scenes | bodies, collision, friction, CCD, constraints | clutter, projectiles, tools |
| Soft and slender matter | rods, deformables, stable coupling | cables, sutures, tissue research |
| Terrain and support | height fields, support contacts, pressure | walking, balance, recovery |
| Vision and identity | RGB-D, segmentation, motion, camera history | grasping, navigation, inspection |
| Touch and force | wrench, CoP, area, pressure, tactile history | manipulation, locomotion, tool use |
| Learning and teachers | MotionProposal, PolicyPack, InteractionPack, rollout evidence | BC, PPO, motion imagination, VLA teachers |
| World variation | compiled distributions and deterministic reset | robustness, curricula, system ID |

The examples are deliberately non-exclusive. A new capability should improve
the shared family and become available to several workloads whenever its
semantics permit.

## Highest-leverage next work

1. **General sensor composition.** The first `numi.visual-observation.v1`
   artifact now binds arbitrary VisualPacks and an articulated camera to the
   compiled task sensor program. Continue the same contract for tactile fields,
   additional camera kinds, and histories; the old G1/ball flags are only a
   compatibility preset.
2. **Generic action-chunk compilation.** The first
   `numi.foundation-adapter.v1` artifact now owns state layout, named model
   outputs, robot joints, controller limits, and contact intent. Author and
   qualify Franka, PSM, and additional provider adapters without adding robot
   branches to the simulator or compiler path.
3. **Motion-provider realization.** The first `numi.motion-proposal.v1`
   provider now runs ARDY Core ONNX on Apple Silicon and preserves its human
   skeleton trajectory without pretending it is a robot action policy. Add
   robot-authored retargeters and InteractionPack compilation so imagined
   motion becomes physically testable intent through the same native path.
4. **Broader physical qualification.** Maintain focused deterministic replay,
   transaction, contact, joint-limit, sensor, and coupling probes for each
   physics family, plus end-to-end robot workloads that expose real behavior.
5. **Manipulation and multi-robot depth.** Turn the existing Franka and PSM
   mechanics and scenes into fully physical tasks with tools, movable objects,
   coordinated contacts, and task-owned sensing.
6. **Coupled-material production.** Continue integrating rigid, articulated,
   rod, and deformable solvers through shared contacts and explicit energy,
   stability, capacity, and memory evidence.
7. **Apple performance as a feature.** Profile bytes, synchronization,
   occupancy, retained memory, transient arenas, MLX overlap, and energy. Fuse
   work only when exact replay and physical semantics remain unchanged.
8. **One user flow.** Evolve `numi` as a small discoverable doorway into
   transparent capabilities, not a second planner or fixed robotics workflow.

## How workloads should be reported

Reports lead with the shared capability exercised, then the particular task.
For example, a ball-dodge run qualifies dynamic scene bodies, visual history,
whole-body contact, anticipation, and recovery. It does not redefine Numi Lab
as a dodge simulator. Likewise, a surgical workcell image does not establish
deformable contact or clinical competence without the corresponding physical
execution evidence.

The platform advances when a change adds physics breadth, correctness, scale,
sensorimotor intelligence, robot/task generality, or a shorter path from
authored intent to measured physical behavior.
