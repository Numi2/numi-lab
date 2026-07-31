# Authoring and compilation

MetalRobo accepts semantic source assets and packs. Only the compiler creates
packed indices, offsets, capacities, execution cohorts, and GPU layouts.

## Inputs

`WorldSource` accepts one of:

- URDF plus optional SRDF and referenced assets;
- MJCF and recursively included assets once the importer is qualified;
- the current persisted WorldPack format.

A simulation also accepts a TaskPack and optional PolicyPack. Sensors are
authored with the world and selected or transformed by TaskPack observation
operators.

## Compiler pipeline

```text
parse with source locations
        -> expand includes and defaults
        -> validate units and semantics
        -> resolve names and provenance
        -> build ModelIR / ActuatorIR / SensorIR
        -> compile constraints and rod domains
        -> compile TaskIR / PolicyIR
        -> derive capacities and execution cohorts
        -> fingerprint artifacts and contracts
        -> publish immutable CompiledSimulation
```

Compilation is transactional. A failed compile returns diagnostics without a
partially usable program.

## Model rules

- SI units are canonical: metres, kilograms, seconds, radians, newtons, and
  newton-metres.
- Body translation and linear velocity refer to centre of mass.
- Floating orientation uses normalized quaternion `xyzw`; angular increments
  and derivatives use tangent coordinates.
- Inertias must be finite, positive, and physically valid after frame
  transformation.
- Collision and visual geometry are independent authored resources.
- Dynamic triangle soups are rejected unless converted to a declared convex
  decomposition.
- Near-zero scalar joint ranges compile as equality constraints; inverted
  ranges fail compilation.
- Every actuator/controller preset has a stable fingerprint. A policy cannot
  mix fields from different presets.

Bundled robots are generated from pinned upstream assets. Their manifest
records source revision, hashes, licenses, conversions, geometry substitutions,
and actuator/controller presets. Application code must not clone or manually
rebase bundled mechanics tables.

## MJCF compatibility target

The importer must preserve order-sensitive MJCF behavior and cascading
defaults/classes before constructing ModelIR. The first qualified surface is:

- recursive includes with cycle detection and asset provenance;
- bodies, frames, sites, inertials, and common geoms;
- hinge, slide, ball, and free joints;
- motors, position/velocity actuators, and general actuator gearing;
- fixed and spatial tendons;
- equality constraints, contact rules, options, sensors, and keyframes.

Flex, plugins, SDFs, fluids, cloth, general deformables, and undecomposed
dynamic concave geometry fail with source-located diagnostics. A pinned
companion MJCF used by a bundled robot is not evidence of a general importer.

## TaskPack

TaskPack is being generalized in place; there is no second task format. The
currently implemented native surface resolves actions, joints, bodies, contact
groups, named body-local frames, and static SE(3) goals at compilation. It
supports:

- joint, root, command, terrain, parameter, contact-metric, and contact-wrench
  observations;
- world frame position/orientation and frame-to-goal position/orientation
  errors for bodies in the selected articulation;
- frame position/orientation squared-error and exponential-tracking rewards;
- maximum frame position/orientation error termination;
- fixed-shape actor/critic histories, deterministic corruption, curriculum,
  randomization, and transactional reset.

Reset frame observations are evaluated from the randomized reset
configuration through the generic articulated-kinematics operator before
policy inference. They never reuse body poses retained by the preceding
episode.

The remaining TaskIR target is a phase-separated graph covering action,
command/event, observation, reward, termination, recorder, reset, and
curriculum phases. Site semantics, scene-object frames, frame twist and
acceleration, point/Jacobian quantities, sampled and trajectory goals, generic
gates/reductions, and sensor references are not yet production operators.

All implemented names resolve at compilation. The GPU receives only typed
indices, counts, and fixed output layouts. Adding another body layout or static
pose goal does not add a robot-specific shader.

## Sensor authoring

The current WorldPack sensor declaration persists:

- modality and parent frame;
- image/tactile dimensions and calibration;
- sample phase/rate, exposure, latency, and history length;
- modality-independent noise, bias, and dropout;
- actor, critic, truth, and recorder permissions.

`SimulationCompiler` resolves asset-relative parents to a stable world/body
index and compiles every declaration into one immutable descriptor table. Its
output/history offsets, nanosecond schedule period, execution domain, tactile
atlas binding, and fingerprint are topology-derived. Sensor compilation is
transactional: duplicate names, unresolved parents, uncovered latency, and
tactile metadata disagreement leave the previous program unchanged.

Today this is the common SensorIR contract, not yet a common executor.
Presentation sensors still execute in the native renderer and tactile sensors
still execute in the native tactile context. Persistent session-owned phase
accumulators, unified reset/history state, TaskIR sensor references, contact,
ray/LiDAR/IMU operators, recorder routing, deterministic counter corruption,
and compiler dead-code elimination are the next runtime tranche.

The runtime target uses deterministic fixed-point phase accumulators and
counter randomness keyed by environment, episode, sensor, sample, and channel.
Unobserved and unrecorded sensors will be removed by the execution-plan
compiler once TaskIR/recorder bindings own their liveness.

RGB, depth, identities, normals, and motion consume only authored Visual
Presentation V3 packs. Tactile deformation consumes authored undeformed and
sensing surfaces; solver impulses remain the force and wrench authority.

## PolicyPack

PolicyPack binds a typed PolicyIR to exact observation, action, actuator,
history, normalization, recurrent-state, dtype, and reset contracts. Mutable
weights carry a monotonic revision and install only into an inactive native
weight bank. The runtime swaps revisions between rollout chunks.

A mismatch in dimensions, fingerprints, history, reset behavior, or actuator
preset is a compile/install error, never an implicit adapter.

## Fingerprints

Fingerprints appear at artifact, cache, replay, policy-contract, and external
comparison boundaries. They are not computed per frame and are not duplicated
across adjacent metadata without an independent trust boundary.
