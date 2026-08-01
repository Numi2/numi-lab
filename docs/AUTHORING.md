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
  errors for links in any compiled articulation and for static, kinematic, or
  dynamic scene bodies;
- frame-to-frame position in reference-frame axes and tangent orientation
  error across articulated and scene-body domains;
- world linear velocity at a named frame origin and world angular velocity,
  materialized from the same generalized state consumed by physics;
- relative linear and angular velocity in the named reference frame, including
  reference translation and rotating-frame transport;
- frame position/orientation squared-error and exponential-tracking rewards;
- maximum frame position/orientation error termination;
- fixed-shape actor/critic histories, deterministic corruption, curriculum,
  randomization, and transactional reset.
- named SensorIR scalar values and validity bits, with actor/critic permission
  checks and an exact SensorIR fingerprint in the compiled TaskIR contract.

Reset frame observations are evaluated from the randomized reset
configuration through the generic articulated-kinematics operator and the
transactional scene-state layout before policy inference. They never reuse
body or scene poses retained by the preceding episode.

Frame-to-frame operators use an explicit named `reference`; they do not
overload static goal identities. The remaining TaskIR target is a
phase-separated graph covering action, command/event, observation, reward,
termination, recorder, reset, and curriculum phases. Site semantics, frame
acceleration, point/Jacobian quantities, sampled and trajectory goals, and
generic gates/reductions are not yet production operators.

All implemented names resolve at compilation. The GPU receives only typed
indices, counts, and fixed output layouts. Adding another body layout or static
pose goal does not add a robot-specific shader.

## Sensor authoring

The current WorldPack sensor declaration persists:

- modality and parent frame, or an asset-owned semantic target for a
  non-spatial sensor;
- semantic counterpart-body filters for contact-state sensors;
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

Body-attached sensor transforms are authored in the imported link/body frame.
The compiler converts translation once to the COM-centred runtime origin;
tactile descriptors instead use the cooked tactile surface transform as their
spatial authority.

The common SensorIR executor samples scalar-joint state, parent-frame pose,
world-space frame twist, six-axis IMU, six-axis contact-wrench, and
five-channel contact-state sensors on two explicit control boundaries.
An asset-owned joint-state declaration names one joint and publishes
`(position, velocity)` directly from the accepted generalized state. The
compiler currently accepts only revolute, prismatic, and continuous joints
with `nq=nv=1`; fixed and multi-DOF targets fail with a precise diagnostic
until their public tangent-coordinate observation contract is implemented.
The compiled descriptor carries only its joint and q/v indices, and a
joint-only execution plan does not materialize body poses or velocities.
Frame-twist linear velocity is evaluated at the authored sensor origin,
including the angular `omega x r` term. Force/torque consumes the committed
NumiSolver contact impulses from the final accepted physics microstep, divides
by that microstep duration, sums the resultant force and moment about the
authored sensor origin, and expresses both in sensor-local axes. It excludes
authored generalized rows; it does not infer force from acceleration or run a
second collision query.

A contact-state sensor publishes `(active, count, normal_force,
tangential_force, max_penetration)`. `count` is the number of accepted contact
blocks involving the parent body, `normal_force` is the sum of nonnegative
normal impulses divided by the final physics-microstep duration,
`tangential_force` is the magnitude of the resultant tangential impulse over
that duration, and `max_penetration` is the deepest negative signed
separation. Optional counterpart body names compile into a sorted immutable
index table. An empty filter accepts every counterpart. No name lookup or
robot-specific branch occurs while sampling.

An IMU publishes sensor-local specific force `(ax, ay, az)` followed by
sensor-local angular velocity `(wx, wy, wz)`. Specific force is the accepted
change in sensor-origin world velocity over the actual scheduled sample
interval minus world gravity. The point velocity includes the authored offset
from the parent centre of mass. Its previous sample timestamp and point
velocity are persistent SensorIR state, so non-divisor rates and rejected
reset transactions preserve the same training and deployment contract.

A reset-only pass seeds the randomized state before the first action; a post-
physics pass advances the schedule from the newly accepted state for the next
action and terminal value bootstrap. The executor owns per-environment
nanosecond phase accumulators, latency history, latest compact output,
timestamp/age/validity metadata, and reset state in persistent private
buffers. It supports the control rate or any slower rate, including non-
divisor schedules, and never publishes retained history as ordinary learner
tensors.

Task observations reference a sensor by authored ID. Compilation resolves that
ID to a descriptor and output offset, enforces actor or critic permission, and
fingerprints the exact SensorIR program. The runtime can bind either a scalar
channel or one of the `valid`, `fresh`, `reset`, `stale`, and `nonfinite` bits.
The `dropped` bit is also available and separate from `valid`: a coherently
withheld sample may be fresh, but it is never valid. Task-level observation
transformations remain a separate deterministic layer after SensorIR
publication.

Native scalar modalities apply authored corruption after selecting the
latency-delayed acquired sample. Bias is episode-static per channel, value
noise is Gaussian per acquired sample and channel, and dropout withholds the
complete sensor sample rather than independently deleting fields. Every draw
is counter-derived from the session seed, environment, episode, stable sensor
ID, acquired-sample sequence, channel, and purpose. No mutable RNG state or
host scheduling participates. Pose translation uses metre-valued scalar noise;
pose orientation uses a local tangent rotation in radians and is renormalized.
Contact-state active/count channels remain discrete, while corrupted force and
penetration channels remain nonnegative.

An asset-owned actuator-state sensor targets one canonical generalized-
velocity coordinate by name. Its eight scalar channels are, in order: raw
command, delay-selected command, backlash-effective command, unclamped motor
effort, envelope-limited motor effort, passive friction, applied generalized
effort, and active motor envelope. Command delay is rounded upward to a whole
control period. The delay ring and backlash play are persistent native state;
reset seeds a neutral command and a rejected physics transition restores the
pre-step state. Fixed and multi-coordinate transmission observations remain
outside this scalar contract.

Presentation sensors still execute in the native renderer and tactile sensors
still execute in the native tactile context. Folding those passes into the
session schedule, dedicated ray/LiDAR operators, recorder routing, and compiler
dead-code elimination remain incomplete. Native joint, actuator, pose, twist,
IMU, force/torque, contact-state, corruption episode identity, and histories
already journal reset environments and restore on a rejected physics
transaction.
Presentation- and tactile-domain corruption remain with their current native
owners until those passes join the common schedule.

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
