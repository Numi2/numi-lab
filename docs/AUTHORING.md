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
- Named sites are unique link-local frames in the semantic model. They are
  persisted and fingerprinted, but resolve to a body plus composed local
  transform before execution; Metal never performs a site-name lookup.

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
groups, named body-local or site-relative frames, and typed SE(3) goals at
compilation. It supports:

- joint, root, command, terrain, parameter, contact-metric, and contact-wrench
  observations;
- world frame position/orientation and frame-to-goal position/orientation
  errors for links in any compiled articulation and for static, kinematic, or
  dynamic scene bodies;
- frame-to-frame position in reference-frame axes and tangent orientation
  error across articulated and scene-body domains;
- world linear velocity at a named frame origin and world angular velocity,
  materialized from the same generalized state consumed by physics;
- one world-axis component of the linear or angular frame Jacobian with
  respect to one named generalized-velocity coordinate. The compiler emits
  one immutable point query per distinct articulated frame, cohorts queries
  by articulation, and resolves the coordinate to a stable global velocity
  index;
- relative linear and angular velocity in the named reference frame, including
  reference translation and rotating-frame transport;
- frame position/orientation squared error, exponential tracking, and maximum
  error termination composed from semantic leaves and scalar SignalIR;
- topologically ordered scalar signals with semantic observation leaves,
  constants, arithmetic, min/max, absolute, square/root, safe division,
  clamp, exponential tracking/decay, `atan2`, comparisons, and bounds gates;
- heading-frame velocity, joint acceleration, previous-action delta,
  compiler-resolved soft-limit violation, mechanical power, and desired
  support-contact leaves;
- contiguous semantic-source reductions with identity, absolute, or square
  transforms and sum, mean, minimum, or maximum reductions;
- generic signal rewards and below/above/outside termination thresholds;
- a topology-derived count of named scalar commands with compiled initial
  ranges, hard limits, curriculum expansion, cohort-zero probability, and a
  shared resample duration;
- named generalized-velocity-delta events with compiled target coordinates,
  curriculum-interpolated ranges, stable counter-RNG identities, and a shared
  event schedule;
- named compact recorders that bind directly to SignalIR nodes and publish
  three generic metric values without robot-shaped transition fields;
- fixed goals, episode-sampled poses, and two-pose trajectories with clamped,
  looped, or ping-pong playback;
- a generic accepted-step phase oscillator and SignalIR-driven scalar
  curriculum success metric;
- fixed-shape actor/critic histories, deterministic corruption, randomization,
  and transactional reset;
- named SensorIR scalar values and validity bits, with actor/critic permission
  checks and an exact SensorIR fingerprint in the compiled TaskIR contract.

Reset frame observations are evaluated from the randomized reset
configuration through the generic articulated-kinematics operator and the
transactional scene-state layout before policy inference. They never reuse
body or scene poses retained by the preceding episode.

Frame-to-frame operators use an explicit named `reference`; they do not
overload static goal identities. A task frame authors exactly one body or site
source. A site-relative transform is composed with the model site's link-local
pose during compilation and then converted once to the body's COM-centred
runtime origin.

Signal operands may reference only earlier named nodes. This rejects forward
references and cycles during compilation and gives the GPU one deterministic
pass per environment. Signal leaves are truth-only: actor noise, mutable bias,
and vector normalization are not accepted. A SensorIR leaf requires truth
consumer permission and reads the current accepted sample after native sensor
advancement but before reward and termination evaluation.

There are no task-shaped reward or termination opcodes. Goal, frame, contact,
joint, and actuator identity belongs only on semantic source leaves; reward
records contain only a resolved signal channel, reporting channel, and weight.
Termination records similarly consume scalar signal indices and generic bounds.
This keeps fixed, sampled, and trajectory goals on one execution path and
prevents every new robot objective from adding a native branch.

Soft joint ranges are resolved from the selected mechanics preset during
compilation. The GPU consumes concrete lower and upper bounds; task completion
does not bind or reread the mechanics DoF table. Desired support contact is an
accepted-step phase signal keyed by the compiled contact group, so gait intent
and observed support remain ordinary graph inputs rather than a locomotion
shader mode.

Every reward also selects one of eight generic reporting channels: primary,
stability, velocity, acceleration, control, configuration, energy, or contact.
The channel changes only compact metrics; it does not alter reward evaluation.
There is no opcode-to-reporting switch.

Each scalar command has a unique authored identity. Actor, critic, and
SignalIR observations name that identity; the compiler resolves it to one
stable native slot and rejects unresolved or duplicate identities before any
session state is replaced. The immutable command record owns its initial
lower/upper range, hard lower/upper limits, and symmetric per-curriculum-level
expansion. The compiler also assigns a stable 64-bit semantic counter-RNG key,
so inserting or reordering unrelated commands does not perturb its stream.
Metal samples only the immutable table and compiled count.

Commands follow compact contact reductions in one topology-sized native
scalar-state arena. The compiler derives both resident and checkpoint
capacities, and every physics transaction journals the complete per-environment
stride. The task-state record therefore contains no fixed command vector. A
rejected reset restores even an episode-resampled command before the next
accepted transition. Vector-valued or correlated distributions and
per-command schedules remain incomplete; they must use typed operator tables
rather than reintroducing an anonymous fixed vector.

Each event also has a unique authored identity. The
`generalizedVelocityDelta` operator names one generalized-velocity coordinate;
the compiler resolves it to a stable native index and assigns an independent
64-bit semantic RNG key. Its lower and upper endpoints interpolate from the
initial range to the final range as curriculum advances. Metal applies the
sampled delta at the control boundary before physics. This is a task event,
not a modeled force or solver impulse. The bundled G1 x/y perturbations and a
fixed-base scalar-joint fixture use this same table, with no floating-root or
robot-identity branch. TaskPack 17 persists the exact event contract.

The current event cohort shares one minimum/maximum interval schedule.
Per-event schedules, scene-body wrench/state events, and richer typed event
operators remain incomplete. They must extend the compiled event table and
topology-sized transactional state rather than add another task mode.

Each compact recorder resolves an authored identity and one SignalIR node at
compilation. Recorder identities remain immutable host metadata; Metal carries
only resolved indices and values. The public C and Swift transition ABI exposes
three generic metric slots, while a session exposes their ordered identities.
The eight reward-reporting channels are separate generic aggregates. Neither
surface contains G1-named fields. Larger or scheduled recorder streams must use
the future unified recorder/SensorIR schedule rather than extending the compact
transition with another task-specific layout.

The current curriculum program consumes the episode mean of an authored
SignalIR node, plus an evaluation window, success threshold, and minimum
survival fraction. A multi-level program without a resolvable success signal
fails compilation transactionally. The accepted-step phase oscillator is also
ordinary TaskPack data; gait frequency is not a robot runtime mode.

A reduction cohort is compiled to one SignalIR node plus contiguous resolved
semantic sources. The GPU applies its transform and reduction directly; it
does not publish one intermediate signal per joint or contact group. Empty
cohorts fail compilation. SensorIR value and validity sources may participate
in the same cohorts when the sensor grants truth permission. The compiler
assigns one dense current-sample scratch slot to each SensorIR-backed semantic
source; direct mechanics sources allocate no slot. Metal materializes that
table after the accepted sensor sample and before SignalIR, so scalar leaves
and reductions observe the same control boundary without publishing a sensor
tensor or carrying a per-robot branch.

The remaining TaskIR target is a phase-separated graph covering action,
command/event, observation, reward, termination, reset, curriculum, and
arbitrary recorder-stream phases. Frame acceleration, arbitrary point queries,
full Jacobian tensors, multi-knot trajectory splines, independently scheduled
or richer event operators, vector and scheduled command operators, scheduled
recorder streams, and richer curriculum operators are not yet production
operators. The topology-sized named scalar command path, generalized-velocity
event path, three-slot compact recorder, and scalar SignalIR-driven curriculum
are production paths.

Frame-Jacobian observations are analytic operator outputs, not finite
differences of poses. The point is the compiled frame origin and the three
selectable rows are expressed in world axes. A coordinate belonging to a
different disconnected articulation has an exact zero entry. Scene-body
frames do not currently expose generalized-coordinate Jacobians and fail
compilation rather than returning an invented value.

An episode-sampled goal adds independent world-axis translation offsets and a
local tangent rotation-vector perturbation to its base pose. Its counter key is
the session seed, environment, episode, stable goal identity, channel, and goal
purpose. A trajectory linearly interpolates position and shortest-arc slerps
orientation from its start to end pose using accepted episode time. These goal
poses are derived values, not mutable simulator buffers: a rejected transition
cannot advance or partially publish one, and a reset selects a new episode
sample without adding another reset owner.

All implemented names resolve at compilation. The GPU receives only typed
indices, counts, and fixed output layouts. Adding another body layout or goal
does not add a robot-specific shader.

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

A spatial sensor may instead name one model site owned by its parent asset.
Its authored local pose is site-relative. The compiler validates site
ownership, composes the site and sensor transforms, resolves the actual rigid
or articulated body, and emits the ordinary body-frame SensorIR descriptor.
Tactile sensors retain their cooked tactile-surface authority and therefore do
not use this shortcut.

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
session schedule, dedicated ray/LiDAR operators, scheduled recorder-stream
routing, and compiler dead-code elimination remain incomplete. Native joint,
actuator, pose, twist, IMU, force/torque, contact-state, corruption episode
identity, and histories already journal reset environments and restore on a
rejected physics transaction.
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
