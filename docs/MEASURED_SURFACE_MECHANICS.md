# Measured-surface mechanics

`MeasuredSurfaceRobotPack` is the generic Numi Lab artifact for a robot whose
force-producing geometry is an indexed, frame-major deformable surface. The
artifact owns source identity and hashes, frame times, complete vertex and
triangle payloads, component ranges, phase-boundary semantics, aerodynamic
coefficients, body mass/inertia, and up to 32 bounded actuator contracts.

`compileMeasuredSurfaceRobot` rejects incomplete payloads, overlapping or
uncovered component ranges, non-finite data, non-monotonic frame times,
out-of-range indices, and periodic wrapping of a source declared nonperiodic.
The compiled fingerprint covers the complete payload and execution contract.

## MetalWorld execution

`MetalMeasuredSurfaceMechanics` materializes the compiled artifact lazily on
MetalWorld's device. Each environment receives persistent accepted surface
state and transient candidate state. One 256-thread threadgroup evaluates one
environment: it advances bounded second-order actuator state, interpolates the
measured frames, deforms each authored component, reduces triangle-level
aerodynamic force and torque, and adds that wrench to the selected floating
root before ABA.

Immutable geometry is uploaded once from shared staging storage into private
Metal buffers. Triangle loads reduce first within eight SIMD32 groups and then
across their eight leaders. Dynamic threadgroup storage is 896 bytes: 352 bytes
of reduction partials plus two 272-byte staged surface states. This avoids each
triangle lane repeatedly loading the full accepted/candidate actuator state.

The runtime uses `MetalWorldDeviceMechanicsProgram`, a generic borrowed-resource
callback boundary. `prepare` executes after TaskPack actuator lowering and
before ABA. `commit` executes only after universal physics acceptance. Invalid
surface state marks the ordinary per-environment MetalWorld status, causing q,
v, contact, task, and surface candidate state to roll back together. The
extension never owns or commits a command buffer and never waits for the GPU.
The runtime checkpoints accepted surface state and evidence at physics substep
zero. A failure at any later substep restores that control-step checkpoint,
matching MetalWorld's q/v/contact transaction rather than retaining a partial
phase or actuator advance. The same stateful mechanics instance is rejected if
submitted on overlapping command buffers; independent instances may still use
the ordinary asynchronous arena ring.

Accepted state records filtered actuator position/velocity, source phase and
direction, accumulated aerodynamic impulse, and accepted physics-substep count.
Accepted evidence retains the signed instantaneous world force and torque plus
their transactionally accumulated impulses and integration time. This permits
control-authority qualification without inferring direction from body motion;
late-substep rollback restores these accumulators with the rest of the control
step.
`inspectAccepted()` is an explicit post-submission diagnostic boundary that
blits these private records once; rollout and training never call it.

The flight SensorPack appends four non-temporal actor/critic lanes from the
accepted mechanics transaction: normalized measured-frame phase, phase
direction, scaled aerodynamic-force magnitude, and normalized actuator-state
norm. `MR_TASK_OBSERVE_DEVICE_MECHANICS` is generic extension telemetry rather
than a dove-only observation opcode. The task kernel observes zeros when no
device mechanics program is attached. The probe executes a one-step-shorter
prefix and compares its private accepted state against the next policy
observation, proving the pre-action observation cadence explicitly.

`RobotActuatorKind::measuredSurface` / `MR_TASK_ACTUATOR_MEASURED_SURFACE`
binds each normalized TaskPack action to a stable surface component. The native
joint actuator kernel deliberately skips this kind; the compiled mechanics
program is its sole execution owner. `CompiledRun` fingerprints and retains the
resolved surface binding, and the ordinary task-rollout C API constructs the
device mechanics program automatically.

The neutral rhythm command preserves the measured 3.50 Hz reflected wingbeat
and its exact surface positions. The bounded robot envelope can drive cadence
from 2.27 to 7.00 Hz and scales wing/tail displacement from the first measured
frame between 0.5x and 2.25x. These are authored actuator capabilities, not
additional measured claims; the source geometry and provenance remain
unchanged at neutral command.

## Current physical boundary

Aerodynamic force geometry uses every authored surface vertex and triangle.
Ground collision currently uses a first-frame body-only box proxy. Wings and
tail are therefore not yet deformable contact geometry. A reflected phase
boundary can sustain a nonperiodic measurement in a robot experiment, but it
is an explicitly authored counterfactual and is not evidence that the measured
sequence is periodic. Mass, inertia, density, and drag coefficients are robot
model assumptions until separately calibrated against hardware.

## Executable probe

Build `metalrobo_measured_surface_robot_probe` and pass a source manifest whose
directory contains `positions.f32le` and `triangles.u16le`. The probe compiles a
normal RobotPack + SensorPack + TaskPack + RunProfile, runs the surface through
the unified MetalWorld path, compares it with a gravity-only execution, and
requires bit-identical replay plus complete prepare/commit accounting.

`numi dove runtime-benchmark 64 32` runs the bounded scale profile. On the
2026-08-09 Apple GPU host, five repeated executions of the original `bd1c567`
kernel had a median 52.1508 GPU ms / 39,270.8 environment steps/s. The private
immutable buffers, staged surface state, SIMD32 reduction, and accepted
mechanics observation publication measured 49.1230 GPU ms / 41,691.3
environment steps/s: 1.0616x throughput. Dynamic threadgroup
storage fell from 11,808 to 896 bytes (92.4 percent). The benchmark also runs
deterministic replay, whole-control-step rollback at a deliberately induced
late substep failure, overlapping-runtime rejection, topology admission, and
accepted impulse/substep accounting. These are simulator measurements, not
hardware-flight or aerodynamic-calibration evidence.

`numi dove authority-sweep 2048 36` batches a neutral profile, both signs of
every action lane, explicit bilateral recovery modes, and deterministic
harmonic candidates. The diagnostic reads signed accepted force/torque
impulses after two reflected wingbeats and requires mean vertical force to
exceed 1.05 times modeled weight. Roll, pitch, and yaw must each provide both
torque signs at a modeled angular acceleration of at least 12 rad/s2. The
repeated sweep must be bit-identical and publish zero failed environment
steps. This qualifies the simulator's authored control envelope; it does not
calibrate aerodynamic coefficients or prove hardware flight.
