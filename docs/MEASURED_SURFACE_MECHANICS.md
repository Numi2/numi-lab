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

The runtime uses `MetalWorldDeviceMechanicsProgram`, a generic borrowed-resource
callback boundary. `prepare` executes after TaskPack actuator lowering and
before ABA. `commit` executes only after universal physics acceptance. Invalid
surface state marks the ordinary per-environment MetalWorld status, causing q,
v, contact, task, and surface candidate state to roll back together. The
extension never owns or commits a command buffer and never waits for the GPU.

`RobotActuatorKind::measuredSurface` / `MR_TASK_ACTUATOR_MEASURED_SURFACE`
binds each normalized TaskPack action to a stable surface component. The native
joint actuator kernel deliberately skips this kind; the compiled mechanics
program is its sole execution owner. `CompiledRun` fingerprints and retains the
resolved surface binding, and the ordinary task-rollout C API constructs the
device mechanics program automatically.

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
