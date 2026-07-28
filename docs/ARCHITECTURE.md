# MetalRobo architecture

## Product boundary

MetalRobo owns the physics implementation, compiled model, runtime, GPU memory,
task execution, and public APIs. The initial release is macOS 26 / Metal 4
only. A Python package drives learning through MLX, while the engine itself is
C++23 plus Objective-C++ and Metal Shading Language.

The headless training path is the product core. Rendering is an optional
consumer of body-pose buffers and must not add work to headless steps.

## Immutable model and batched state

`metalrobo::Model` is compiled once into fixed-capacity GPU records:

- `MRModelGPU`: counts, timestep, gravity, ground, task and reward constants.
- `MRJointGPU`: topology, joint frame, limits, drive and armature.
- `MRLinkGPU`: mass, center of mass and full symmetric inertia.
- `MRColliderGPU`: link-local primitive shape and contact material.

Mutable runtime buffers are structure-of-arrays batches with environment as
the outer dimension. Model topology never changes during a rollout. Resetting
an environment changes only its row of state.

Version 0.1 reserves 32 DoF, 33 links, and 64 colliders per compiled model.
Joints are stored in parent-before-child order and can form a fixed-base tree;
Franka uses a 7-DoF chain inside that format. The capacities cover G1's 29
actuated joints, while its floating root will use dedicated pose/twist state
rather than consuming joint slots.

## One control step

The native runtime executes this order without per-environment host loops:

1. Decode normalized actions into joint targets and bounded actuator efforts.
2. Build joint transforms and spatial velocities.
3. Detect primitive contacts and assemble external spatial forces.
4. Run reduced-coordinate articulated-body forward dynamics.
5. Apply joint limits and integrate each physics substep.
6. Compute body poses, observations, reward and termination.
7. Reset terminated rows and sample their next targets on the device.

One Metal threadgroup owns one environment and uses fixed threadgroup scratch.
The first implementation dispatches 32 lanes but keeps recursion on lane zero
within that group; parallelism is across environments. The ABI allows later
SIMD-parallel spatial-matrix work without changing task code.

## API layers

- `metalrobo::Runtime` is the native C++ ownership boundary.
- `c_api.h` is the stable, exception-free ABI for Python, Swift, and other
  languages.
- The Python package exposes NumPy views over shared result buffers and an MLX
  PPO learner.
- Future MLX custom primitives will schedule physics and policy operations on
  one Metal stream. The C ABI remains useful for applications and debugging.

## Memory and synchronization

All runtime buffers use Apple-silicon shared storage. That avoids PCIe copies,
but it does not remove synchronization: the C API's `step` completes the
submitted command buffer before returning its shared views.

The Python wrapper exposes those buffers as stable, read-only NumPy views.
The current PPO path then materializes MLX arrays for policy work, so v0.1 is
not a fused physics/learner command stream. That boundary is deliberately
visible rather than described as zero-copy end-to-end training.

Allocation is preflighted against both
`MTLDevice.recommendedMaxWorkingSetSize` and `MTLDevice.maxBufferLength`.
Training must leave room in the unified pool for MLX policy parameters,
rollout storage, macOS, and an optional viewer.
