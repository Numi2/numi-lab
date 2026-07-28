# Robotics simulation landscape

Research snapshot: **2026-07-28**. This note uses primary project
documentation and repositories only. “Runs on Apple silicon” is kept separate
from “runs physics on the Apple GPU”: a native arm64 CPU build is not a Metal
backend.

The versions visible at this snapshot are MuJoCo 3.10.0, Isaac Sim 6.0.1 plus
the in-development Isaac Lab 3.0 documentation, Genesis World 1.2.1, and MLX
0.32.0. `latest`, `stable`, and `develop` documentation can move, so benchmark
records must also pin package versions and commits.

## Comparison

| Stack | Solver and model | GPU batching | Apple-silicon status | Rendering coupling | Licensing |
| --- | --- | --- | --- | --- | --- |
| **MuJoCo / MJX** | The native C engine uses generalized coordinates, recursive Newton-Euler bias forces, composite-rigid-body inertia, and a soft-contact constraint problem solved with Newton, CG, or PGS. MJCF/URDF/OpenUSD inputs are compiled into an `mjModel`; runtime storage is preallocated. ([computation](https://mujoco.readthedocs.io/en/stable/computation/), [modeling](https://mujoco.readthedocs.io/en/stable/modeling.html), [programming](https://mujoco.readthedocs.io/en/stable/programming/)) | Native MuJoCo is primarily a CPU engine; 3.10 added an internal thread pool for parts of collision and island solving. MJX-JAX adds batch dimensions to model/data and uses JAX `vmap`/JIT. MJX-Warp is the NVIDIA-optimized implementation. MJX implementations have an explicit feature-parity table and are not interchangeable with every native feature. ([3.10 changelog](https://mujoco.readthedocs.io/en/stable/changelog.html), [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)) | The native library ships macOS arm64 builds. MJX-JAX documents Apple Silicon support, but Apple’s JAX Metal plug-in is explicitly experimental, does not pass all JAX tests, and lacks `float64`; therefore “MJX on Mac” must be measured with the actual JAX backend reported, not assumed to be GPU execution. ([MuJoCo platforms](https://github.com/google-deepmind/mujoco#installation), [MJX devices](https://mujoco.readthedocs.io/en/stable/mjx.html), [JAX Metal limitations](https://developer.apple.com/metal/jax/)) | Core stepping is independent of rendering. MuJoCo ships separate classic OpenGL and Filament renderers; MJX still depends on native MuJoCo for model compilation and conventional visualization, while the Warp path has a separate batched renderer. ([programming](https://mujoco.readthedocs.io/en/stable/programming/), [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)) | MuJoCo source is Apache-2.0. Model and asset licenses remain per asset. ([license](https://github.com/google-deepmind/mujoco/blob/main/LICENSE)) |
| **NVIDIA Isaac Lab / Isaac Sim** | Isaac Lab 3.0 is moving to a backend factory spanning PhysX, Newton, and OvPhysX. The reference PhysX backend uses TGS by default and exposes PGS; assets and parameters flow through USD schemas. Backend coverage is not yet uniform. ([multi-backend architecture](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/multi_backend_architecture.html), [PhysX backend](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/physx/index.html), [Isaac Sim physics](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/index.html)) | Isaac Lab vectorizes cloned environments and exposes batched state/action tensors; the PhysX path has configurable GPU contact and heap capacities. Recent Lab documentation also describes kit-less Newton and standalone OvPhysX paths. ([vectorization](https://isaac-sim.github.io/IsaacLab/develop/source/setup/quickstart.html), [repository/backends](https://isaac-sim.github.io/IsaacLab/develop/source/overview/developer-guide/repo_structure.html), [PhysX configuration](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/physx/index.html)) | Isaac Sim 6.0 requirements list Ubuntu/Windows x86_64 with an NVIDIA RTX GPU, plus NVIDIA DGX Spark for aarch64—not macOS. A macOS WebRTC client only views a remotely running simulator. NVIDIA Warp supports Apple-silicon hosts with its **CPU backend only**, so the newer kit-less Lab paths do not supply Metal GPU physics. ([requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html), [streaming client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_faq.html), [Warp compatibility](https://nvidia.github.io/warp/stable/user_guide/compatibility.html)) | The historical/default PhysX path runs through Omniverse Kit; physics objects are created from USD and updated state is written back to USD by default. Headless stepping can skip the Kit update/render path. Lab 3.0 also documents pluggable renderers and kit-less backends, so “Isaac is inseparable from rendering” is no longer accurate, although the full sensor/RTX stack remains NVIDIA-specific. ([physics flow](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/index.html), [headless stepping](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.html), [renderers](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/renderers.html)) | Isaac Lab is BSD-3-Clause. Isaac Sim source is Apache-2.0, but Kit, models, textures, and other components have additional terms; some redistribution/service cases require NVIDIA AI Enterprise. ([Isaac Lab license](https://github.com/isaac-sim/IsaacLab#license), [Isaac Sim license FAQ](https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-faq.html)) |
| **Genesis World** | Its rigid solver is reduced-coordinate and follows MuJoCo’s soft constraint formulation, with Newton-Cholesky or conjugate-gradient solving. The same scene can also host MPM, FEM, PBD, SPH, and stable-fluid solvers. `build()` lays out fields and JIT-compiles kernels for the scene. ([rigid constraint model](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_collision/rigid_constraint_model.html), [solver inventory](https://genesis-world.readthedocs.io/en/latest/api_reference/engine/solvers/index.html), [scene build](https://genesis-world.readthedocs.io/en/latest/user_guide/configuration/concepts.html)) | `scene.build(n_envs=...)` creates a leading environment batch in solver state; state access, control, domain randomization, and partial reset operate on that batch. Official guidance keeps the step loop device-resident and avoids host synchronization. ([parallel state/reset](https://genesis-world.readthedocs.io/en/latest/user_guide/configuration/checkpoints.html), [efficient RL environments](https://genesis-world.readthedocs.io/en/latest/user_guide/policy_training/best_practices/efficient_environment.html)) | `gs.metal` is a documented Apple-silicon GPU backend, selected automatically after CUDA/AMD and before CPU. Metal is FP32-only in Genesis; requesting its FP64 mode is an error. This makes Genesis the closest direct same-machine competitor to MetalRobo. ([backends and precision](https://genesis-world.readthedocs.io/en/latest/user_guide/configuration/initialization.html), [repository](https://github.com/Genesis-Embodied-AI/genesis-world)) | The viewer is optional and physics runs headless. Camera rendering is separate from the viewer, with raster, path-traced, and batched renderer choices. Rendering still consumes scene-owned kinematic state, but it is not required in the training loop. ([viewer](https://genesis-world.readthedocs.io/en/latest/api_reference/visualization/viewer.html), [rendering](https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/index.html)) | Genesis source is Apache-2.0. Optional renderers, imported assets, and dependencies need their own license review. ([license and acknowledgements](https://github.com/Genesis-Embodied-AI/genesis-world#license-and-acknowledgments)) |
| **Apple MLX + Metal** | Neither is a physics engine. Metal is the low-level graphics/compute API; MLX supplies arrays, neural-network operations, autodiff, vectorization, and graph execution. Physics equations, collision detection, constraints, integration, and model semantics remain MetalRobo’s responsibility. ([Metal compute](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu), [MLX repository](https://github.com/ml-explore/mlx)) | MLX operations can batch over array dimensions and use `vmap`. It supports Python/C++ custom Metal kernels and full custom primitives; the extension API can encode into MLX’s active command buffer rather than creating a separate queue. ([custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)) | MLX is specifically for Apple silicon and uses its unified memory pool, allowing CPU and GPU operations to address the same arrays. Shared addressability does **not** remove ordering or synchronization costs. ([unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html), [Metal overview](https://developer.apple.com/metal/)) | MLX has no scene renderer or sensor simulator. Metal itself can run both compute and graphics pipelines, which lets MetalRobo keep rendering as an optional consumer of pose buffers without making MLX part of the viewer. ([Metal overview](https://developer.apple.com/metal/)) | MLX is MIT-licensed. Metal is an Apple platform API/SDK rather than an open-source simulator dependency. ([MLX license](https://github.com/ml-explore/mlx/blob/main/LICENSE)) |

## Roadmap consequences for MetalRobo

1. **Use the right baseline for each claim.** Genesis is the immediate
   Metal-GPU throughput and usability comparator. Native MuJoCo is the
   numerical/modeling reference. Isaac Lab is the workflow, task-composition,
   sensor, and large-stack reference plus a later cross-simulator validation
   target. MLX is the learner and command-stream integration substrate.

2. **Do not claim that Metal physics is unique.** The defensible product wedge
   is a small native C++/Metal runtime with predictable fixed-topology memory,
   fast startup, a stable C ABI, a headless core, and first-class MLX
   integration. Those properties must be measured; they are not implied by
   using Metal.

3. **For the Franka milestone, prioritize dynamics evidence over breadth.**
   Keep the fixed compiled model and batched state layout, but publish:
   environment count, control and physics timestep, substeps, contact/solver
   settings, warm-up policy, peak memory, environment-steps/s, and trajectory
   error against a pinned MuJoCo scene. Compare the same task and observation
   work on Genesis rather than quoting vendor headline throughput.

4. **Make contact quality the next heavy physics investment.** The current
   compliant ground response is enough to bring up Franka reach, but
   manipulation requires broadphase/narrowphase, frictional multi-contact,
   warm starting, equality constraints, free bodies, and diagnostics. MuJoCo
   and Genesis both expose coherent soft constraint systems; PhysX exposes an
   iterative TGS/PGS model. MetalRobo should specify its own constraint
   objective and tolerances before expanding shape breadth.

5. **Import formats; do not make them runtime ownership boundaries.** Add
   URDF/MJCF import into MetalRobo’s immutable model, preserving source units,
   joint frames, inertias, collision filters, actuator semantics, and asset
   provenance. Franka and G1 reference models exist in the
   [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), but
   each model’s license must be retained independently of MuJoCo’s engine
   license.

6. **Keep rendering downstream of physics.** Isaac’s USD/RTX stack is useful
   for high-fidelity sensor validation, while MuJoCo and Genesis both
   demonstrate optional rendering. MetalRobo’s viewer and future camera/LiDAR
   kernels should consume versioned pose/state buffers; headless stepping must
   allocate no render targets and perform no render submission.

7. **Close the learner boundary deliberately.** The current shared-buffer C
   ABI is appropriate for debugging and applications, but a synchronous
   `step()` still creates a CPU-visible boundary. A later MLX custom primitive
   should encode physics, observation/reward, and policy work into MLX’s active
   Metal command stream. Unified memory avoids a PCIe copy; it does not by
   itself provide zero-synchronization training.

8. **Advance to G1 only after the Franka evidence set is reproducible.** The G1
   milestone should reuse the same ABI and solver, then add floating-base
   state, self-collision filtering, stable foot friction, IMU/projected
   gravity, and actuator semantics. Validate small deterministic trajectories
   in MuJoCo before comparing batched policy throughput against Genesis and
   Isaac Lab.

## Caveats

- Backend labels do not imply numerical parity. Solver, cone, timestep,
  substeps, contact parameters, asset revision, and actuator model must be
  recorded for every comparison.
- Isaac Lab 3.0 material cited above is from its official `develop`
  documentation and may change before a tagged stable release.
- MJX-JAX’s documented Apple-silicon compatibility is broader than guaranteed
  Metal-GPU feature coverage; Apple still labels the JAX Metal plug-in
  experimental.
- Renderer quality or image throughput is not evidence of rigid-contact
  accuracy, and a physics benchmark with rendering enabled is not comparable
  to a headless one.
- Engine licenses do not automatically cover robot meshes, textures, CAD
  exports, model data, optional backends, or datasets.
