<!-- GENERATED FILE: python/generate_capability_matrix.py -->
# MetalRobo capability matrix

This file is generated from `schemas/capability_matrix.json`.
A status is a product claim boundary, not a roadmap estimate.

Current registry: qualified: 8, implemented: 2, experimental: 3, unsupported: 5.

| Capability | Status | Owning check | Evidence artifact | Exact scope |
|---|---|---|---|---|
| `compiler.urdf_srdf` | **qualified** | `compiler.robot_description` | `metalrobo_robot_description_cooker_probe` | Rigid articulated URDF/SRDF import, semantic names, meshes, and deterministic fingerprints. |
| `compiler.mjcf` | **unsupported** | `none` | `none` | No production MJCF parser. A pinned G1 companion-MJCF preset is data, not general MJCF support. |
| `physics.numi_solver` | **qualified** | `physics.metal_contact` | `metalrobo_metal_world_contact_probe` | Production Apple-Metal rigid/articulated solver: fixed temporal microsteps, one coupled nonlinear block sweep per microstep, immediate integration, and transactional warm state. Rod endpoints are rejected until the retained banded rod response participates in the sweep. |
| `physics.quality_newton` | **experimental** | `none` | `metalrobo_metal_unified_quality_probe` | Available for focused study; not yet the qualified production solver. |
| `physics.true_temporal_tgs` | **unsupported** | `none` | `none` | NumiSolver uses temporal small steps but does not claim the complete vendor-specific PhysX TGS contract. |
| `physics.constraint_ir` | **implemented** | `physics.constraint_ir` | `metalrobo_constraint_ir_probe` | Shared row, endpoint, cone, stable-key, and warm-start representation. Active scalar joint limits and contacts are coupled in NumiSolver; generic runtime widths beyond three rows remain incomplete. |
| `physics.literal_convex_ccd` | **experimental** | `physics.metal_contact` | `metalrobo_metal_world_contact_probe` | Selected convex event-time paths exist; complete advance-to-impact, solve, and continue coverage is not qualified. |
| `mechanics.g1_source_exact_presets` | **qualified** | `mechanics.actuation` | `metalrobo_g1_model_probe` | Distinct fingerprinted URDF, companion-MJCF, and RL Lab actuator contracts; no silent 25/35/50 N m mixing. |
| `runtime.swift_metal_session` | **qualified** | `integration.native_simulation` | `metalrobo_simulation` | Swift schedules bounded native submissions; Metal owns persistent simulator state and transactions. |
| `learning.swift_mlx_ppo` | **implemented** | `integration.native_simulation` | `metalrobo_train` | In-process Swift/MLX learner. Python PPO and learner-worker runtimes are removed. |
| `tasks.compiled_native_runtime` | **qualified** | `task.program` | `metalrobo_task_program_check` | Fixed- and floating-base tasks share compiled semantic bindings, native joint/contact observations, reward, termination, reset, curriculum, and randomization. |
| `tasks.generic_typed_operators` | **experimental** | `task.program` | `metalrobo_task_program_check` | Root capability is explicit and fixed-base execution is qualified, but body/site/goal operators and three complete task families are not yet qualified. |
| `sensors.native_tactile` | **qualified** | `sensors.tactile` | `metalrobo_tactile_check` | Native deformation, contact evidence, wrench summaries, history, and device-buffer access. |
| `sensors.unified_sensor_ir` | **unsupported** | `none` | `none` | Renderer, tactile, contact, and range sensing do not yet compile through one SensorIR schedule. |
| `policy.dense_native_inference` | **qualified** | `task.program` | `metalrobo_task_program_check` | Fingerprint-bound normalization, dense actor/critic layers, Gaussian behavior policy, and native inference. |
| `policy.recurrent_visual_attention` | **unsupported** | `none` | `none` | Convolution, recurrence, and attention are not production PolicyIR operators. |
| `differentiation.native_adjoint` | **unsupported** | `none` | `none` | Forward Jacobians exist, but no production tape, backward kernels, implicit adjoint, or validity-mask API exists. |
| `presentation.visual_v3` | **qualified** | `presentation.visual_platform` | `metalrobo_visual_platform_probe` | Authored Visual Presentation V3 path; collision geometry is not a visual fallback. |

Status meanings:

- **qualified**: an owning check exercises the stated product path.
- **implemented**: code exists, but the full competitive acceptance gate is not published.
- **experimental**: focused research or diagnostic path; not a production promise.
- **unsupported**: compilation or API must not imply this capability exists.
