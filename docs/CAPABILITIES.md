<!-- GENERATED FILE: python/generate_capability_matrix.py -->
# MetalRobo capability matrix

This file is generated from `schemas/capability_matrix.json`.
A status is a product claim boundary, not a roadmap estimate. A
qualified row is rejected unless an evidence manifest records its
owning check as passed.

Current registry: qualified: 8, implemented: 3, experimental: 4, unsupported: 4.

| Capability | Status | Owning check | Executable | Last evidence | Exact scope |
|---|---|---|---|---|---|
| `compiler.urdf_srdf` | **qualified** | `compiler.robot_description` | `metalrobo_robot_description_cooker_probe` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Rigid articulated URDF/SRDF import, semantic names, meshes, and deterministic fingerprints. |
| `compiler.mjcf` | **unsupported** | `none` | `none` | none | No production MJCF parser. A pinned G1 companion-MJCF preset is data, not general MJCF support. |
| `physics.numi_solver` | **qualified** | `physics.metal_contact` | `metalrobo_metal_world_contact_probe` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Qualified Apple-Metal rigid, articulated, and rod solver path: fixed temporal microsteps, coupled nonlinear block sweeps, retained banded rod response for generalized and surface-contact rows, immediate integration, and transactional warm state. This does not qualify all rod mechanics or the residual-converged profile. |
| `physics.numi_residual_converged` | **experimental** | `none` | `metalrobo_metal_unified_quality_probe` | none | Private residual-converged NumiSolver profile is available for focused study; its current backend is transitional and not yet the qualified production path. |
| `physics.true_temporal_tgs` | **unsupported** | `none` | `none` | none | NumiSolver uses temporal small steps but does not claim the complete vendor-specific PhysX TGS contract. |
| `physics.constraint_ir` | **implemented** | `physics.constraint_ir` | `metalrobo_constraint_ir_probe` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Shared row, endpoint, cone, stable-key, and warm-start representation. Active scalar joint limits and contacts are coupled in NumiSolver; generic runtime widths beyond three rows remain incomplete. |
| `physics.literal_convex_ccd` | **experimental** | `physics.metal_contact` | `metalrobo_metal_world_contact_probe` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Selected convex event-time paths exist; complete advance-to-impact, solve, and continue coverage is not qualified. |
| `mechanics.g1_source_exact_presets` | **implemented** | `mechanics.actuation` | `metalrobo_g1_model_probe` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Distinct fingerprinted URDF, companion-MJCF, and RL Lab actuator contracts; no silent 25/35/50 N m mixing. |
| `runtime.swift_metal_session` | **qualified** | `integration.native_simulation` | `metalrobo_simulation` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Swift schedules bounded native submissions; Metal owns persistent simulator state and transactions. |
| `learning.swift_mlx_ppo` | **implemented** | `integration.native_simulation` | `metalrobo_train` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | In-process Swift/MLX learner. Python PPO and learner-worker runtimes are removed. |
| `tasks.compiled_native_runtime` | **qualified** | `task.program` | `metalrobo_task_program_check` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Fixed- and floating-base tasks share compiled semantic bindings, native joint/contact observations, reward, termination, reset, curriculum, and randomization. |
| `tasks.generic_typed_operators` | **experimental** | `task.program` | `metalrobo_task_program_check` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Compiled selected-articulation body frames, static SE(3) goals, pose observations, pose rewards, pose terminations, and reset-correct pre-policy kinematics are implemented. Scene-object/site semantics, twists, accelerations, sampled or trajectory goals, generic reductions, and three complete task families are not yet qualified. |
| `sensors.native_tactile` | **qualified** | `sensors.tactile` | `metalrobo_tactile_check` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Native deformation, contact evidence, wrench summaries, history, and device-buffer access. |
| `sensors.native_state_schedule` | **qualified** | `task.program` | `metalrobo_task_program_check` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Native pre-control parent-frame pose sensors use topology-derived persistent buffers, deterministic nanosecond phase accumulation, whole-sample latency history, per-environment transactional reset, and compact value/age/validity publication. The owner covers an exact 50 Hz schedule and a 30 Hz non-divisor schedule over a 20 ms control period. |
| `sensors.unified_sensor_ir` | **experimental** | `task.program` | `metalrobo_task_program_check` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | WorldPack state, image, and tactile declarations compile transactionally into one generated-ABI descriptor table. State-pose sensors now execute in the session-owned scheduler; presentation and tactile still use their existing native owners, and TaskIR sensor references, deterministic corruption, recorder routing, contact, ray, LiDAR, force-torque, and IMU operators remain incomplete. |
| `policy.dense_native_inference` | **qualified** | `task.program` | `metalrobo_task_program_check` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Fingerprint-bound normalization, dense actor/critic layers, Gaussian behavior policy, and native inference. |
| `policy.recurrent_visual_attention` | **unsupported** | `none` | `none` | none | Convolution, recurrence, and attention are not production PolicyIR operators. |
| `differentiation.native_adjoint` | **unsupported** | `none` | `none` | none | Forward Jacobians exist, but no production tape, backward kernels, implicit adjoint, or validity-mask API exists. |
| `presentation.visual_v3` | **qualified** | `presentation.visual_platform` | `metalrobo_visual_platform_probe` | `db62e37` at 2026-08-01T00:13:59+02:00 (`evidence/baselines/2026-08-01-db62e37.json`) | Authored Visual Presentation V3 path; collision geometry is not a visual fallback. |

Status meanings:

- **qualified**: an owning check exercises the stated product path.
- **implemented**: code exists, but the full competitive acceptance gate is not published.
- **experimental**: focused research or diagnostic path; not a production promise.
- **unsupported**: compilation or API must not imply this capability exists.
