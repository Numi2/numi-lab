#include "metalrobo_mlx.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_mlx_ext, module) {
    module.doc() =
        "MLX-native MetalRobo primitives using MLX's active encoder";

    nb::class_<metalrobo::MetalWorldCapacityProfile>(
        module,
        "MetalWorldCapacityProfile"
    )
        .def(nb::init<>())
        .def_rw(
            "candidate_pairs",
            &metalrobo::MetalWorldCapacityProfile::candidatePairs
        )
        .def_rw(
            "raw_contacts",
            &metalrobo::MetalWorldCapacityProfile::rawContacts
        )
        .def_rw(
            "manifolds",
            &metalrobo::MetalWorldCapacityProfile::manifolds
        )
        .def_rw(
            "constraint_blocks",
            &metalrobo::MetalWorldCapacityProfile::
                constraintBlocks
        )
        .def_rw(
            "constraint_rows",
            &metalrobo::MetalWorldCapacityProfile::constraintRows
        )
        .def_rw(
            "islands",
            &metalrobo::MetalWorldCapacityProfile::islands
        )
        .def_rw(
            "hard_convex_pairs",
            &metalrobo::MetalWorldCapacityProfile::
                hardConvexPairs
        )
        .def_rw(
            "mesh_triangle_candidates",
            &metalrobo::MetalWorldCapacityProfile::
                meshTriangleCandidates
        )
        .def_rw(
            "solver_tiles",
            &metalrobo::MetalWorldCapacityProfile::solverTiles
        )
        .def_rw(
            "spill_rows",
            &metalrobo::MetalWorldCapacityProfile::spillRows
        )
        .def_rw(
            "ccd_candidates",
            &metalrobo::MetalWorldCapacityProfile::ccdCandidates
        )
        .def_rw(
            "ccd_events",
            &metalrobo::MetalWorldCapacityProfile::ccdEvents
        );

    nb::class_<metalrobo::mlx_ext::MLXCompiledWorld>(
        module,
        "MLXCompiledWorld"
    )
        .def_prop_ro(
            "nq",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().nq();
            }
        )
        .def_prop_ro(
            "nv",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().nv();
            }
        )
        .def_prop_ro(
            "body_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().bodyCount();
            }
        )
        .def_prop_ro(
            "collider_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().colliderCount();
            }
        )
        .def_prop_ro(
            "eligible_pair_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().eligiblePairCount();
            }
        )
        .def_prop_ro(
            "control_timestep",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                controlTimestep
        )
        .def_prop_ro(
            "physics_substeps",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                physicsSubsteps
        )
        .def_prop_ro(
            "apply_body_damping",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                applyBodyDamping
        )
        .def_prop_ro(
            "environment_capacity",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                environmentCapacity
        )
        .def_prop_ro(
            "scene_body_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().sceneBodyCount();
            }
        )
        .def_prop_ro(
            "manifold_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().manifolds;
            }
        )
        .def_prop_ro(
            "contact_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().constraintBlocks;
            }
        )
        .def_prop_ro(
            "pair_cache_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().eligiblePairCount();
            }
        )
        .def_prop_ro(
            "manifold_header_words",
            [](const metalrobo::mlx_ext::MLXCompiledWorld&) {
                return sizeof(MRManifoldHeaderGPU) /
                    sizeof(std::uint32_t);
            }
        )
        .def_prop_ro(
            "manifold_point_words",
            [](const metalrobo::mlx_ext::MLXCompiledWorld&) {
                return sizeof(MRManifoldPointGPU) /
                    sizeof(std::uint32_t);
            }
        )
        .def_prop_ro(
            "manifold_point_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld&) {
                return MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
            }
        )
        .def_prop_ro(
            "pair_cache_words",
            [](const metalrobo::mlx_ext::MLXCompiledWorld&) {
                return sizeof(MRConvexQueryCacheGPU) /
                    sizeof(std::uint32_t);
            }
        )
        .def_prop_ro(
            "metallib_path",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                metallibPath
        )
        .def_prop_ro(
            "default_q",
            &metalrobo::mlx_ext::MLXCompiledWorld::defaultQ
        )
        .def_prop_ro(
            "default_v",
            &metalrobo::mlx_ext::MLXCompiledWorld::defaultV
        )
        .def_prop_ro(
            "effort_limits",
            &metalrobo::mlx_ext::MLXCompiledWorld::effortLimits
        )
        .def_prop_ro(
            "default_scene_positions",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    4u * world.defaultSceneBodies().size()
                );
                for (const auto& body : world.defaultSceneBodies()) {
                    values.insert(
                        values.end(),
                        {
                            body.position.x,
                            body.position.y,
                            body.position.z,
                            body.position.w,
                        }
                    );
                }
                return values;
            }
        )
        .def_prop_ro(
            "default_scene_orientations",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    4u * world.defaultSceneBodies().size()
                );
                for (const auto& body : world.defaultSceneBodies()) {
                    values.insert(
                        values.end(),
                        {
                            body.orientation.x,
                            body.orientation.y,
                            body.orientation.z,
                            body.orientation.w,
                        }
                    );
                }
                return values;
            }
        )
        .def_prop_ro(
            "default_scene_linear_velocities",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    4u * world.defaultSceneBodies().size()
                );
                for (const auto& body : world.defaultSceneBodies()) {
                    values.insert(
                        values.end(),
                        {
                            body.linearVelocityAndInverseMass.x,
                            body.linearVelocityAndInverseMass.y,
                            body.linearVelocityAndInverseMass.z,
                            0.0f,
                        }
                    );
                }
                return values;
            }
        )
        .def_prop_ro(
            "default_scene_angular_velocities",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    4u * world.defaultSceneBodies().size()
                );
                for (const auto& body : world.defaultSceneBodies()) {
                    values.insert(
                        values.end(),
                        {
                            body.angularVelocity.x,
                            body.angularVelocity.y,
                            body.angularVelocity.z,
                            0.0f,
                        }
                    );
                }
                return values;
            }
        )
        .def_prop_ro(
            "solver_mode",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                switch (world.solverMode()) {
                    case metalrobo::MetalWorldSolverMode::
                        freeMotionABA:
                        return "free_motion_aba";
                    case metalrobo::MetalWorldSolverMode::
                        throughputPGS:
                        return "throughput_pgs";
                    case metalrobo::MetalWorldSolverMode::
                        throughputTGS:
                        return "throughput_tgs";
                }
                return "unsupported";
            }
        )
        .def_prop_ro(
            "ccd_mode",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                switch (world.ccdMode()) {
                    case metalrobo::MetalWorldCCDMode::disabled:
                        return "disabled";
                    case metalrobo::MetalWorldCCDMode::speculative:
                        return "speculative";
                    case metalrobo::MetalWorldCCDMode::hybrid:
                        return "hybrid";
                }
                return "unsupported";
            }
        )
        .def_prop_ro(
            "contact_supported",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.solverMode() !=
                        metalrobo::MetalWorldSolverMode::
                            freeMotionABA &&
                    world.world().sceneBodyCount() != 0u;
            }
        )
        .def(
            "prepare_stream",
            &metalrobo::mlx_ext::MLXCompiledWorld::prepareStream,
            "stream"_a = nb::none(),
            "Prepare immutable Metal resources for an MLX stream."
        );

    module.def(
        "compile_world",
        &metalrobo::mlx_ext::compileWorld,
        "model"_a = "franka",
        nb::kw_only(),
        "scene"_a = "none",
        "environment_capacity"_a = 1024u,
        "capacity_profile"_a =
            metalrobo::MetalWorldCapacityProfile{},
        "control_timestep"_a = 1.0f / 60.0f,
        "physics_substeps"_a = 4u,
        "apply_body_damping"_a = true,
        "solver_mode"_a = "free_motion_aba",
        "ccd_mode"_a = "speculative",
        "max_ccd_advance_solve_passes"_a =
            MR_CCD_DEFAULT_ADVANCE_SOLVE_PASSES,
        "max_ccd_zero_time_replays"_a =
            MR_CCD_DEFAULT_ZERO_TIME_REPLAYS,
        "ccd_simultaneous_tolerance"_a = 1.0e-5f,
        "metallib_path"_a = "",
        "stream"_a = nb::none(),
        "Compile an immutable Franka or G1 world for MLX."
    );
    module.def(
        "aba_step",
        &metalrobo::mlx_ext::abaStep,
        "world"_a,
        "q"_a,
        "v"_a,
        "actions"_a,
        nb::kw_only(),
        "stream"_a = nb::none(),
        "Encode a transactional ABA step into MLX's active Metal encoder."
    );
    module.def(
        "world_step",
        &metalrobo::mlx_ext::worldStep,
        "world"_a,
        "q"_a,
        "v"_a,
        "actions"_a,
        "scene_position"_a,
        "scene_orientation"_a,
        "scene_linear_velocity"_a,
        "scene_angular_velocity"_a,
        "manifold_headers"_a,
        "manifold_points"_a,
        "manifold_counts"_a,
        "pair_cache"_a,
        nb::kw_only(),
        "stream"_a = nb::none(),
        "Encode a complete contact step into MLX's active Metal encoder."
    );
    module.def(
        "_debug_cpu_step",
        &metalrobo::mlx_ext::debugCPUStep,
        "world"_a,
        "q"_a,
        "v"_a,
        "actions"_a,
        "Run one synchronous FP64 step for validation only."
    );
}
