#include "metalrobo_mlx.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>

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
        )
        .def_rw(
            "endpoint_runtime_records",
            &metalrobo::MetalWorldCapacityProfile::
                endpointRuntimeRecords
        )
        .def_rw(
            "articulation_point_queries",
            &metalrobo::MetalWorldCapacityProfile::
                articulationPointQueries
        )
        .def_rw(
            "rod_candidate_pairs",
            &metalrobo::MetalWorldCapacityProfile::
                rodCandidatePairs
        )
        .def_rw(
            "rod_raw_contacts",
            &metalrobo::MetalWorldCapacityProfile::
                rodRawContacts
        )
        .def_rw(
            "rod_manifolds",
            &metalrobo::MetalWorldCapacityProfile::rodManifolds
        )
        .def_rw(
            "rod_ccd_events",
            &metalrobo::MetalWorldCapacityProfile::rodCCDEvents
        )
        .def_rw(
            "quality_generalized_velocities",
            &metalrobo::MetalWorldCapacityProfile::
                qualityGeneralizedVelocities
        )
        .def_rw(
            "quality_rows",
            &metalrobo::MetalWorldCapacityProfile::qualityRows
        )
        .def_rw(
            "quality_krylov_vectors",
            &metalrobo::MetalWorldCapacityProfile::
                qualityKrylovVectors
        )
        .def_rw(
            "quality_direct_tiles",
            &metalrobo::MetalWorldCapacityProfile::
                qualityDirectTiles
        )
        .def_rw(
            "dynamic_nodes",
            &metalrobo::MetalWorldCapacityProfile::dynamicNodes
        )
        .def_rw(
            "island_node_references",
            &metalrobo::MetalWorldCapacityProfile::
                islandNodeReferences
        )
        .def_rw(
            "island_constraint_references",
            &metalrobo::MetalWorldCapacityProfile::
                islandConstraintReferences
        )
        .def_rw(
            "rod_factor_blocks",
            &metalrobo::MetalWorldCapacityProfile::
                rodFactorBlocks
        )
        .def_rw(
            "operator_velocity_elements",
            &metalrobo::MetalWorldCapacityProfile::
                operatorVelocityElements
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
            "articulation_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().model().articulations.size();
            }
        )
        .def_prop_ro(
            "model_body_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().model().bodies.size();
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
            "rod_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().rodCount();
            }
        )
        .def_prop_ro(
            "rod_node_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().rodNodeCount();
            }
        )
        .def_prop_ro(
            "rod_edge_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().rodEdgeCount();
            }
        )
        .def_prop_ro(
            "rod_witness_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().rodToolPairs().size() *
                    MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR;
            }
        )
        .def_prop_ro(
            "rod_witness_words",
            [](const metalrobo::mlx_ext::MLXCompiledWorld&) {
                return sizeof(MRRodToolWitnessGPU) /
                    sizeof(std::uint32_t);
            }
        )
        .def_prop_ro(
            "floating_root",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                const auto& articulations =
                    world.world().model().articulations;
                return !articulations.empty() &&
                    articulations.front().rootType ==
                        MR_ROOT_FLOATING;
            }
        )
        .def_prop_ro(
            "manifold_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().manifolds;
            }
        )
        .def_prop_ro(
            "candidate_pair_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().candidatePairs;
            }
        )
        .def_prop_ro(
            "raw_contact_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().rawContacts;
            }
        )
        .def_prop_ro(
            "contact_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().constraintBlocks;
            }
        )
        .def_prop_ro(
            "island_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().islands;
            }
        )
        .def_prop_ro(
            "island_constraint_reference_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                const auto& capacity =
                    world.world().capacities();
                return std::min(
                    capacity.islandConstraintReferences,
                    capacity.constraintBlocks
                );
            }
        )
        .def_prop_ro(
            "ccd_candidate_capacity",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.world().capacities().ccdCandidates;
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
            "default_actuator_targets",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                defaultActuatorTargets
        )
        .def_prop_ro(
            "actuator_profile_values",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                actuatorProfileValues
        )
        .def_prop_ro(
            "actuator_profile_flags",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                actuatorProfileFlags
        )
        .def_prop_ro(
            "tactile_sensor_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.tactile().sensors.size();
            }
        )
        .def_prop_ro(
            "tactile_sample_count",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.tactile().samples.size();
            }
        )
        .def_prop_ro(
            "tactile_sensor_ids",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.tactile().sensorIds;
            }
        )
        .def_prop_ro(
            "tactile_observation_metadata_json",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.hasTactile()
                    ? metalrobo::tactileObservationMetadataJSON(
                          world.tactile()
                      )
                    : std::string{};
            }
        )
        .def_prop_ro(
            "authored_pack_hash",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                authoredPackHash
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
            "default_rod_positions",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    4u * world.world().defaultRodNodes().size()
                );
                for (const auto& node :
                     world.world().defaultRodNodes()) {
                    values.insert(
                        values.end(),
                        {
                            node.position.x,
                            node.position.y,
                            node.position.z,
                            node.position.w,
                        }
                    );
                }
                return values;
            }
        )
        .def_prop_ro(
            "default_rod_velocities",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    4u * world.world().defaultRodNodes().size()
                );
                for (const auto& node :
                     world.world().defaultRodNodes()) {
                    values.insert(
                        values.end(),
                        {
                            node.velocity.x,
                            node.velocity.y,
                            node.velocity.z,
                            node.velocity.w,
                        }
                    );
                }
                return values;
            }
        )
        .def_prop_ro(
            "default_rod_twists",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    world.world().defaultRodEdges().size()
                );
                for (const auto& edge :
                     world.world().defaultRodEdges()) {
                    values.push_back(edge.twistAndRate.x);
                }
                return values;
            }
        )
        .def_prop_ro(
            "default_rod_twist_rates",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                std::vector<float> values;
                values.reserve(
                    world.world().defaultRodEdges().size()
                );
                for (const auto& edge :
                     world.world().defaultRodEdges()) {
                    values.push_back(edge.twistAndRate.y);
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
                    case metalrobo::MetalWorldSolverMode::
                        qualityNewton:
                        return "quality_newton";
                }
                return "unsupported";
            }
        )
        .def_prop_ro(
            "velocity_iterations",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                velocityIterations
        )
        .def_prop_ro(
            "final_velocity_iterations",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                finalVelocityIterations
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
            "actuation_mode",
            [](const metalrobo::mlx_ext::MLXCompiledWorld& world) {
                return world.actuationMode() ==
                    metalrobo::MetalWorldActuationMode::
                        implicitPositionDrive
                    ? "implicit_position"
                    : "effort";
            }
        )
        .def_prop_ro(
            "wave_worker_groups",
            &metalrobo::mlx_ext::MLXCompiledWorld::
                waveWorkerGroups
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

    nb::class_<
        metalrobo::mlx_ext::
            MLXCompiledMultiArticulatedProgram
    >(
        module,
        "MLXCompiledMultiArticulatedProgram"
    )
        .def_prop_ro(
            "nq",
            [](const metalrobo::mlx_ext::
                   MLXCompiledMultiArticulatedProgram& program) {
                return program.program().model().world.nq;
            }
        )
        .def_prop_ro(
            "nv",
            [](const metalrobo::mlx_ext::
                   MLXCompiledMultiArticulatedProgram& program) {
                return program.program().model().world.nv;
            }
        )
        .def_prop_ro(
            "row_count",
            [](const metalrobo::mlx_ext::
                   MLXCompiledMultiArticulatedProgram& program) {
                return program.program().rowCount();
            }
        )
        .def_prop_ro(
            "articulation_count",
            [](const metalrobo::mlx_ext::
                   MLXCompiledMultiArticulatedProgram& program) {
                return program.program()
                    .model().articulations.size();
            }
        )
        .def_prop_ro(
            "environment_capacity",
            &metalrobo::mlx_ext::
                MLXCompiledMultiArticulatedProgram::
                    environmentCapacity
        )
        .def_prop_ro(
            "fingerprint",
            [](const metalrobo::mlx_ext::
                   MLXCompiledMultiArticulatedProgram& program) {
                return program.program().fingerprint();
            }
        )
        .def_prop_ro(
            "default_q",
            &metalrobo::mlx_ext::
                MLXCompiledMultiArticulatedProgram::
                    defaultQ
        )
        .def_prop_ro(
            "default_v",
            &metalrobo::mlx_ext::
                MLXCompiledMultiArticulatedProgram::
                    defaultV
        )
        .def(
            "prepare_stream",
            &metalrobo::mlx_ext::
                MLXCompiledMultiArticulatedProgram::
                    prepareStream,
            "stream"_a = nb::none()
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
        "actuation_mode"_a = "effort",
        "solver_mode"_a = "free_motion_aba",
        "velocity_iterations"_a = 1u,
        "final_velocity_iterations"_a = 1u,
        "ccd_mode"_a = "speculative",
        "max_ccd_advance_solve_passes"_a =
            MR_CCD_DEFAULT_ADVANCE_SOLVE_PASSES,
        "max_ccd_zero_time_replays"_a =
            MR_CCD_DEFAULT_ZERO_TIME_REPLAYS,
        "ccd_simultaneous_tolerance"_a = 1.0e-5f,
        "wave_worker_groups"_a = 0u,
        "metallib_path"_a = "",
        "stream"_a = nb::none(),
        "Compile an immutable Franka or G1 world for MLX."
    );
    module.def(
        "compile_multi_articulated_program",
        &metalrobo::mlx_ext::compileMultiArticulatedProgram,
        "model"_a = "dual_psm",
        nb::kw_only(),
        "environment_capacity"_a = 256u,
        "solver_mode"_a = "throughput_pgs",
        "solver_iterations"_a = 192u,
        "convergence_tolerance"_a = 5.0e-5f,
        "timestep"_a = 1.0e-3f,
        "metallib_path"_a = "",
        "stream"_a = nb::none(),
        "Cook a fixed-capacity multi-articulation MLX program."
    );
    module.def(
        "compile_world_pack",
        &metalrobo::mlx_ext::compileWorldPack,
        "path"_a,
        nb::kw_only(),
        "environment_capacity"_a = 1024u,
        "capacity_profile"_a =
            metalrobo::MetalWorldCapacityProfile{},
        "control_timestep"_a = 0.0f,
        "physics_substeps"_a = 4u,
        "apply_body_damping"_a = true,
        "actuation_mode"_a = "implicit_position",
        "solver_mode"_a = "throughput_pgs",
        "velocity_iterations"_a = 1u,
        "final_velocity_iterations"_a = 1u,
        "ccd_mode"_a = "speculative",
        "max_ccd_advance_solve_passes"_a =
            MR_CCD_DEFAULT_ADVANCE_SOLVE_PASSES,
        "max_ccd_zero_time_replays"_a =
            MR_CCD_DEFAULT_ZERO_TIME_REPLAYS,
        "ccd_simultaneous_tolerance"_a = 1.0e-5f,
        "wave_worker_groups"_a = 0u,
        "metallib_path"_a = "",
        "stream"_a = nb::none(),
        "Compile an explicit authored world pack for MLX."
    );
    module.def(
        "generalized_constraint_step",
        &metalrobo::mlx_ext::generalizedConstraintStep,
        "program"_a,
        "q"_a,
        "free_velocity"_a,
        nb::kw_only(),
        "stream"_a = nb::none(),
        "Encode multi-articulation constraints into MLX's active encoder."
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
        "rod_positions"_a,
        "rod_velocities"_a,
        "rod_twists"_a,
        "rod_twist_rates"_a,
        "rod_witness_cache"_a,
        "body_parameters"_a,
        "controller_parameters"_a,
        "tactile_previous_depth"_a,
        "tactile_previous_validity"_a,
        "tactile_previous_object"_a,
        "tactile_previous_motion"_a,
        "tactile_target_anchor"_a,
        "tactile_frame_index"_a,
        "tactile_timestamp"_a,
        "reset_mask"_a,
        "actuator_profile_values"_a,
        nb::kw_only(),
        "stream"_a = nb::none(),
        "Encode a complete contact step into MLX's active Metal encoder."
    );
    module.def(
        "world_family_state",
        &metalrobo::mlx_ext::worldFamilyState,
        "world"_a,
        "reset_q_buffer"_a,
        "reset_v_buffer"_a,
        "reset_scene_bodies_buffer"_a,
        "scenario_headers_buffer"_a,
        "scenario_values_buffer"_a,
        "body_parameters_buffer"_a,
        "controller_parameters_buffer"_a,
        "environment_count"_a,
        "variation_count"_a,
        "body_count"_a,
        "articulation_count"_a,
        "generation"_a,
        nb::kw_only(),
        "stream"_a = nb::none(),
        "Import GPU-resident world-family resets into MLX arrays."
    );
    module.def(
        "visual_observation",
        &metalrobo::mlx_ext::visualObservation,
        "renderer_handle"_a,
        "world_family_handle"_a,
        "current_body_states"_a,
        "previous_body_states"_a,
        "frame_index"_a = 0u,
        "sensor_sequence"_a = 0u,
        "camera_index"_a = 0u,
        nb::kw_only(),
        "stream"_a = nb::none(),
        "Encode synchronized visual modalities on MLX's active Metal "
        "compute encoder."
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
