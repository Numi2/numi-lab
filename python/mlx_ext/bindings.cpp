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
            "solver_mode",
            [](const metalrobo::mlx_ext::MLXCompiledWorld&) {
                return "free_motion_aba";
            }
        )
        .def_prop_ro(
            "contact_supported",
            [](const metalrobo::mlx_ext::MLXCompiledWorld&) {
                return false;
            }
        );

    module.def(
        "compile_world",
        &metalrobo::mlx_ext::compileWorld,
        "model"_a = "franka",
        nb::kw_only(),
        "control_timestep"_a = 1.0f / 60.0f,
        "physics_substeps"_a = 4u,
        "apply_body_damping"_a = true,
        "metallib_path"_a = "",
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
        "_debug_cpu_step",
        &metalrobo::mlx_ext::debugCPUStep,
        "world"_a,
        "q"_a,
        "v"_a,
        "actions"_a,
        "Run one synchronous FP64 step for validation only."
    );
}
