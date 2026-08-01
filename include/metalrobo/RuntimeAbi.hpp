#pragma once

#include "metalrobo/engine_types.h"
#include "metalrobo/runtime_abi_generated.h"
#include "metalrobo/generalized_constraint_shared.h"
#include "metalrobo/multi_contact_shared.h"
#include "metalrobo/parallel_aba_shared.h"
#include "metalrobo/policy_program_types.h"
#include "metalrobo/r2s2r_types.h"
#include "metalrobo/rod_gpu_shared.h"
#include "metalrobo/scene_query_types.h"
#include "metalrobo/tactile_types.h"
#include "metalrobo/task_program_types.h"
#include "metalrobo/unified_quality_shared.h"
#include "metalrobo/world_compiler_types.h"

#include <cstddef>
#include <cstdint>

namespace metalrobo {
namespace detail {

constexpr std::uint64_t appendRuntimeAbiWord(
    std::uint64_t hash,
    std::uint64_t value
) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= value & 0xffu;
        hash *= 1099511628211ull;
        value >>= 8u;
    }
    return hash;
}

template <typename Type>
constexpr std::uint64_t appendRuntimeAbiType(
    std::uint64_t hash
) noexcept {
    hash = appendRuntimeAbiWord(hash, sizeof(Type));
    return appendRuntimeAbiWord(hash, alignof(Type));
}

} // namespace detail

// Fingerprints the host/Metal records and generated binding schemas crossing
// the native-library boundary. Consumers reject mismatches before GPU work.
constexpr std::uint64_t runtimeAbiFingerprint() noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_ENGINE_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_METAL_WORLD_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_METAL_WORLD_CONTACT_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_RUNTIME_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY |
            MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY |
            MR_ARTICULATED_OPERATOR_BROADCAST_POINTS |
            MR_ARTICULATED_OPERATOR_SPATIAL_JACOBIANS |
            MR_ARTICULATED_OPERATOR_SKIP_GENERALIZED_OUTPUTS
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_METAL_WORLD_CONTACT_DETERMINISTIC |
            MR_METAL_WORLD_CONTACT_WARM_START |
            MR_METAL_WORLD_CONTACT_CAPTURE_EVIDENCE |
            MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS |
            MR_METAL_WORLD_CONTACT_CCD |
            MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS |
            MR_METAL_WORLD_CONTACT_QUALITY |
            MR_METAL_WORLD_CONTACT_BODY_PARAMETERS
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_TACTILE_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_ROD_GPU_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_R2S2R_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_WORLD_COMPILER_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_CONSTRAINT_IR_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_GENERALIZED_CONSTRAINT_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_MULTI_CONTACT_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_PARALLEL_ABA_SCHEDULE_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_SCENE_QUERY_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_UNIFIED_QUALITY_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_TASK_PROGRAM_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_POLICY_PROGRAM_ABI_VERSION
    );
    hash = detail::appendRuntimeAbiWord(
        hash,
        MR_SENSOR_PROGRAM_ABI_VERSION
    );

    hash = detail::appendRuntimeAbiType<MRWorldGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRArticulationGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRJointDescriptorGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRDofPropertiesGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRBodyPropertiesGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRBodyStateGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRShapeGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRGeometryHeaderGPU>(hash);
    hash = detail::appendRuntimeAbiType<
        MRMetalWorldDispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRMetalWorldContactDispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRMetalWorldContactStatusGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<MRTactileSensorGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRTactileSampleGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRTactileDispatchGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRTactileSummaryGPU>(hash);
    hash = detail::appendRuntimeAbiType<MRRodGPUDispatch>(hash);
    hash = detail::appendRuntimeAbiType<
        MRWorldFamilySampleUniformsGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRWorldScenarioHeaderGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRGeneralizedConstraintDispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRMultiContactDispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRMultiABADispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRSceneQueryDispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRUnifiedQualityDispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<MRTaskDispatchGPU>(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskProgramHeaderGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskSignalOperatorGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskCommandOperatorGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskEventOperatorGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskRewardOperatorGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskTerminationOperatorGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<MRTaskStateGPU>(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskCurriculumStateGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<MRTaskTransitionGPU>(hash);
    hash = detail::appendRuntimeAbiType<
        MRPolicyProgramHeaderGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRSensorProgramHeaderGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRSensorDescriptorGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRSensorDispatchGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRSensorRuntimeStateGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRSensorSampleMetadataGPU
    >(hash);
    hash = detail::appendRuntimeAbiType<
        MRTaskKinematicFrameGPU
    >(hash);
    return hash;
}

} // namespace metalrobo
