#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace metalrobo {

// A bounded conversion for a source model whose physical root is represented
// by a six-coordinate FunctionBased joint below a synthetic fixed anchor.
// The conversion removes only that anchor, requires a stationary source root,
// seeds a standard 7-q/6-v floating root from the exact source default pose,
// and retains all remaining immutable source joint programs. It does not
// equate arbitrary source Euler-coordinate root perturbations with the
// floating-root representation.
struct FunctionBasedMobileRootReduction {
    EngineModel model{};
    // Maps source EngineModel body/joint indices to the reduced model. The
    // removed fixed anchor/root joint maps to MR_INVALID_INDEX.
    std::vector<std::uint32_t> sourceBodyToMobileBody;
    std::vector<std::uint32_t> sourceJointToMobileJoint;
};

enum class FunctionBasedMobileRootStatus : std::uint32_t {
    success = 0u,
    invalidSourceModel,
    unsupportedTopology,
    sourceKinematicsFailed,
    reducedModelInvalid,
};

[[nodiscard]] const char* functionBasedMobileRootStatusName(
    FunctionBasedMobileRootStatus status
) noexcept;

struct FunctionBasedMobileRootDiagnostics {
    FunctionBasedMobileRootStatus status =
        FunctionBasedMobileRootStatus::success;
    std::string message{};

    [[nodiscard]] bool succeeded() const noexcept {
        return status == FunctionBasedMobileRootStatus::success;
    }
};

[[nodiscard]] FunctionBasedMobileRootDiagnostics
reduceFixedFunctionBasedRootToMobileDefaultPose(
    const EngineModel& source,
    FunctionBasedMobileRootReduction& result
);

} // namespace metalrobo
