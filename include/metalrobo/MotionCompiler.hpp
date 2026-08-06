#pragma once

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/InteractionPack.hpp"
#include "metalrobo/LearningPacks.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace metalrobo {

enum class MotionCompileStatus : std::uint32_t {
    success = 0u,
    invalidInput,
    invalidModel,
    unresolvedClip,
    unresolvedJoint,
    unresolvedBody,
    unsupportedTopology,
    kinematicsFailure,
};

struct InteractionMotionCompileConfig {
    std::string id;
    std::string clipId;
    std::string anchorBody;
    std::vector<std::string> trackedBodies;
    std::uint32_t articulationIndex = 0u;
};

struct MotionCompileResult {
    MotionCompileStatus status = MotionCompileStatus::success;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MotionCompileStatus::success;
    }
};

// Compiles generated joint intent into the exact anchor-relative body-COM
// feature contract emitted by mr_locomotion_task_motion_features. The source
// proposes style only; this compiler never executes or certifies physics.
// Output publication is transactional.
[[nodiscard]] MotionCompileResult compileInteractionMotionPack(
    const InteractionPack& interactions,
    const EngineModel& model,
    const InteractionMotionCompileConfig& config,
    MotionPack& output
);

[[nodiscard]] const char* motionCompileStatusName(
    MotionCompileStatus status
) noexcept;

} // namespace metalrobo
