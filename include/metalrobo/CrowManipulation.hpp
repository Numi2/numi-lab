#pragma once
#include "metalrobo/CrowForaging.hpp"

namespace metalrobo {
// Ten-degree downward engineering camera pitch covers the beak workspace.
inline constexpr mr_float4 crowObjectPushCameraOrientation={-.5416752204f,.5416752204f,-.4545194777f,.4545194777f};
// Estimated engineering task: an initially resting rigid 2g sphere on a
// support, moved by ordinary solved contacts. No grasp latch or host motion.
// Disabling crow/object contact is an evaluation control with a distinct world.
[[nodiscard]] RunManifest makeAmericanCrowObjectPushFixtureManifest(bool crowObjectContact=true);
[[nodiscard]] TaskPack makeAmericanCrowObjectPushTaskPack(
    TaskObservationProgram& observations,TaskResetProgram& reset,
    const RetinalSensorProgram* retinal=nullptr);
[[nodiscard]] VisualSensorProgram authorAmericanCrowObjectPushRetinalProgram(
    const CompiledRun& run,const CrowForagingVisual& visual);
} // namespace metalrobo
