#pragma once
#include "metalrobo/VisualPresentation.hpp"
#include "metalrobo/EngineModel.hpp"
#include <span>

namespace metalrobo {
// Inspection/calibration assets derived from collision geometry. Boxes retain
// exact dimensions and COM-local transforms. Infinite +Y collision planes
// become explicitly bounded quads; callers must admit that extent against
// their camera frusta. Unsupported/disabled geometry fails closed. This does
// not change, replace, or add any physics geometry.
[[nodiscard]] VisualAssetPackV2 makeCollisionProxyVisualAssetPack(
    const EngineModel& model, std::span<const std::uint32_t> bodyIndices,
    std::uint64_t sourceWorldFingerprint, const std::string& id,
    mr_float4 linearColor, float planeHalfExtentMeters);
} // namespace metalrobo
