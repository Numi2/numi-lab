#pragma once

#include "metalrobo/SurgicalWorld.hpp"
#include "metalrobo/WorldCompiler.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace metalrobo {

// Authored plantar shell over the four executable contact spheres on each
// Unitree G1 foot, plus an explicit static training surface.
[[nodiscard]] EngineModel makeUnitreeG1TactileEngineModel();
[[nodiscard]] EpisodeTwin makeUnitreeG1TactileEpisodeTwin();

// Research-only PSM Large Needle Driver scene with one compound tactile atlas
// per inner jaw and an explicitly authored dynamic curved needle.
[[nodiscard]] EngineModel makeDvrkPsmTactileEngineModel();
[[nodiscard]] EpisodeTwin makeDvrkPsmTactileEpisodeTwin();

// Applies the same compound jaw backing configuration to both arms while
// preserving the normal dual-PSM composition metadata.
[[nodiscard]] DualPsmWorld makeDualDvrkPsmTactileWorld(
    const DualPsmWorldConfig& config = {}
);

// Reuses the same two-jaw authoring for each arm in a dual-PSM composition.
// targetShapeIndices must already be expressed in the composed model.
[[nodiscard]] std::vector<TactileSensorSpec>
makeDualDvrkPsmTactileSensors(
    const EngineModel& model,
    const DualPsmWorldMetadata& metadata,
    std::span<const std::uint32_t> targetShapeIndices
);

} // namespace metalrobo
