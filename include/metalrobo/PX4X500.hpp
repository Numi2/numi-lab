#pragma once

#include "metalrobo/multicopter_types.h"

#include <array>

namespace metalrobo {

// Source-derived PX4 Gazebo X500 mechanics pinned by robots/px4_x500/UPSTREAM.json.
// This is configuration data for the generic multicopter primitive, not a
// separate runtime path.
[[nodiscard]] MRMulticopterModelGPU makePX4X500MulticopterModel(
    float physicsTimestep
);
[[nodiscard]] std::array<MRMulticopterRotorGPU, MR_MULTICOPTER_MAX_ROTORS>
makePX4X500Rotors();
[[nodiscard]] MRBodyPropertiesGPU makePX4X500BodyProperties();

} // namespace metalrobo
