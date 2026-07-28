#pragma once

#include "metalrobo/EngineModel.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace metalrobo {

struct SurgicalBasePose {
    std::array<float, 3> position{};
    // Body-to-world quaternion, xyzw.
    std::array<float, 4> orientation{
        0.0f,
        0.0f,
        0.0f,
        1.0f,
    };
};

struct DualPsmWorldConfig {
    SurgicalBasePose leftBase{
        .position = {-0.18f, 0.0f, 0.0f},
    };
    SurgicalBasePose rightBase{
        .position = {0.18f, 0.0f, 0.0f},
    };
    std::array<float, 3> gravity{0.0f, 0.0f, -9.81f};
    float timestep = 1.0f / 1000.0f;
    bool lockBases = true;
    bool coupleJaws = true;
};

struct DualPsmWorldMetadata {
    std::array<std::uint32_t, 2> articulationIndices{};
    std::array<std::uint32_t, 2> qOffsets{};
    std::array<std::uint32_t, 2> vOffsets{};
    std::array<std::uint32_t, 2> rootBodies{};
    std::array<std::uint32_t, 2> firstShapes{};
    std::array<std::uint32_t, 2> firstJawVelocity{};
    std::array<std::uint32_t, 2> secondJawVelocity{};
    std::uint32_t baseLockBlockCount = 0u;
    std::uint32_t jawCouplingBlockCount = 0u;
};

struct DualPsmWorld {
    EngineModel model;
    DualPsmWorldMetadata metadata;
};

// Composes two independently placed PSMs into one executable multi-
// articulation EngineModel. Fixed PSM roots are promoted to floating roots so
// their world poses remain explicit semantic state. Six bilateral generalized
// rows per arm lock those bases without hidden kinematic attachments. One
// gear row per arm enforces q_jaw_a + q_jaw_b = 0 when requested.
//
// The returned model is deterministic and valid or the function throws
// std::invalid_argument/std::logic_error without publishing a partial world.
[[nodiscard]] DualPsmWorld makeDualDvrkPsmWorld(
    const DualPsmWorldConfig& config = {}
);

} // namespace metalrobo
