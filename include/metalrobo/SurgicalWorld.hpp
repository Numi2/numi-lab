#pragma once

#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/SurgicalAssets.hpp"

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

struct DualPsmNeedleThreadWorldConfig {
    DualPsmWorldConfig robots{};
    CurvedSutureNeedleSpec needle{};
    SurgicalBasePose needlePose{
        .position = {0.0f, 0.0f, 0.08f},
    };
    DiscreteRodMaterial threadMaterial{};
    std::uint32_t threadNodeCount = 65u;
    double threadLengthM = 0.18;
    // Direction in the needle body frame. The default leaves the rear swage
    // opposite the increasing needle-arc tangent.
    std::array<double, 3> threadExitDirectionLocal{
        -1.0,
        0.0,
        0.0,
    };
    double attachmentCompliance = 0.0;
};

struct DualPsmNeedleThreadMetadata {
    std::uint32_t needleSceneBodyIndex = 0u;
    std::uint32_t threadAttachmentNode = 0u;
    std::array<double, 3> swageAnchorLocal{};
    std::array<double, 3> swageAnchorWorld{};
    std::array<double, 3> initialThreadDirectionWorld{};
};

// Executable surgical composition. The two PSMs remain in the generic
// multi-articulation EngineModel; the needle is an independent dynamic scene
// body and the thread is a DER coupled at the geometry-derived rear swage.
// The binding is consumed directly by MetalDiscreteElasticRod, creating the
// force chain PSM contact -> needle -> thread without a weld or hidden grasp.
struct DualPsmNeedleThreadWorld {
    DualPsmWorld robots;
    CurvedSutureNeedleAsset needle;
    MRBodyStateGPU needleState{};
    DiscreteElasticRodModel threadModel;
    DiscreteElasticRodState threadState;
    std::array<DiscreteRodAttachment, 1> attachments{};
    std::array<DiscreteRodRigidAttachmentBinding, 1> rigidBindings{};
    DualPsmNeedleThreadMetadata metadata;
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

// Transactionally composes the dual-PSM mechanism, dynamic GS-21-scale
// needle, and physically parameterized thread reset state.
[[nodiscard]] DualPsmNeedleThreadWorld
makeDualDvrkPsmNeedleThreadWorld(
    const DualPsmNeedleThreadWorldConfig& config = {}
);

} // namespace metalrobo
