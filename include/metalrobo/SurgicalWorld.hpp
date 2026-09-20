#pragma once

#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace metalrobo {

// A composed dual PSM promotes each canonical fixed root to an explicit
// floating pose.  Keep these widths public so state producers and visual
// consumers validate the same serialized ABI instead of mistaking the eight
// actuated joint coordinates for the complete articulation state.
inline constexpr std::uint32_t kSurgicalFloatingRootQCount = 7u;
inline constexpr std::uint32_t kSurgicalFloatingRootVCount = 6u;
inline constexpr std::uint32_t kDualPsmQCount =
    2u * (kSurgicalFloatingRootQCount + kSurgicalPSMJointCount);
inline constexpr std::uint32_t kDualPsmVCount =
    2u * (kSurgicalFloatingRootVCount + kSurgicalPSMJointCount);

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
    // Effective dry PDO contact preset for the unresolved 3-0 monofilament.
    // Published studies support qualitative low-friction monofilament
    // behavior, but do not establish this exact instrument/pad pair. These
    // coefficients therefore remain explicit research calibration values.
    // The DER self-contact path conservatively uses the dynamic component as
    // its single Coulomb coefficient until separate static/sliding data exist.
    MRMaterialGPU threadContactMaterial{
        .friction = {0.18f, 0.12f, 0.0f, 0.0f},
        .response = {0.0f, 0.05f, 1.0e-9f, 0.0f},
        .geometry = {0.0f, 0.0f, 0.0f, 0.0f},
    };
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
    // Two-axis swing compliance of the thread exit direction in rad/Nm. The
    // swaged product construction clamps this direction, so zero is the
    // source-sized default. Axial response remains the DER stretch mode.
    double tangentAttachmentComplianceRadPerNm = 0.0;
    // Permanent swage torsional compliance in rad/Nm. The product construction
    // does not permit thread rotation inside the crimp, so the source-sized
    // default is a hard material-frame weld rather than numerical damping.
    double torsionalAttachmentComplianceRadPerNm = 0.0;
};

// A source-traceable training specimen for robotic bowel closure. Product
// records own needle curvature and USP size; the 25 cm working length follows
// a published robotic bowel technique. The measured PDO-wire modulus and
// Poisson ratio are material-scale priors rather than package calibration,
// while density remains an explicit research default.
struct BowelAnastomosisSutureSpec {
    CurvedSutureNeedleSpec needle{};
    SurgicalScalar threadLengthM{
        0.25,
        SurgicalValueBasis::roboticBowelClosureTechnique,
    };
    // Radius is half the 0.200 mm lower diameter bound for USP 3-0 synthetic
    // suture, not a claim that every D7463 strand measures exactly 0.200 mm.
    SurgicalScalar threadRadiusM{
        0.00010,
        SurgicalValueBasis::uspSyntheticSutureDiameterStandard,
    };
    SurgicalScalar threadDensityKgPerM3{
        1300.0,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar threadYoungModulusPa{
        958.0e6,
        SurgicalValueBasis::polydioxanoneMonofilamentStudy,
    };
    SurgicalScalar threadPoissonRatio{
        0.34,
        SurgicalValueBasis::polydioxanoneMonofilamentStudy,
    };
    // The production DER kernel owns a checked 128-node SIMD bucket. At the
    // 25 cm technique length this resolves the monofilament at 1.97 mm per
    // edge without asking the GPU to execute an unsupported topology.
    std::uint32_t threadNodeCount = 128u;
};

[[nodiscard]] BowelAnastomosisSutureSpec
makeBowelAnastomosisSutureSpec() noexcept;

[[nodiscard]] DualPsmNeedleThreadWorldConfig
makeBowelAnastomosisNeedleThreadWorldConfig(
    const BowelAnastomosisSutureSpec& spec =
        makeBowelAnastomosisSutureSpec()
);

struct DualPsmNeedleThreadMetadata {
    std::uint32_t needleSceneBodyIndex = 0u;
    std::uint32_t threadAttachmentNode = 0u;
    std::uint32_t hardSwagedThreadNodeCount = 1u;
    std::uint32_t threadBoundaryNodeCount = 2u;
    std::array<double, 3> swageAnchorLocal{};
    std::array<double, 3> swageAnchorWorld{};
    std::array<double, 3> initialThreadDirectionWorld{};
    std::array<double, 3> swageMaterialDirectorLocal{};
    double swageTangentComplianceRadPerNm = 0.0;
};

// Executable surgical composition. The two PSMs remain in the generic
// multi-articulation EngineModel; the needle is an independent dynamic scene
// body and the thread is a DER coupled at the geometry-derived rear swage.
// The root node is hard-coupled to the steel swage. Two transverse rows keep
// the first edge on the swage line while leaving axial stretch to the DER, so
// no second node is welded to the steel. A scalar material-frame boundary at
// the first edge completes the permanent swage and removes the otherwise free
// uniform-twist mode. The
// persistent Metal world cooks all three relations into one coupled island,
// creating the force/torque chain PSM contact -> needle -> thread without a
// hidden gripper weld.
struct DualPsmNeedleThreadWorld {
    DualPsmWorld robots;
    CurvedSutureNeedleAsset needle;
    MRBodyStateGPU needleState{};
    DiscreteElasticRodModel threadModel;
    DiscreteElasticRodState threadState;
    MRMaterialGPU threadContactMaterial{};
    std::array<DiscreteRodAttachment, 1> attachments{};
    std::array<DiscreteRodRigidAttachmentBinding, 1> rigidBindings{};
    std::array<DiscreteRodRigidTangentAttachmentBinding, 1>
        tangentBindings{};
    std::array<DiscreteRodRigidTwistAttachmentBinding, 1> twistBindings{};
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

// Transactionally composes the dual-PSM mechanism, dynamic curved needle,
// and physically parameterized thread reset state.
[[nodiscard]] DualPsmNeedleThreadWorld
makeDualDvrkPsmNeedleThreadWorld(
    const DualPsmNeedleThreadWorldConfig& config = {}
);

} // namespace metalrobo
