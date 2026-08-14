#pragma once

#include "metalrobo/engine_types.h"

#include <array>
#include <cstdint>
#include <numbers>
#include <string_view>
#include <vector>

namespace metalrobo {

// Provenance is carried beside every physical input so a published/catalog
// dimension is never confused with a simulator tuning choice.
enum class SurgicalValueBasis : std::uint32_t {
    gs21ProductGeometry = 0u,
    orbitSurgicalOpenAsset = 1u,
    derivedGeometry = 2u,
    researchDefault = 3u,
    pdsII3_0ProductGeometry = 4u,
    roboticBowelClosureTechnique = 5u,
    needleHandlingInstructions = 6u,
    polydioxanoneMonofilamentStudy = 7u,
    uspSyntheticSutureDiameterStandard = 8u,
};

struct SurgicalScalar {
    double value = 0.0;
    SurgicalValueBasis basis = SurgicalValueBasis::researchDefault;
};

[[nodiscard]] std::string_view surgicalValueBasisName(
    SurgicalValueBasis basis
) noexcept;

[[nodiscard]] std::string_view surgicalValueSourceReference(
    SurgicalValueBasis basis
) noexcept;

struct SurgicalAssetIds {
    std::uint32_t bodyIndex = 0u;
    std::uint32_t materialIndex = 0u;
    std::uint32_t slotGenerationBase = 1u;
    std::uint32_t collisionGroup = 1u;
    std::uint32_t collisionMask = ~0u;
    std::uint32_t motionType = MR_MOTION_DYNAMIC;
};

struct CurvedSutureNeedleSpec {
    // GS-21 is a 37 mm, 1/2-circle taper needle. Length is measured along
    // curvature. ORBIT-Surgical reports using GS-21 hardware.
    SurgicalScalar arcLengthM{
        0.037,
        SurgicalValueBasis::gs21ProductGeometry,
    };
    SurgicalScalar arcAngleRad{
        std::numbers::pi,
        SurgicalValueBasis::gs21ProductGeometry,
    };

    // Gauge, material and collision discretization are explicit research
    // defaults pending measurement of a specific packaged needle.
    SurgicalScalar crossSectionRadiusM{
        0.00045,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar densityKgPerM3{
        8000.0,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar tipTaperLengthM{
        0.0040,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar swageLengthM{
        0.0040,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar graspZoneStartFraction{
        0.35,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar graspZoneEndFraction{
        0.70,
        SurgicalValueBasis::researchDefault,
    };
    std::uint32_t arcSegments = 32u;
    double tipRadiusRatio = 0.20;
    double swageRadiusRatio = 1.12;
    double contactOffsetM = 0.00002;
};

// Source-locked bowel-anastomosis setup: USP 3-0 PDS II on a 26 mm curved-
// centerline, half-circle taper needle, with the driving grasp limited to the
// handling interval published for curved surgical needles. Needle gauge,
// taper and collision discretization remain explicit research values because
// the product catalogue does not publish them.
[[nodiscard]] CurvedSutureNeedleSpec
makeBowelAnastomosisNeedleSpec() noexcept;

struct SurgicalNeutralZonePadSpec {
    // A low-profile sterile-field pad is a physical table surface, not a
    // hidden support or kinematic grasp. Its dimensions and contact material
    // are research/training values pending measurement of a selected product.
    SurgicalScalar sizeXM{
        0.180,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar sizeYM{
        0.120,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar thicknessM{
        0.003,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar densityKgPerM3{
        1150.0,
        SurgicalValueBasis::researchDefault,
    };
    double contactOffsetM = 0.00002;
};

struct SurgicalTrainingRingSpec {
    SurgicalScalar majorRadiusM{
        0.0050,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar tubeRadiusM{
        0.00075,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar densityKgPerM3{
        1150.0,
        SurgicalValueBasis::researchDefault,
    };
    std::uint32_t segments = 32u;
    double contactOffsetM = 0.00005;
};

struct SurgicalPegBlockSpec {
    SurgicalScalar baseSizeXM{
        0.090,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar baseSizeYM{
        0.008,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar baseSizeZM{
        0.060,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar pegRadiusM{
        0.0020,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar pegHeightM{
        0.016,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar pegSpacingXM{
        0.025,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar pegSpacingZM{
        0.025,
        SurgicalValueBasis::researchDefault,
    };
    SurgicalScalar densityKgPerM3{
        1180.0,
        SurgicalValueBasis::researchDefault,
    };
    std::uint32_t columns = 3u;
    std::uint32_t rows = 2u;
    double contactOffsetM = 0.00010;
};

struct CurvedSutureNeedleMetadata {
    double centerlineRadiusM = 0.0;
    double representedArcLengthM = 0.0;
    double maximumCenterlineErrorM = 0.0;
    double orbitReferenceMeshScale = 0.0;
    std::array<double, 3> orbitReferenceScaledExtentM{};
    double tipTaperStartM = 0.0;
    double swageEndM = 0.0;
    double graspZoneStartM = 0.0;
    double graspZoneEndM = 0.0;
    std::uint32_t swageShapeBegin = 0u;
    std::uint32_t swageShapeEnd = 0u;
    std::uint32_t graspShapeBegin = 0u;
    std::uint32_t graspShapeEnd = 0u;
    std::uint32_t tipShapeBegin = 0u;
    std::uint32_t tipShapeEnd = 0u;
};

struct SurgicalTrainingRingMetadata {
    double innerRadiusM = 0.0;
    double outerRadiusM = 0.0;
    double maximumCenterlineErrorM = 0.0;
};

struct SurgicalPegBlockMetadata {
    std::uint32_t baseShape = 0u;
    std::uint32_t firstPegShape = 0u;
    std::uint32_t pegCount = 0u;
    std::vector<std::array<double, 3>> pegCenters;
};

struct SurgicalNeutralZonePadMetadata {
    double topSurfaceLocalM = 0.0;
};

// One COM-centred rigid body plus a material and compound primitive colliders.
// bodyIndex/materialIndex/generation are already rebased for direct append to
// an EngineModel or an environment's body/material/shape streams.
struct SurgicalRigidAsset {
    MRBodyPropertiesGPU body{};
    MRMaterialGPU material{};
    std::vector<MRShapeGPU> shapes;
    double volumeM3 = 0.0;
    double massKg = 0.0;
    std::array<double, 3> geometryCenterOfMassM{};
    std::array<double, 3> localAabbLowerM{};
    std::array<double, 3> localAabbUpperM{};
    std::string_view name;
    std::string_view validationBoundary;
};

struct CurvedSutureNeedleAsset {
    SurgicalRigidAsset rigid;
    CurvedSutureNeedleSpec spec;
    CurvedSutureNeedleMetadata metadata;
};

struct SurgicalTrainingRingAsset {
    SurgicalRigidAsset rigid;
    SurgicalTrainingRingSpec spec;
    SurgicalTrainingRingMetadata metadata;
};

struct SurgicalPegBlockAsset {
    SurgicalRigidAsset rigid;
    SurgicalPegBlockSpec spec;
    SurgicalPegBlockMetadata metadata;
};

struct SurgicalNeutralZonePadAsset {
    SurgicalRigidAsset rigid;
    SurgicalNeutralZonePadSpec spec;
    SurgicalNeutralZonePadMetadata metadata;
};

[[nodiscard]] CurvedSutureNeedleAsset makeCurvedSutureNeedleAsset(
    const SurgicalAssetIds& ids = {},
    const CurvedSutureNeedleSpec& spec = {}
);

[[nodiscard]] SurgicalTrainingRingAsset makeSurgicalTrainingRingAsset(
    const SurgicalAssetIds& ids = {},
    const SurgicalTrainingRingSpec& spec = {}
);

[[nodiscard]] SurgicalPegBlockAsset makeSurgicalPegBlockAsset(
    const SurgicalAssetIds& ids = {
        .motionType = MR_MOTION_STATIC,
    },
    const SurgicalPegBlockSpec& spec = {}
);

[[nodiscard]] SurgicalNeutralZonePadAsset
makeSurgicalNeutralZonePadAsset(
    const SurgicalAssetIds& ids = {
        .motionType = MR_MOTION_STATIC,
    },
    const SurgicalNeutralZonePadSpec& spec = {}
);

} // namespace metalrobo
