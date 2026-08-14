#pragma once

#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/VisualPresentation.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace metalrobo {

// Presentation-only bindings for the canonical PSM distal chain. Geometry is
// authored independently of collision primitives and follows authoritative
// articulated/rigid body poses through the V3 visual binding ABI.
struct DvrkSutureVisualBindings {
    std::uint32_t shaftBodyIndex = 3u;
    std::uint32_t wristPitchBodyIndex = 4u;
    std::uint32_t wristYawBodyIndex = 5u;
    std::uint32_t toolBodyIndex = 6u;
    std::uint32_t jawABodyIndex = 7u;
    std::uint32_t jawBBodyIndex = 8u;
    std::uint32_t needleBodyIndex = 9u;
};

struct DvrkSutureVisualStyle {
    std::uint32_t needleArcSections = 128u;
    std::uint32_t needleRadialSections = 16u;
    std::uint32_t threadSubsectionsPerEdge = 3u;
    std::uint32_t threadRadialSections = 10u;
    std::uint32_t instrumentRadialSections = 20u;
};

// Optional visual-only scene additions. The secondary instrument remains
// bound to its own articulated links; the tissue surface is a presentation
// of an explicitly supplied FEM boundary and never becomes collision truth.
struct DvrkSutureVisualScene {
    bool hasSecondaryInstrument = false;
    DvrkSutureVisualBindings secondaryInstrument{};
    // When present, this is the authored physical support geometry used by
    // the replayed world. It prevents presentation from inventing a larger or
    // vertically displaced surgical field around the thread snapshot.
    bool hasSurgicalFieldGeometry = false;
    std::array<double, 3> surgicalFieldCenterM{};
    std::array<double, 3> surgicalFieldHalfExtentM{};
    std::vector<std::array<double, 3>> tissuePositions;
    std::vector<std::array<std::uint32_t, 3>> tissueTriangles;
    std::array<double, 3> tissueTranslationM{};
};

struct DvrkSutureVisualMetrics {
    std::uint32_t vertexCount = 0u;
    std::uint32_t triangleCount = 0u;
    std::uint32_t instanceCount = 0u;
    std::uint32_t needleTriangleCount = 0u;
    std::uint32_t threadTriangleCount = 0u;
    std::uint32_t tissueTriangleCount = 0u;
    double threadCenterlineLengthM = 0.0;
};

struct DvrkSutureVisualAsset {
    VisualAssetPackV2 pack;
    DvrkSutureVisualMetrics metrics;
};

// Builds a compact, high-resolution PBR pack for the source-grounded dVRK
// distal instrument, curved needle, and one physically resolved DER thread
// state. Needle and instrument meshes are body-local and remain bound to live
// physics poses. The deformable thread is baked from the accepted rod state at
// an explicit replay/media boundary; it is never used as collision geometry.
[[nodiscard]] DvrkSutureVisualAsset makeDvrkSutureVisualAsset(
    const CurvedSutureNeedleAsset& needle,
    const DiscreteElasticRodModel& threadModel,
    const DiscreteElasticRodState& threadState,
    const DvrkSutureVisualBindings& bindings = {},
    const DvrkSutureVisualStyle& style = {},
    const DvrkSutureVisualScene& scene = {}
);

} // namespace metalrobo
