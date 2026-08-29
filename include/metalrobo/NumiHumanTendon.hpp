#pragma once

#include "metalrobo/MujocoMuscleReference.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class NumiHumanTendonStatus : std::uint32_t {
    success = 0u,
    truncatedPayload,
    invalidPayload,
    sourceMismatch,
    incompleteCoverage,
    invalidBinding,
    nonfiniteResult,
};

enum class NumiHumanTendonAttachmentMode : std::uint32_t {
    sourceSitePoint = 0u,
    registeredBoneTriangle = 1u,
    registeredBoneDistributedEnvelope = 2u,
    registeredBoneMigratedDistributedEnvelope = 3u,
};

struct NumiHumanTendonTriangle {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t boneStableId = 0u;
    std::uint32_t sourceTriangleIndex = MR_INVALID_INDEX;
    std::array<std::array<double, 3>, 3> localVertices{};
};

struct NumiHumanTendonBinding {
    std::uint32_t muscleIndex = MR_INVALID_INDEX;
    std::uint32_t endpointOrdinal = MR_INVALID_INDEX;
    std::uint32_t routeNodeIndex = MR_INVALID_INDEX;
    std::uint32_t sourceSiteIndex = MR_INVALID_INDEX;
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    NumiHumanTendonAttachmentMode mode = NumiHumanTendonAttachmentMode::sourceSitePoint;
    std::uint32_t triangleIndex = MR_INVALID_INDEX;
    std::uint32_t boneStableId = 0u;
    std::array<double, 3> resolvedLocalPoint{};
    std::array<double, 3> barycentric{};
    double endpointMigration = 0.0;
    double surfaceDistance = 0.0;
    double forceAmplification = 0.0;
    double patchRadius = 0.0;
    double compiledMomentResidual = 0.0;
};

struct NumiHumanTendonEnvelope {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t boneStableId = 0u;
    std::uint32_t sourceTriangleIndex = MR_INVALID_INDEX;
    std::array<std::array<double, 3>, 4> localNodes{};
    // source-body-local nodal force = forceMaps[node] * terminal local force.
    std::array<std::array<std::array<double, 3>, 3>, 4> forceMaps{};
    double surfaceDistance = 0.0;
    double patchRadius = 0.0;
    double forceAmplification = 0.0;
    double l2ForceAmplification = 0.0;
};

struct NumiHumanTendonPayload {
    std::uint32_t payloadAbi = 0u;
    std::uint32_t bodyCount = 0u;
    std::uint32_t muscleCount = 0u;
    std::uint32_t sourceSiteCount = 0u;
    std::array<std::uint8_t, 32> sourceSha256{};
    std::array<std::uint8_t, 32> musclePayloadSha256{};
    std::array<std::uint8_t, 32> bonePayloadSha256{};
    std::uint32_t boneCount = 0u;
    std::uint32_t registrationFingerprint = 0u;
    std::vector<NumiHumanTendonBinding> bindings;
    std::vector<NumiHumanTendonTriangle> triangles;
    std::vector<NumiHumanTendonEnvelope> envelopes;
};

struct NumiHumanTendonResolvedProgram {
    std::vector<MujocoMuscleSite> sites;
    std::vector<MujocoMuscleDefinition> muscles;
    std::uint32_t pointBindingCount = 0u;
    std::uint32_t triangleBindingCount = 0u;
    std::uint32_t envelopeBindingCount = 0u;
    std::uint32_t migratedEnvelopeBindingCount = 0u;
    double maximumEndpointMigration = 0.0;
};

struct NumiHumanTendonDiagnostics {
    NumiHumanTendonStatus status = NumiHumanTendonStatus::success;
    std::uint32_t failingIndex = MR_INVALID_INDEX;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanTendonStatus::success;
    }
};

struct NumiHumanTendonTractionResult {
    std::array<double, 3> terminalForce{};
    std::array<std::array<double, 3>, 4> nodalForces{};
    double forceResidual = 0.0;
    double momentResidual = 0.0;
};

struct NumiHumanTendonReferenceCalibration {
    std::vector<double> pathLengthDeltas;
    std::uint32_t calibratedMuscleCount = 0u;
    double maximumAbsolutePathLengthDelta = 0.0;
    double maximumArchitectureScaleChange = 0.0;
};

// Decode NHTENDON1, NHTENDON2, or NHTENDON3. expected hashes may be empty to skip identity
// comparison; production callers pass both source identities from
// NHRIGID2/NHMYO1. NHTENDON2/3 additionally carry the exact NHBONES1 hash and
// registration fingerprint for the caller to bind to loaded geometry.
[[nodiscard]] NumiHumanTendonDiagnostics decodeNumiHumanTendonPayload(
    std::span<const std::byte> bytes,
    std::span<const std::uint8_t> expectedSourceSha256,
    std::span<const std::uint8_t> expectedMusclePayloadSha256,
    NumiHumanTendonPayload& payload
);

// Resolve every route endpoint before force evaluation. Triangle bindings
// receive route-private sites, so a shared source site cannot accidentally
// migrate another muscle. The existing route J^T scatter remains authoritative.
[[nodiscard]] NumiHumanTendonDiagnostics resolveNumiHumanTendonProgram(
    const NumiHumanTendonPayload& payload,
    std::span<const MujocoMuscleSite> sourceSites,
    std::span<const MujocoMuscleDefinition> sourceMuscles,
    NumiHumanTendonResolvedProgram& result
);

// Rebase the source length range and compliant L0/LT at one exact reference
// pose after NHTENDON3 changes a route endpoint. Scaling both architecture
// lengths by resolved/source path length preserves its normalized equilibrium
// at that pose; shifting both source length-range bounds preserves the rigid
// MuJoCo force-law coordinate. The update commits transactionally only after
// every migrated muscle passes its bounds. This is not applied to NHTENDON1/2.
[[nodiscard]] NumiHumanTendonDiagnostics calibrateNumiHumanMigratedTendonReference(
    const EngineModel& model,
    std::uint32_t articulationIndex,
    std::span<const double> referenceQ,
    std::span<const MujocoWrapGeometry> wraps,
    std::span<const MujocoMuscleSite> sourceSites,
    std::span<const MujocoMuscleDefinition> sourceMuscles,
    std::span<const MujocoMuscleSite> resolvedSites,
    std::span<MujocoMuscleDefinition> resolvedMuscles,
    std::span<MujocoCompliantMuscleArchitecture> architectures,
    const NumiHumanTendonPayload& payload,
    NumiHumanTendonReferenceCalibration& calibration
);

// Convert the already-evaluated route tension into an inspection traction
// field. This never scatters generalized force; dynamics continue to use the
// resolved route site exactly once. For a triangle binding, worldTriangle is
// the posed named bone face in the same vertex order as the payload.
[[nodiscard]] NumiHumanTendonDiagnostics evaluateNumiHumanTendonTraction(
    const NumiHumanTendonBinding& binding,
    std::span<const std::array<double, 3>> worldTriangle,
    const std::array<double, 3>& terminalWorld,
    const std::array<double, 3>& adjacentRouteWorld,
    const std::array<double, 3>& bodyOriginWorld,
    double actuatorForce,
    NumiHumanTendonTractionResult& result
);

// Evaluate an NHTENDON2/3 distributed attachment in source-body coordinates.
// The four nodal forces reproduce the resolved endpoint's exact force and
// moment. NHTENDON2 resolves to the source site; NHTENDON3 may use a private
// named-bone surface site without changing the skeleton.
[[nodiscard]] NumiHumanTendonDiagnostics evaluateNumiHumanTendonEnvelopeTraction(
    const NumiHumanTendonBinding& binding,
    const NumiHumanTendonEnvelope& envelope,
    const std::array<double, 3>& terminalLocalForce,
    NumiHumanTendonTractionResult& result
);

[[nodiscard]] const char* numiHumanTendonStatusName(NumiHumanTendonStatus status) noexcept;

} // namespace metalrobo
