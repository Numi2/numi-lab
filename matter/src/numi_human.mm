#include "numi/matter/numi_human.hpp"

#include <CommonCrypto/CommonDigest.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace numi::matter {
namespace {

static_assert(
    NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS ==
        MR_NUMI_HUMAN_STAND_MAX_STEPS);

std::uint64_t appendFingerprint(
    std::uint64_t fingerprint,
    const void* bytes,
    const std::size_t count
) noexcept {
    const auto* values = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0u; index < count; ++index) {
        fingerprint ^= values[index];
        fingerprint *= 1099511628211ull;
    }
    return fingerprint;
}

bool finiteScale(const nm_float4& scale) noexcept {
    return std::isfinite(scale.x) && std::isfinite(scale.y) &&
        std::isfinite(scale.z) && std::isfinite(scale.w);
}

float scaleComponent(
    const nm_float4& scale, const std::uint32_t index
) noexcept {
    switch (index) {
        case 0u: return scale.x;
        case 1u: return scale.y;
        case 2u: return scale.z;
        case 3u: return scale.w;
    }
    return std::numeric_limits<float>::quiet_NaN();
}

constexpr std::string_view kAdapterSnapshotDomain =
    "numi.matter.numi-human-tendon-fem-adapter-authority.v1";

class AdapterSnapshotWriter {
public:
    bool appendU32(
        const std::string_view field, const std::uint32_t value
    ) {
        return appendFieldName(field) && appendLittleEndian(value);
    }

    bool appendU64(
        const std::string_view field, const std::uint64_t value
    ) {
        return appendFieldName(field) && appendLittleEndian(value);
    }

    template <typename T>
    bool appendVector(
        const std::string_view field, const std::vector<T>& values
    ) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!appendFieldName(field) ||
            !appendLittleEndian(static_cast<std::uint64_t>(sizeof(T))) ||
            !appendLittleEndian(static_cast<std::uint64_t>(values.size()))) {
            return false;
        }
        if (values.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            return false;
        }
        const std::size_t byteCount = values.size() * sizeof(T);
        if (byteCount != 0u) {
            const auto* begin = reinterpret_cast<const std::uint8_t*>(
                values.data());
            bytes_.insert(bytes_.end(), begin, begin + byteCount);
        }
        return true;
    }

    bool appendDomain() {
        return appendString(kAdapterSnapshotDomain);
    }

    [[nodiscard]] std::vector<std::uint8_t> take() && {
        return std::move(bytes_);
    }

private:
    template <typename T>
    bool appendLittleEndian(const T value) {
        static_assert(std::is_unsigned_v<T>);
        for (std::size_t byte = 0u; byte < sizeof(T); ++byte) {
            bytes_.push_back(static_cast<std::uint8_t>(
                value >> (8u * byte)));
        }
        return true;
    }

    bool appendString(const std::string_view value) {
        return appendLittleEndian(static_cast<std::uint64_t>(value.size())) &&
            (bytes_.insert(bytes_.end(), value.begin(), value.end()), true);
    }

    bool appendFieldName(const std::string_view field) {
        return appendString(field);
    }

    std::vector<std::uint8_t> bytes_;
};

template <typename T>
bool sameVectorBytes(
    const std::vector<T>& lhs, const std::vector<T>& rhs
) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    return lhs.size() == rhs.size() &&
        (lhs.empty() ||
         std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0);
}

} // namespace

bool canonicalNumiHumanTendonFEMLoadAdapterSnapshotV1(
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& snapshot,
    std::vector<std::uint8_t>& output,
    std::string& error
) noexcept {
    try {
        error.clear();
        if (!snapshot.available ||
            snapshot.formatVersion !=
                kNumiHumanTendonFEMLoadAdapterSnapshotVersionV1) {
            error = "Numi Human tendon/FEM adapter snapshot is unavailable or has an unsupported version";
            return false;
        }
        AdapterSnapshotWriter writer;
        if (!writer.appendDomain() ||
            !writer.appendU32("formatVersion", snapshot.formatVersion) ||
            !writer.appendU64(
                "adapterFingerprint", snapshot.adapterFingerprint) ||
            !writer.appendU64(
                "runtimeDeviceProgramFingerprint",
                snapshot.runtimeDeviceProgramFingerprint) ||
            !writer.appendU32(
                "environmentCount", snapshot.environmentCount) ||
            !writer.appendU32("endpointCount", snapshot.endpointCount) ||
            !writer.appendU32("femNodeCount", snapshot.femNodeCount) ||
            !writer.appendU32(
                "contactSampleCount", snapshot.contactSampleCount) ||
            !writer.appendU32(
                "femBodyContactSampleCount",
                snapshot.femBodyContactSampleCount) ||
            !writer.appendU32(
                "articularContactSampleCount",
                snapshot.articularContactSampleCount) ||
            !writer.appendU32(
                "articularContactPairCount",
                snapshot.articularContactPairCount) ||
            !writer.appendU32(
                "passiveLigamentCount", snapshot.passiveLigamentCount) ||
            !writer.appendU32(
                "passiveRoutedBandCount",
                snapshot.passiveRoutedBandCount) ||
            !writer.appendU32(
                "boundBodyPoseStride", snapshot.boundBodyPoseStride) ||
            !writer.appendU32(
                "encodedPassCount", snapshot.encodedPassCount) ||
            !writer.appendU32("abortCount", snapshot.abortCount) ||
            !writer.appendU32(
                "articularAttemptedStepCount",
                snapshot.articularAttemptedStepCount) ||
            !writer.appendVector(
                "anchorReactionAcceptedHistory:nm_float4",
                snapshot.anchorReactionAcceptedHistory) ||
            !writer.appendVector(
                "femBodyContactAcceptedHistory:NMNumiHumanArticularContactAuditGPU",
                snapshot.femBodyContactAcceptedHistory) ||
            !writer.appendVector(
                "articularContactAcceptedHistory:NMNumiHumanArticularContactAuditGPU",
                snapshot.articularContactAcceptedHistory) ||
            !writer.appendVector(
                "articularContactPairForceAcceptedHistory:f32",
                snapshot.articularContactPairForceAcceptedHistory) ||
            !writer.appendVector(
                "passiveLigamentAcceptedState:NMNumiHumanPassiveLigamentAuditGPU",
                snapshot.passiveLigamentAcceptedState) ||
            !writer.appendVector(
                "passiveRoutedBandAcceptedState:NMNumiHumanPassiveRoutedBandAuditGPU",
                snapshot.passiveRoutedBandAcceptedState)) {
            error = "Numi Human tendon/FEM adapter snapshot canonical framing overflowed";
            return false;
        }
        std::vector<std::uint8_t> candidate = std::move(writer).take();
        output = std::move(candidate);
        return true;
    } catch (const std::exception& exception) {
        error = std::string(
            "Numi Human tendon/FEM adapter snapshot canonicalization failed: ") +
            exception.what();
        return false;
    } catch (...) {
        error =
            "Numi Human tendon/FEM adapter snapshot canonicalization failed";
        return false;
    }
}

bool digestNumiHumanTendonFEMLoadAdapterSnapshotV1(
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& snapshot,
    NumiHumanTendonFEMLoadAdapterDigest& output,
    std::string& error
) noexcept {
    std::vector<std::uint8_t> canonical;
    if (!canonicalNumiHumanTendonFEMLoadAdapterSnapshotV1(
            snapshot, canonical, error)) {
        return false;
    }
    if (canonical.size() > std::numeric_limits<CC_LONG>::max()) {
        error = "Numi Human tendon/FEM adapter authority exceeds SHA-256 input bounds";
        return false;
    }
    NumiHumanTendonFEMLoadAdapterDigest candidate{};
    if (CC_SHA256(
            canonical.data(), static_cast<CC_LONG>(canonical.size()),
            candidate.data()) == nullptr) {
        error = "Numi Human tendon/FEM adapter SHA-256 failed";
        return false;
    }
    output = candidate;
    error.clear();
    return true;
}

bool sameNumiHumanTendonFEMLoadAdapterSnapshotAuthorityV1(
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& lhs,
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& rhs
) noexcept {
    return lhs.available && rhs.available &&
        lhs.formatVersion == rhs.formatVersion &&
        lhs.adapterFingerprint == rhs.adapterFingerprint &&
        lhs.runtimeDeviceProgramFingerprint ==
            rhs.runtimeDeviceProgramFingerprint &&
        lhs.environmentCount == rhs.environmentCount &&
        lhs.endpointCount == rhs.endpointCount &&
        lhs.femNodeCount == rhs.femNodeCount &&
        lhs.contactSampleCount == rhs.contactSampleCount &&
        lhs.femBodyContactSampleCount == rhs.femBodyContactSampleCount &&
        lhs.articularContactSampleCount == rhs.articularContactSampleCount &&
        lhs.articularContactPairCount == rhs.articularContactPairCount &&
        lhs.passiveLigamentCount == rhs.passiveLigamentCount &&
        lhs.passiveRoutedBandCount == rhs.passiveRoutedBandCount &&
        lhs.boundBodyPoseStride == rhs.boundBodyPoseStride &&
        lhs.encodedPassCount == rhs.encodedPassCount &&
        lhs.abortCount == rhs.abortCount &&
        lhs.articularAttemptedStepCount ==
            rhs.articularAttemptedStepCount &&
        sameVectorBytes(
            lhs.anchorReactionAcceptedHistory,
            rhs.anchorReactionAcceptedHistory) &&
        sameVectorBytes(
            lhs.femBodyContactAcceptedHistory,
            rhs.femBodyContactAcceptedHistory) &&
        sameVectorBytes(
            lhs.articularContactAcceptedHistory,
            rhs.articularContactAcceptedHistory) &&
        sameVectorBytes(
            lhs.articularContactPairForceAcceptedHistory,
            rhs.articularContactPairForceAcceptedHistory) &&
        sameVectorBytes(
            lhs.passiveLigamentAcceptedState,
            rhs.passiveLigamentAcceptedState) &&
        sameVectorBytes(
            lhs.passiveRoutedBandAcceptedState,
            rhs.passiveRoutedBandAcceptedState) &&
        lhs.authoritySHA256 == rhs.authoritySHA256;
}

bool evaluateNumiHumanPassiveLigamentFiber(
    const NMNumiHumanPassiveLigamentGPU& ligament,
    const double currentCentroidLengthMeters,
    NumiHumanPassiveLigamentFiberEvaluation& result
) noexcept {
    result = {};
    const nm_float4 material = ligament.material;
    const nm_float4 reference = ligament.reference;
    if (ligament.flags != NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE ||
        ligament.reserved0 != 0u || !finiteScale(material) ||
        !finiteScale(reference) || material.x <= 0.0f ||
        material.y <= 0.0f || material.z <= 0.0f ||
        material.w <= 1.0f || reference.x <= 0.0f ||
        reference.y <= 0.0f || reference.z < 1.0f ||
        reference.w != 0.0f ||
        !std::isfinite(currentCentroidLengthMeters) ||
        currentCentroidLengthMeters <= 0.0) {
        return false;
    }
    const double stretch = reference.z *
        currentCentroidLengthMeters / reference.x;
    if (!std::isfinite(stretch) || stretch <= 0.0) return false;
    double stress = 0.0;
    if (stretch > 1.0 && stretch < material.w) {
        stress = material.x * std::expm1(material.y * (stretch - 1.0));
    } else if (stretch >= material.w) {
        const double c6 = material.x *
                std::expm1(material.y * (material.w - 1.0)) -
            material.z * material.w;
        stress = material.z * stretch + c6;
    }
    const double tension = stress * reference.y;
    if (!std::isfinite(stress) || stress < 0.0 ||
        !std::isfinite(tension) || tension < 0.0) return false;
    result.effectiveStretch = stretch;
    result.fiberStressPascals = stress;
    result.tensionNewtons = tension;
    return true;
}

bool evaluateNumiHumanPassiveRoutedBand(
    const NMNumiHumanPassiveRoutedBandGPU& band,
    const double geometricLengthMeters,
    const double signedWindingRadians,
    NumiHumanPassiveRoutedBandEvaluation& result
) noexcept {
    result = {};
    const nm_float4 axisRadius = band.pulleyLocalAxisAndRadius;
    const nm_float4 neutral = band.neutralRelativeOrientation;
    const nm_float4 material = band.material;
    const nm_float4 reference = band.reference;
    const double axisLength = std::sqrt(
        static_cast<double>(axisRadius.x) * axisRadius.x +
        static_cast<double>(axisRadius.y) * axisRadius.y +
        static_cast<double>(axisRadius.z) * axisRadius.z);
    const double neutralLength = std::sqrt(
        static_cast<double>(neutral.x) * neutral.x +
        static_cast<double>(neutral.y) * neutral.y +
        static_cast<double>(neutral.z) * neutral.z +
        static_cast<double>(neutral.w) * neutral.w);
    if (band.flags != NM_NUMI_HUMAN_PASSIVE_ROUTED_BAND_ACTIVE ||
        band.originBodyIndex == NM_INVALID_INDEX ||
        band.pulleyBodyIndex == NM_INVALID_INDEX ||
        band.insertionBodyIndex == NM_INVALID_INDEX ||
        !finiteScale(band.originLocalPoint) ||
        !finiteScale(band.pulleyLocalPoint) ||
        !finiteScale(band.insertionLocalPoint) ||
        band.originLocalPoint.w != 0.0f ||
        band.pulleyLocalPoint.w != 0.0f ||
        band.insertionLocalPoint.w != 0.0f ||
        !finiteScale(axisRadius) || !finiteScale(neutral) ||
        !finiteScale(material) || !finiteScale(reference) ||
        std::abs(axisLength - 1.0) > 1.0e-5 || axisRadius.w <= 0.0f ||
        std::abs(neutralLength - 1.0) > 1.0e-5 ||
        material.x <= 0.0f || material.y <= 0.0f ||
        material.z <= 0.0f || material.w < 0.0f || material.w >= 0.5f ||
        reference.x <= 0.0f || reference.y <= 0.0f ||
        reference.z <= 0.0f || reference.w != 0.0f ||
        !std::isfinite(geometricLengthMeters) ||
        geometricLengthMeters <= 0.0 ||
        !std::isfinite(signedWindingRadians)) {
        return false;
    }
    const double routeLength = geometricLengthMeters +
        axisRadius.w * std::abs(signedWindingRadians);
    const double stretch = routeLength / reference.x;
    const double strain = stretch - 1.0;
    if (!std::isfinite(routeLength) || !std::isfinite(stretch) ||
        stretch <= 0.0 || !std::isfinite(strain) ||
        strain > reference.z) return false;

    const auto tensionAtStretch = [&](const double lambda) noexcept {
        if (lambda <= 1.0) return 0.0;
        const double lambda2 = lambda * lambda;
        const double stress =
            static_cast<double>(material.x) *
                (lambda2 - 1.0 / lambda) +
            static_cast<double>(material.y) /
                (2.0 * static_cast<double>(material.z)) *
                std::expm1(static_cast<double>(material.z) *
                    (lambda2 - 1.0)) * lambda2;
        const double area = static_cast<double>(reference.y) *
            std::pow(lambda, -2.0 * static_cast<double>(material.w));
        return stress * area;
    };
    const double tension = tensionAtStretch(stretch);
    if (!std::isfinite(tension) || tension < 0.0) return false;

    double energy = 0.0;
    if (stretch > 1.0) {
        constexpr std::uint32_t intervals = 32u;
        const double h = (stretch - 1.0) / intervals;
        double integral = tensionAtStretch(1.0) + tension;
        for (std::uint32_t index = 1u; index < intervals; ++index) {
            integral += (index & 1u ? 4.0 : 2.0) *
                tensionAtStretch(1.0 + h * index);
        }
        energy = reference.x * h * integral / 3.0;
    }
    if (!std::isfinite(energy) || energy < 0.0) return false;
    result.routeLengthMeters = routeLength;
    result.stretch = stretch;
    result.strain = strain;
    result.tensionNewtons = tension;
    result.storedEnergyJoules = energy;
    return true;
}

NumiHumanFEMPrestressDiagnostics prepareNumiHumanFEMPrestressStage(
    const CompiledWorld& world,
    const std::span<const NumiHumanFEMPrestressTarget> targets,
    const float fraction,
    RuntimeStateSnapshot& snapshot
) {
    NumiHumanFEMPrestressDiagnostics diagnostics;
    diagnostics.fraction = fraction;
    const auto reject = [&](const NumiHumanFEMPrestressStatus status,
                            const std::uint32_t target,
                            std::string message) {
        diagnostics.status = status;
        diagnostics.failingTarget = target;
        diagnostics.message = std::move(message);
        return diagnostics;
    };
    if (!snapshot.available || snapshot.deviceProgramFingerprint == 0u) {
        return reject(
            NumiHumanFEMPrestressStatus::invalidSnapshot,
            NM_INVALID_INDEX,
            "prestress staging requires an available device snapshot"
        );
    }
    if (!std::isfinite(fraction) || fraction < 0.0f || fraction > 1.0f) {
        return reject(
            NumiHumanFEMPrestressStatus::invalidFraction,
            NM_INVALID_INDEX,
            "prestress fraction must be finite and within [0, 1]"
        );
    }
    const std::size_t parameterCount = world.dispatch.parameterCount;
    const std::size_t environmentCount = world.dispatch.environmentCount;
    if (parameterCount == 0u || environmentCount == 0u ||
        world.parameters.size() != parameterCount ||
        snapshot.environmentParameters.size() !=
            parameterCount * environmentCount) {
        return reject(
            NumiHumanFEMPrestressStatus::invalidParameterArena,
            NM_INVALID_INDEX,
            "prestress parameter arena does not match the cooked world"
        );
    }
    if (targets.empty()) {
        return reject(
            NumiHumanFEMPrestressStatus::invalidParameter,
            NM_INVALID_INDEX,
            "prestress staging requires at least one parameter target"
        );
    }

    std::vector<bool> parameterSeen(parameterCount, false);
    std::vector<std::pair<std::uint32_t, float>> stagedValues;
    stagedValues.reserve(targets.size());
    for (std::uint32_t targetIndex = 0u;
         targetIndex < targets.size(); ++targetIndex) {
        const auto& target = targets[targetIndex];
        if (target.materialIndex >= world.materials.size()) {
            return reject(
                NumiHumanFEMPrestressStatus::invalidMaterial,
                targetIndex,
                "prestress target material index escapes the cooked world"
            );
        }
        const NMMaterialGPU& material = world.materials[target.materialIndex];
        if (target.localParameterIndex >= material.parameterCount ||
            material.parameterOffset >= parameterCount ||
            target.localParameterIndex >=
                parameterCount - material.parameterOffset) {
            return reject(
                NumiHumanFEMPrestressStatus::invalidParameter,
                targetIndex,
                "prestress target parameter index escapes its material"
            );
        }
        const std::uint32_t parameter =
            material.parameterOffset + target.localParameterIndex;
        if (parameterSeen[parameter]) {
            return reject(
                NumiHumanFEMPrestressStatus::duplicateParameter,
                targetIndex,
                "prestress target parameter is duplicated"
            );
        }
        parameterSeen[parameter] = true;
        const auto& bounds = world.parameters[parameter].valueAndBounds;
        const float value = target.neutralValue +
            fraction * (target.sourceValue - target.neutralValue);
        if (!std::isfinite(target.neutralValue) ||
            !std::isfinite(target.sourceValue) || !std::isfinite(value) ||
            target.neutralValue < bounds.y || target.neutralValue > bounds.z ||
            target.sourceValue < bounds.y || target.sourceValue > bounds.z ||
            value < bounds.y || value > bounds.z) {
            return reject(
                NumiHumanFEMPrestressStatus::invalidBounds,
                targetIndex,
                "prestress target or staged value violates authored bounds"
            );
        }
        diagnostics.maximumAbsoluteParameterDelta = std::max(
            diagnostics.maximumAbsoluteParameterDelta,
            std::abs(value - target.neutralValue)
        );
        stagedValues.emplace_back(parameter, value);
    }

    std::vector<float> staged = snapshot.environmentParameters;
    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        const std::size_t base = environment * parameterCount;
        for (const auto& [parameter, value] : stagedValues)
            staged[base + parameter] = value;
    }
    snapshot.environmentParameters = std::move(staged);
    diagnostics.status = NumiHumanFEMPrestressStatus::success;
    diagnostics.appliedParameterCount = static_cast<std::uint32_t>(
        stagedValues.size() * environmentCount);
    diagnostics.message = "prestress parameter stage prepared";
    return diagnostics;
}

struct NumiHumanTendonFEMLoadAdapter::State {
    Runtime* runtime = nullptr;
    std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads;
    std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors;
    std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU> replacements;
    std::vector<NMNumiHumanFEMContactSampleGPU> contactSamples;
    std::vector<NMNumiHumanFEMContactContributionGPU> contactContributions;
    std::vector<NMIncidenceRangeGPU> contactRanges;
    std::vector<NMNumiHumanFEMBodyContactSampleGPU> femBodyContactSamples;
    std::vector<NMNumiHumanArticularContactSampleGPU> articularContactSamples;
    std::vector<NMIncidenceRangeGPU> articularContactPairRanges;
    std::vector<NMNumiHumanPassiveLigamentGPU> passiveLigaments;
    std::vector<NMNumiHumanPassiveRoutedBandGPU> passiveRoutedBands;
    std::filesystem::path metallib;
    NumiHumanTendonFEMLoadExecutionMode executionMode =
        NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime;
    std::uint32_t endpointCount = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t activeAnchorCount = 0u;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t preSourceCorrectionEncodeCount = 0u;
    std::uint32_t deferredExternalForceBorrowCount = 0u;
    std::uint32_t runtimePreDynamicsEncodeCount = 0u;
    std::uint32_t runtimePostCommitEncodeCount = 0u;
    std::uint32_t articularBodyPoseStride = 0u;
    // Allocation cache retained across accepted-state restore. The bound
    // stride above is authoritative; this value only describes private
    // scratch capacity that may be reused after restoring a pre-pass state.
    std::uint32_t allocatedArticularBodyPoseStride = 0u;
    std::uint32_t articularMechanicalSampleCount = 0u;
    std::uint32_t articularInternalSameBodySampleCount = 0u;
    std::uint32_t articularContactPairCount = 0u;
    std::uint32_t articularAttemptedStepCount = 0u;
    std::uint64_t fingerprint = 0u;
    std::uint64_t runtimeDeviceProgramFingerprint = 0u;
    std::uintptr_t deferredCommandBufferIdentity = 0u;
    std::uint32_t deferredStepIndex = 0u;
    bool deferredExternalForcesAvailable = false;
    std::string message;

    // Host shadows make an accepted snapshot restorable before the lazy Metal
    // buffers have been materialized by a borrowed pass. Once materialized,
    // snapshot() reads the authoritative shared GPU buffers directly.
    std::vector<nm_float4> anchorReactionAcceptedHistory;
    std::vector<NMNumiHumanArticularContactAuditGPU>
        femBodyContactAcceptedHistory;
    std::vector<NMNumiHumanArticularContactAuditGPU>
        articularContactAcceptedHistory;
    std::vector<float> articularContactPairForceAcceptedHistory;
    std::vector<NMNumiHumanPassiveLigamentAuditGPU>
        passiveLigamentAcceptedState;
    std::vector<NMNumiHumanPassiveRoutedBandAuditGPU>
        passiveRoutedBandAcceptedState;

    __strong id<MTLDevice> device = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> provisionalStatusPipeline = nil;
    __strong id<MTLComputePipelineState> statusPipeline = nil;
    __strong id<MTLComputePipelineState> forcePipeline = nil;
    __strong id<MTLComputePipelineState> forceAuditPipeline = nil;
    __strong id<MTLComputePipelineState> contactForcePipeline = nil;
    __strong id<MTLComputePipelineState> femBodyContactForcePipeline = nil;
    __strong id<MTLComputePipelineState> femBodyContactWrenchPipeline = nil;
    __strong id<MTLComputePipelineState> femBodyContactAuditPipeline = nil;
    __strong id<MTLComputePipelineState> articularContactPipeline = nil;
    __strong id<MTLComputePipelineState> articularContactAuditPipeline = nil;
    __strong id<MTLComputePipelineState> articularContactAuditCommitPipeline = nil;
    __strong id<MTLComputePipelineState> articularContactPairAuditPipeline = nil;
    __strong id<MTLComputePipelineState> articularContactPairAuditCommitPipeline = nil;
    __strong id<MTLComputePipelineState> passiveLigamentAuditPipeline = nil;
    __strong id<MTLComputePipelineState> passiveLigamentAuditCommitPipeline = nil;
    __strong id<MTLComputePipelineState> passiveRoutedBandAuditPipeline = nil;
    __strong id<MTLComputePipelineState> passiveRoutedBandAuditCommitPipeline = nil;
    __strong id<MTLComputePipelineState> targetPipeline = nil;
    __strong id<MTLComputePipelineState> reactionAuditPipeline = nil;
    __strong id<MTLComputePipelineState> reactionAuditCommitPipeline = nil;
    __strong id<MTLComputePipelineState> reactionPipeline = nil;
    __strong id<MTLComputePipelineState> preSourceCorrectionPipeline = nil;
    __strong id<MTLBuffer> nodeLoadBuffer = nil;
    __strong id<MTLBuffer> nodeAnchorBuffer = nil;
    __strong id<MTLBuffer> replacementBuffer = nil;
    __strong id<MTLBuffer> contactSampleBuffer = nil;
    __strong id<MTLBuffer> contactContributionBuffer = nil;
    __strong id<MTLBuffer> contactRangeBuffer = nil;
    __strong id<MTLBuffer> femBodyContactSampleBuffer = nil;
    __strong id<MTLBuffer> femBodyContactWrenchBuffer = nil;
    __strong id<MTLBuffer> femBodyContactAuditBuffer = nil;
    __strong id<MTLBuffer> femBodyContactAuditHistoryBuffer = nil;
    __strong id<MTLBuffer> articularContactSampleBuffer = nil;
    __strong id<MTLBuffer> articularContactPairRangeBuffer = nil;
    __strong id<MTLBuffer> articularBodyWrenchBuffer = nil;
    __strong id<MTLBuffer> articularContactAuditBuffer = nil;
    __strong id<MTLBuffer> articularContactAuditHistoryBuffer = nil;
    __strong id<MTLBuffer> articularContactPairAuditBuffer = nil;
    __strong id<MTLBuffer> articularContactPairAuditHistoryBuffer = nil;
    __strong id<MTLBuffer> passiveLigamentBuffer = nil;
    __strong id<MTLBuffer> passiveLigamentAuditBuffer = nil;
    __strong id<MTLBuffer> passiveRoutedBandBuffer = nil;
    __strong id<MTLBuffer> passiveRoutedBandAuditBuffer = nil;
    __strong id<MTLBuffer> externalForceBuffer = nil;
    __strong id<MTLBuffer> externalForceAuditBuffer = nil;
    __strong id<MTLBuffer> anchorReactionAuditBuffer = nil;
    __strong id<MTLBuffer> anchorReactionAuditHistoryBuffer = nil;
    __strong id<MTLBuffer> kinematicTargetBuffer = nil;
    __strong id<MTLBuffer> worldStatusBuffer = nil;
};

NumiHumanTendonFEMLoadAdapter::NumiHumanTendonFEMLoadAdapter() = default;
NumiHumanTendonFEMLoadAdapter::~NumiHumanTendonFEMLoadAdapter() = default;
NumiHumanTendonFEMLoadAdapter::NumiHumanTendonFEMLoadAdapter(
    NumiHumanTendonFEMLoadAdapter&&
) noexcept = default;
NumiHumanTendonFEMLoadAdapter& NumiHumanTendonFEMLoadAdapter::operator=(
    NumiHumanTendonFEMLoadAdapter&&
) noexcept = default;

bool NumiHumanTendonFEMLoadAdapter::initialize(
    Runtime& runtime,
    const NumiHumanTendonFEMLoadSource& source,
    const NumiHumanTendonFEMLoadConfiguration& configuration
) {
    const bool passiveAttachmentOnly = source.endpointReplacements.empty();
    const bool hasContact = !source.contactSamples.empty();
    const bool standalone = configuration.executionMode ==
        NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime;
    const bool deferred = configuration.executionMode ==
        NumiHumanTendonFEMLoadExecutionMode::numanXDeferred;
    if (!runtime.valid() || source.nodeLoads.empty() ||
        source.nodeAnchors.size() != source.nodeLoads.size() ||
        source.nodeLoads.size() != runtime.femNodeCount() ||
        source.environmentCount != runtime.environmentCount() ||
        source.endpointCount == 0u || source.environmentCount == 0u ||
        !std::isfinite(source.productionForceOwnerFraction) ||
        (passiveAttachmentOnly
            ? source.productionForceOwnerFraction != 0.0f
            : !(source.productionForceOwnerFraction > 0.0f)) ||
        source.productionForceOwnerFraction > 1.0f ||
        configuration.metallib.empty() ||
        !std::filesystem::is_regular_file(configuration.metallib) ||
        (!standalone && !deferred) || configuration.reserved0 != 0u ||
        source.articularContactSamples.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        source.articularContactPairRanges.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        source.femBodyContactSamples.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        source.passiveLigaments.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        source.passiveRoutedBands.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        (hasContact != !source.contactContributions.empty()) ||
        (hasContact != !source.contactRanges.empty()) ||
        (hasContact &&
            (source.contactRanges.size() != source.nodeLoads.size() ||
             source.contactSamples.size() >
                 std::numeric_limits<std::uint32_t>::max() ||
             source.contactContributions.size() >
                 std::numeric_limits<std::uint32_t>::max() ||
             source.contactSamples.size() >
                 source.contactContributions.size() / 4u))) {
        return false;
    }
    if (hasContact) {
        std::vector<std::array<std::uint32_t, 4u>> roleCounts(
            source.contactSamples.size());
        std::uint64_t nextContribution = 0u;
        for (std::uint32_t node = 0u;
             node < source.contactRanges.size(); ++node) {
            const NMIncidenceRangeGPU range = source.contactRanges[node];
            if (range.first != nextContribution ||
                static_cast<std::uint64_t>(range.first) + range.count >
                    source.contactContributions.size()) return false;
            for (std::uint32_t local = 0u; local < range.count; ++local) {
                const auto& contribution = source.contactContributions[
                    range.first + local];
                if (contribution.sampleIndex >= source.contactSamples.size() ||
                    contribution.role > 3u || contribution.reserved0 != 0u ||
                    contribution.reserved1 != 0u) return false;
                const auto& sample =
                    source.contactSamples[contribution.sampleIndex];
                std::uint32_t owner = sample.slaveNode;
                switch (contribution.role) {
                    case 1u: owner = sample.masterNode0; break;
                    case 2u: owner = sample.masterNode1; break;
                    case 3u: owner = sample.masterNode2; break;
                    default: break;
                }
                if (owner != node ||
                    ++roleCounts[contribution.sampleIndex][contribution.role] !=
                        1u) return false;
            }
            nextContribution += range.count;
        }
        if (nextContribution != source.contactContributions.size() ||
            source.contactContributions.size() !=
                4u * source.contactSamples.size()) return false;
        for (std::uint32_t index = 0u;
             index < source.contactSamples.size(); ++index) {
            const auto& sample = source.contactSamples[index];
            const auto& roles = roleCounts[index];
            const nm_float4 bary =
                sample.barycentricAndReferenceSeparation;
            const nm_float4 normal = sample.normalAndArea;
            const nm_float4 stiffness = sample.stiffness;
            const double normalLength = std::sqrt(
                normal.x * normal.x + normal.y * normal.y +
                normal.z * normal.z);
            if (sample.slaveNode >= source.nodeLoads.size() ||
                sample.masterNode0 >= source.nodeLoads.size() ||
                sample.masterNode1 >= source.nodeLoads.size() ||
                sample.masterNode2 >= source.nodeLoads.size() ||
                sample.slaveNode == sample.masterNode0 ||
                sample.slaveNode == sample.masterNode1 ||
                sample.slaveNode == sample.masterNode2 ||
                sample.masterNode0 == sample.masterNode1 ||
                sample.masterNode0 == sample.masterNode2 ||
                sample.masterNode1 == sample.masterNode2 ||
                !finiteScale(bary) || bary.x < 0.0f || bary.y < 0.0f ||
                bary.z < 0.0f ||
                std::abs(bary.x + bary.y + bary.z - 1.0f) > 1.0e-5f ||
                !finiteScale(normal) ||
                std::abs(normalLength - 1.0) > 1.0e-5 ||
                normal.w <= 0.0f || !finiteScale(stiffness) ||
                stiffness.x <= 0.0f || stiffness.y != 0.0f ||
                stiffness.z != 0.0f || stiffness.w != 0.0f ||
                std::any_of(roles.begin(), roles.end(),
                            [](const std::uint32_t count) {
                                return count != 1u;
                            })) return false;
        }
    }
    {
        std::vector<bool> contactedNode(source.nodeLoads.size(), false);
        for (const auto& sample : source.femBodyContactSamples) {
            const nm_float4 point = sample.bodyLocalPointAndArea;
            const nm_float4 normal =
                sample.bodyLocalNormalAndReferenceSeparation;
            const nm_float4 stiffness =
                sample.stiffnessAndNormalStrainPerPressure;
            const double normalLength = std::sqrt(
                normal.x * normal.x + normal.y * normal.y +
                normal.z * normal.z);
            if (sample.slaveNode >= source.nodeLoads.size() ||
                sample.bodyIndex == NM_INVALID_INDEX ||
                sample.flags != NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE ||
                sample.reserved0 != 0u || !finiteScale(point) ||
                point.w <= 0.0f || !finiteScale(normal) ||
                std::abs(normalLength - 1.0) > 1.0e-5 || normal.w < 0.0f ||
                !finiteScale(stiffness) || stiffness.x <= 0.0f ||
                stiffness.y <= 0.0f || stiffness.z != 0.0f ||
                stiffness.w != 0.0f || contactedNode[sample.slaveNode] ||
                (source.nodeAnchors[sample.slaveNode].flags &
                    NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) != 0u) {
                return false;
            }
            contactedNode[sample.slaveNode] = true;
        }
    }
    if (!source.articularContactPairRanges.empty()) {
        if (source.articularContactSamples.empty()) return false;
        std::uint64_t nextSample = 0u;
        for (const auto& range : source.articularContactPairRanges) {
            if (range.count == 0u || range.first != nextSample ||
                static_cast<std::uint64_t>(range.first) + range.count >
                    source.articularContactSamples.size()) {
                return false;
            }
            nextSample += range.count;
        }
        if (nextSample != source.articularContactSamples.size()) return false;
    }
    for (const auto& sample : source.articularContactSamples) {
        const nm_float4 slave = sample.slaveLocalPointAndArea;
        const nm_float4 triangle0 =
            sample.masterLocalTriangle0AndReferenceSeparation;
        const nm_float4 triangle1 =
            sample.masterLocalTriangle1AndStiffness;
        const nm_float4 triangle2 =
            sample.masterLocalTriangle2AndNormalStrainPerPressure;
        const nm_float4 referenceNormal =
            sample.masterLocalReferenceNormalAndReserved;
        const std::array<nm_float4, 3u> adjacentOpposite{{
            sample.masterLocalAdjacentOpposite0AndActive,
            sample.masterLocalAdjacentOpposite1AndActive,
            sample.masterLocalAdjacentOpposite2AndActive,
        }};
        const double edge10x = triangle1.x - triangle0.x;
        const double edge10y = triangle1.y - triangle0.y;
        const double edge10z = triangle1.z - triangle0.z;
        const double edge20x = triangle2.x - triangle0.x;
        const double edge20y = triangle2.y - triangle0.y;
        const double edge20z = triangle2.z - triangle0.z;
        const double normalX = edge10y * edge20z - edge10z * edge20y;
        const double normalY = edge10z * edge20x - edge10x * edge20z;
        const double normalZ = edge10x * edge20y - edge10y * edge20x;
        const double squaredNormalLength = normalX * normalX +
            normalY * normalY + normalZ * normalZ;
        const double referenceNormalLength = std::sqrt(
            referenceNormal.x * referenceNormal.x +
            referenceNormal.y * referenceNormal.y +
            referenceNormal.z * referenceNormal.z);
        const bool sameBody =
            sample.slaveBodyIndex == sample.masterBodyIndex;
        const std::uint32_t expectedFlags = sameBody
            ? NM_NUMI_HUMAN_ARTICULAR_CONTACT_INTERNAL_SAME_BODY
            : NM_NUMI_HUMAN_ARTICULAR_CONTACT_ACTIVE;
        bool adjacencyValid = true;
        const std::array<nm_float4, 3u> triangle{{
            triangle0, triangle1, triangle2}};
        for (std::uint32_t edge = 0u; edge < 3u; ++edge) {
            const nm_float4 opposite = adjacentOpposite[edge];
            if (!finiteScale(opposite) ||
                (opposite.w != 0.0f && opposite.w != 1.0f)) {
                adjacencyValid = false;
                break;
            }
            if (opposite.w == 0.0f) {
                adjacencyValid = opposite.x == 0.0f &&
                    opposite.y == 0.0f && opposite.z == 0.0f;
                if (!adjacencyValid) break;
                continue;
            }
            const nm_float4 first = triangle[edge];
            const nm_float4 second = triangle[(edge + 1u) % 3u];
            const double firstX = second.x - first.x;
            const double firstY = second.y - first.y;
            const double firstZ = second.z - first.z;
            const double secondX = opposite.x - first.x;
            const double secondY = opposite.y - first.y;
            const double secondZ = opposite.z - first.z;
            const double crossX = firstY * secondZ - firstZ * secondY;
            const double crossY = firstZ * secondX - firstX * secondZ;
            const double crossZ = firstX * secondY - firstY * secondX;
            const double squaredLength = crossX * crossX + crossY * crossY +
                crossZ * crossZ;
            adjacencyValid = std::isfinite(squaredLength) &&
                squaredLength > 1.0e-20;
            if (!adjacencyValid) break;
        }
        if (sample.slaveBodyIndex == NM_INVALID_INDEX ||
            sample.masterBodyIndex == NM_INVALID_INDEX ||
            sample.flags != expectedFlags ||
            sample.reserved0 != 0u || !finiteScale(slave) ||
            !finiteScale(triangle0) || !finiteScale(triangle1) ||
            !finiteScale(triangle2) || !finiteScale(referenceNormal) ||
            slave.w <= 0.0f ||
            !(std::isfinite(squaredNormalLength) &&
              squaredNormalLength > 1.0e-20) ||
            triangle1.w <= 0.0f || triangle2.w <= 0.0f ||
            std::abs(referenceNormalLength - 1.0) > 1.0e-5 ||
            referenceNormal.w != 0.0f || !adjacencyValid) {
            return false;
        }
    }
    for (const auto& ligament : source.passiveLigaments) {
        NumiHumanPassiveLigamentFiberEvaluation referenceEvaluation;
        if (ligament.firstBodyIndex == NM_INVALID_INDEX ||
            ligament.secondBodyIndex == NM_INVALID_INDEX ||
            ligament.firstBodyIndex == ligament.secondBodyIndex ||
            ligament.flags != NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE ||
            ligament.reserved0 != 0u ||
            !finiteScale(ligament.firstLocalPoint) ||
            !finiteScale(ligament.secondLocalPoint) ||
            ligament.firstLocalPoint.w != 0.0f ||
            ligament.secondLocalPoint.w != 0.0f ||
            !evaluateNumiHumanPassiveLigamentFiber(
                ligament, ligament.reference.x, referenceEvaluation)) {
            return false;
        }
    }
    for (const auto& band : source.passiveRoutedBands) {
        NumiHumanPassiveRoutedBandEvaluation referenceEvaluation;
        if (band.pulleyBodyIndex == band.insertionBodyIndex ||
            !evaluateNumiHumanPassiveRoutedBand(
                band, band.reference.x, 0.0, referenceEvaluation)) {
            return false;
        }
    }
    std::vector<double> endpointSignedScales(source.endpointCount, 0.0);
    std::vector<double> endpointAbsoluteScales(source.endpointCount, 0.0);
    std::uint32_t anchorCount = 0u;
    for (const NMNumiHumanTendonFEMNodeLoadGPU& load : source.nodeLoads) {
        if (!finiteScale(load.scale)) return false;
        for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
            const std::uint32_t endpoint = load.endpointIndex[slot];
            const float scale = scaleComponent(load.scale, slot);
            if ((endpoint == NM_INVALID_INDEX) != (scale == 0.0f) ||
                (endpoint != NM_INVALID_INDEX && endpoint >= source.endpointCount)) {
                return false;
            }
            for (std::uint32_t previous = 0u; previous < slot; ++previous) {
                if (endpoint != NM_INVALID_INDEX &&
                    load.endpointIndex[previous] == endpoint) {
                    return false;
                }
            }
            if (endpoint != NM_INVALID_INDEX) {
                endpointSignedScales[endpoint] += scale;
                endpointAbsoluteScales[endpoint] += std::abs(scale);
            }
        }
    }
    for (const NMNumiHumanTendonFEMNodeAnchorGPU& anchor :
         source.nodeAnchors) {
        if (!finiteScale(anchor.localPoint) || anchor.reserved0 != 0u ||
            anchor.reserved1 != 0u ||
            (anchor.flags &
                ~NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) != 0u ||
            anchor.localPoint.w != 0.0f) {
            return false;
        }
        if ((anchor.flags &
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) == 0u) {
            if (anchor.bodyIndex != NM_INVALID_INDEX ||
                anchor.localPoint.x != 0.0f || anchor.localPoint.y != 0.0f ||
                anchor.localPoint.z != 0.0f) {
                return false;
            }
        } else {
            if (anchor.bodyIndex == NM_INVALID_INDEX) return false;
            ++anchorCount;
        }
    }
    std::vector<bool> loadEndpoints(source.endpointCount, false);
    std::vector<bool> anchorEndpoints(source.endpointCount, false);
    std::vector<bool> expectedLoadedEndpoints(source.endpointCount, false);
    for (const NMNumiHumanTendonFEMEndpointReplacementGPU& replacement :
         source.endpointReplacements) {
        constexpr std::uint32_t replacementFlagMask =
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE |
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW |
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE;
        const bool fullMuscleRow = (replacement.flags &
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW) != 0u;
        const bool distalForceCouple = (replacement.flags &
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE) != 0u;
        if ((replacement.flags &
                NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE) == 0u ||
            (replacement.flags & ~replacementFlagMask) != 0u ||
            (distalForceCouple && !fullMuscleRow) ||
            replacement.reserved0 != 0u ||
            replacement.loadEndpointIndex >= source.endpointCount ||
            replacement.anchorEndpointIndex >= source.endpointCount ||
            replacement.loadEndpointIndex == replacement.anchorEndpointIndex ||
            !finiteScale(replacement.forceOwnerFraction) ||
            replacement.forceOwnerFraction.x <= 0.0f ||
            replacement.forceOwnerFraction.x > 1.0f ||
            replacement.forceOwnerFraction.y != 0.0f ||
            replacement.forceOwnerFraction.z != 0.0f ||
            replacement.forceOwnerFraction.w != 0.0f ||
            std::abs(replacement.forceOwnerFraction.x -
                source.productionForceOwnerFraction) > 1.0e-6f ||
            loadEndpoints[replacement.loadEndpointIndex] ||
            anchorEndpoints[replacement.anchorEndpointIndex]) {
            return false;
        }
        loadEndpoints[replacement.loadEndpointIndex] = true;
        anchorEndpoints[replacement.anchorEndpointIndex] = true;
        expectedLoadedEndpoints[replacement.loadEndpointIndex] = true;
        if (distalForceCouple) {
            expectedLoadedEndpoints[replacement.anchorEndpointIndex] = true;
        }
        const double owner = replacement.forceOwnerFraction.x;
        if (std::abs(
                endpointSignedScales[replacement.loadEndpointIndex] - owner) >
                1.0e-6 ||
            std::abs(
                endpointAbsoluteScales[replacement.loadEndpointIndex] - owner) >
                1.0e-6 ||
            std::abs(endpointSignedScales[
                replacement.anchorEndpointIndex]) > 1.0e-6 ||
            std::abs(endpointAbsoluteScales[
                replacement.anchorEndpointIndex] -
                (distalForceCouple ? 2.0 * owner : 0.0)) > 1.0e-6) {
            return false;
        }
    }
    if (anchorCount == 0u ||
        (deferred && runtime.femHumanAttachmentCount() != anchorCount) ||
        std::any_of(
            endpointAbsoluteScales.begin(), endpointAbsoluteScales.end(),
            [](const double scale) {
                return !std::isfinite(scale) || scale > 2.000001;
            }
        ) || (passiveAttachmentOnly
            ? std::any_of(
                endpointAbsoluteScales.begin(), endpointAbsoluteScales.end(),
                [](const double scale) { return scale != 0.0; })
            : std::none_of(
                endpointAbsoluteScales.begin(), endpointAbsoluteScales.end(),
                [](const double scale) { return scale > 0.0; }))) {
        return false;
    }
    for (std::size_t endpointIndex = 0u;
         endpointIndex < endpointAbsoluteScales.size();
         ++endpointIndex) {
        if ((endpointAbsoluteScales[endpointIndex] > 0.0) !=
            expectedLoadedEndpoints[endpointIndex]) {
            return false;
        }
    }
    auto candidate = std::make_unique<State>();
    candidate->runtime = &runtime;
    candidate->nodeLoads.assign(source.nodeLoads.begin(), source.nodeLoads.end());
    candidate->nodeAnchors.assign(
        source.nodeAnchors.begin(), source.nodeAnchors.end()
    );
    candidate->replacements.assign(
        source.endpointReplacements.begin(), source.endpointReplacements.end()
    );
    candidate->contactSamples.assign(
        source.contactSamples.begin(), source.contactSamples.end());
    candidate->contactContributions.assign(
        source.contactContributions.begin(),
        source.contactContributions.end());
    candidate->contactRanges.assign(
        source.contactRanges.begin(), source.contactRanges.end());
    candidate->femBodyContactSamples.assign(
        source.femBodyContactSamples.begin(),
        source.femBodyContactSamples.end());
    candidate->articularContactSamples.assign(
        source.articularContactSamples.begin(),
        source.articularContactSamples.end());
    candidate->articularContactPairRanges.assign(
        source.articularContactPairRanges.begin(),
        source.articularContactPairRanges.end());
    candidate->passiveLigaments.assign(
        source.passiveLigaments.begin(), source.passiveLigaments.end());
    candidate->passiveRoutedBands.assign(
        source.passiveRoutedBands.begin(), source.passiveRoutedBands.end());
    for (const auto& sample : candidate->articularContactSamples) {
        if (sample.flags == NM_NUMI_HUMAN_ARTICULAR_CONTACT_ACTIVE) {
            ++candidate->articularMechanicalSampleCount;
        } else {
            ++candidate->articularInternalSameBodySampleCount;
        }
    }
    candidate->metallib = configuration.metallib;
    candidate->executionMode = configuration.executionMode;
    candidate->endpointCount = source.endpointCount;
    candidate->environmentCount = source.environmentCount;
    candidate->activeAnchorCount = anchorCount;
    candidate->runtimeDeviceProgramFingerprint =
        runtime.deviceProgramFingerprint();
    candidate->articularContactPairCount = static_cast<std::uint32_t>(
        candidate->articularContactPairRanges.size());
    const std::size_t acceptedHistoryElements =
        static_cast<std::size_t>(candidate->environmentCount) *
        NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS;
    candidate->anchorReactionAcceptedHistory.resize(
        acceptedHistoryElements);
    if (!candidate->femBodyContactSamples.empty()) {
        candidate->femBodyContactAcceptedHistory.resize(
            acceptedHistoryElements);
    }
    if (!candidate->articularContactSamples.empty()) {
        candidate->articularContactAcceptedHistory.resize(
            acceptedHistoryElements);
    }
    if (candidate->articularContactPairCount != 0u) {
        const std::size_t pairStride =
            static_cast<std::size_t>(candidate->articularContactPairCount) + 1u;
        if (acceptedHistoryElements >
            std::numeric_limits<std::size_t>::max() / pairStride) {
            return false;
        }
        candidate->articularContactPairForceAcceptedHistory.resize(
            acceptedHistoryElements * pairStride);
    }
    if (!candidate->passiveLigaments.empty()) {
        candidate->passiveLigamentAcceptedState.resize(
            candidate->environmentCount);
    }
    if (!candidate->passiveRoutedBands.empty()) {
        candidate->passiveRoutedBandAcceptedState.resize(
            candidate->environmentCount);
    }
    std::uint64_t fingerprint = 1469598103934665603ull;
    const std::uint64_t runtimeFingerprint =
        candidate->runtimeDeviceProgramFingerprint;
    fingerprint = appendFingerprint(
        fingerprint, &runtimeFingerprint, sizeof(runtimeFingerprint)
    );
    fingerprint = appendFingerprint(
        fingerprint, &candidate->endpointCount, sizeof(candidate->endpointCount)
    );
    fingerprint = appendFingerprint(
        fingerprint, &candidate->environmentCount,
        sizeof(candidate->environmentCount)
    );
    fingerprint = appendFingerprint(
        fingerprint, &candidate->activeAnchorCount,
        sizeof(candidate->activeAnchorCount)
    );
    fingerprint = appendFingerprint(
        fingerprint, &candidate->executionMode,
        sizeof(candidate->executionMode)
    );
    fingerprint = appendFingerprint(
        fingerprint, candidate->nodeLoads.data(),
        candidate->nodeLoads.size() * sizeof(candidate->nodeLoads.front())
    );
    fingerprint = appendFingerprint(
        fingerprint, candidate->nodeAnchors.data(),
        candidate->nodeAnchors.size() * sizeof(candidate->nodeAnchors.front())
    );
    fingerprint = appendFingerprint(
        fingerprint, candidate->replacements.data(),
        candidate->replacements.size() * sizeof(candidate->replacements.front())
    );
    fingerprint = appendFingerprint(
        fingerprint, candidate->contactSamples.data(),
        candidate->contactSamples.size() *
            sizeof(NMNumiHumanFEMContactSampleGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->contactContributions.data(),
        candidate->contactContributions.size() *
            sizeof(NMNumiHumanFEMContactContributionGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->contactRanges.data(),
        candidate->contactRanges.size() * sizeof(NMIncidenceRangeGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->femBodyContactSamples.data(),
        candidate->femBodyContactSamples.size() *
            sizeof(NMNumiHumanFEMBodyContactSampleGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->articularContactSamples.data(),
        candidate->articularContactSamples.size() *
            sizeof(NMNumiHumanArticularContactSampleGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->articularContactPairRanges.data(),
        candidate->articularContactPairRanges.size() *
            sizeof(NMIncidenceRangeGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->passiveLigaments.data(),
        candidate->passiveLigaments.size() *
            sizeof(NMNumiHumanPassiveLigamentGPU));
    fingerprint = appendFingerprint(
        fingerprint, candidate->passiveRoutedBands.data(),
        candidate->passiveRoutedBands.size() *
            sizeof(NMNumiHumanPassiveRoutedBandGPU));
    candidate->fingerprint = fingerprint == 0u ? 1u : fingerprint;
    candidate->message = "initialized";
    state_ = std::move(candidate);
    return true;
}

bool NumiHumanTendonFEMLoadAdapter::encodePreDynamics(
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    @autoreleasepool {
        if (state_ != nullptr && state_->executionMode ==
                NumiHumanTendonFEMLoadExecutionMode::numanXDeferred) {
            state_->deferredExternalForcesAvailable = false;
            state_->deferredCommandBufferIdentity = 0u;
        }
        if (state_ == nullptr || state_->runtime == nullptr ||
            pass.commandBuffer == nullptr || pass.bindings == nullptr ||
            pass.transfers == nullptr || pass.generalizedForces == nullptr ||
            pass.bodyPoses == nullptr || pass.pointJacobians == nullptr ||
            pass.environmentCount != state_->environmentCount ||
            pass.endpointCount != state_->endpointCount ||
            pass.environmentCount == 0u || pass.endpointCount == 0u ||
            pass.dofCount == 0u || pass.muscleCount == 0u ||
            pass.generalizedForceStride < pass.dofCount ||
            pass.pointJacobianStride == 0u ||
            pass.bodyJacobianPointOffset == MR_INVALID_INDEX ||
            pass.bodyPoseStride == 0u ||
            (state_->articularBodyPoseStride != 0u &&
             pass.bodyPoseStride != state_->articularBodyPoseStride) ||
            pass.stepIndex >=
                NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS) {
            if (state_ != nullptr)
                state_->message = "invalid borrowed Human pre-dynamics pass";
            return false;
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> bindings = (__bridge id<MTLBuffer>)pass.bindings;
        id<MTLBuffer> transfers = (__bridge id<MTLBuffer>)pass.transfers;
        id<MTLBuffer> generalizedForces =
            (__bridge id<MTLBuffer>)pass.generalizedForces;
        id<MTLBuffer> bodyPoses = (__bridge id<MTLBuffer>)pass.bodyPoses;
        id<MTLBuffer> pointJacobians =
            (__bridge id<MTLBuffer>)pass.pointJacobians;
        if (command == nil || bindings == nil || transfers == nil ||
            generalizedForces == nil || bodyPoses == nil ||
            pointJacobians == nil) {
            state_->message = "borrowed Human Metal objects are unavailable";
            return false;
        }
        const std::uint64_t muscleRowElements =
            static_cast<std::uint64_t>(pass.environmentCount) *
            pass.muscleCount * pass.dofCount;
        const std::uint64_t reducedForceEnd =
            static_cast<std::uint64_t>(pass.generalizedForceOffset) +
            static_cast<std::uint64_t>(pass.environmentCount - 1u) *
                pass.generalizedForceStride +
            pass.dofCount;
        const std::uint64_t requiredForceElements = std::max(
            muscleRowElements, reducedForceEnd
        );
        if (pass.generalizedForceOffset < muscleRowElements ||
            requiredForceElements > generalizedForces.length / sizeof(float)) {
            state_->message =
                "borrowed Human generalized-force workspace is undersized";
            return false;
        }
        id<MTLDevice> device = transfers.device;
        if (device == nil || bindings.device.registryID != device.registryID ||
            generalizedForces.device.registryID != device.registryID ||
            bodyPoses.device.registryID != device.registryID ||
            pointJacobians.device.registryID != device.registryID) {
            state_->message = "borrowed Human buffers do not share one Metal device";
            return false;
        }
        if (state_->device == nil) {
            state_->device = device;
            NSError* error = nil;
            NSString* path = [NSString
                stringWithUTF8String:state_->metallib.string().c_str()];
            state_->library = path == nil
                ? nil
                : [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                      error:&error];
            if (state_->library == nil) {
                state_->message = "Numi Matter metallib did not load for Human tendon/FEM assembly";
                return false;
            }
            const auto pipeline = [&](const char* name) {
                const std::string qualified =
                    std::string("numi_matter_metal::") + name;
                id<MTLFunction> function = [state_->library
                    newFunctionWithName:[NSString
                        stringWithUTF8String:qualified.c_str()]];
                if (function == nil) {
                    state_->message = std::string("missing Metal function ") + name;
                    return static_cast<id<MTLComputePipelineState>>(nil);
                }
                error = nil;
                id<MTLComputePipelineState> result = [device
                    newComputePipelineStateWithFunction:function
                                                   error:&error];
                if (result == nil) {
                    const char* description = error == nil
                        ? "unknown Metal error"
                        : error.localizedDescription.UTF8String;
                    state_->message = std::string("failed to compile pipeline ") +
                        name + ": " + (description == nullptr
                            ? "unknown Metal error"
                            : description);
                }
                return result;
            };
            state_->provisionalStatusPipeline = pipeline(
                "nm_numi_human_prepare_provisional_status");
            if (state_->executionMode ==
                    NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime) {
                state_->statusPipeline = pipeline(
                    "nm_numi_human_adapt_stand_status");
            }
            state_->forcePipeline = pipeline(
                "nm_numi_human_assemble_tendon_fem_loads");
            state_->forceAuditPipeline = pipeline(
                "nm_numi_human_audit_tendon_fem_loads");
            if (!state_->contactSamples.empty()) {
                state_->contactForcePipeline = pipeline(
                    "nm_numi_human_assemble_internal_fem_contact_loads");
            }
            if (!state_->femBodyContactSamples.empty()) {
                state_->allocatedArticularBodyPoseStride = pass.bodyPoseStride;
                state_->femBodyContactForcePipeline = pipeline(
                    "nm_numi_human_assemble_fem_body_contact_loads");
                state_->femBodyContactWrenchPipeline = pipeline(
                    "nm_numi_human_assemble_fem_body_contact_wrenches");
                state_->femBodyContactAuditPipeline = pipeline(
                    "nm_numi_human_audit_fem_body_contact");
            }
            if (!state_->articularContactSamples.empty()) {
                state_->allocatedArticularBodyPoseStride = pass.bodyPoseStride;
                state_->articularContactPipeline = pipeline(
                    "nm_numi_human_assemble_articular_contact_wrenches");
                state_->articularContactAuditPipeline = pipeline(
                    "nm_numi_human_audit_articular_contact_wrenches");
            }
            if (state_->articularContactPairCount != 0u) {
                state_->articularContactPairAuditPipeline = pipeline(
                    "nm_numi_human_audit_articular_contact_pairs");
                state_->articularContactPairAuditCommitPipeline = pipeline(
                    "nm_numi_human_commit_articular_contact_pair_audit");
            }
            if (!state_->articularContactSamples.empty() ||
                !state_->femBodyContactSamples.empty()) {
                state_->articularContactAuditCommitPipeline = pipeline(
                    "nm_numi_human_commit_articular_contact_audit");
            }
            if (!state_->passiveLigaments.empty()) {
                state_->passiveLigamentAuditPipeline = pipeline(
                    "nm_numi_human_audit_passive_ligaments");
                state_->passiveLigamentAuditCommitPipeline = pipeline(
                    "nm_numi_human_commit_passive_ligament_audit");
            }
            if (!state_->passiveRoutedBands.empty()) {
                state_->passiveRoutedBandAuditPipeline = pipeline(
                    "nm_numi_human_audit_passive_routed_bands");
                state_->passiveRoutedBandAuditCommitPipeline = pipeline(
                    "nm_numi_human_commit_passive_routed_band_audit");
            }
            if (state_->executionMode ==
                    NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime) {
                state_->targetPipeline = pipeline(
                    "nm_numi_human_assemble_fem_kinematic_targets");
                state_->reactionAuditPipeline = pipeline(
                    "nm_numi_human_audit_fem_anchor_reactions");
                state_->reactionAuditCommitPipeline = pipeline(
                    "nm_numi_human_commit_fem_anchor_reaction_audit");
                state_->reactionPipeline = pipeline(
                    "nm_numi_human_apply_fem_anchor_reactions");
            } else {
                state_->preSourceCorrectionPipeline = pipeline(
                    "nm_numi_human_apply_pre_source_rigid_corrections");
            }
            const NSUInteger nodeLoadBytes = static_cast<NSUInteger>(
                state_->nodeLoads.size() * sizeof(state_->nodeLoads.front())
            );
            const NSUInteger nodeAnchorBytes = static_cast<NSUInteger>(
                state_->nodeAnchors.size() * sizeof(state_->nodeAnchors.front())
            );
            const NSUInteger replacementBytes = static_cast<NSUInteger>(
                state_->replacements.size() * sizeof(state_->replacements.front())
            );
            const NSUInteger externalForceBytes = static_cast<NSUInteger>(
                state_->environmentCount * state_->nodeLoads.size() *
                sizeof(nm_float4)
            );
            const NSUInteger statusBytes = static_cast<NSUInteger>(
                state_->environmentCount * sizeof(MRMetalWorldStatusGPU)
            );
            state_->nodeLoadBuffer = [device
                newBufferWithBytes:state_->nodeLoads.data()
                length:nodeLoadBytes
                options:MTLResourceStorageModeShared];
            state_->nodeAnchorBuffer = [device
                newBufferWithBytes:state_->nodeAnchors.data()
                length:nodeAnchorBytes
                options:MTLResourceStorageModeShared];
            state_->replacementBuffer = replacementBytes == 0u
                ? [device
                    newBufferWithLength:sizeof(
                        NMNumiHumanTendonFEMEndpointReplacementGPU)
                    options:MTLResourceStorageModeShared]
                : [device
                    newBufferWithBytes:state_->replacements.data()
                    length:replacementBytes
                    options:MTLResourceStorageModeShared];
            if (!state_->contactSamples.empty()) {
                state_->contactSampleBuffer = [device
                    newBufferWithBytes:state_->contactSamples.data()
                    length:state_->contactSamples.size() *
                        sizeof(NMNumiHumanFEMContactSampleGPU)
                    options:MTLResourceStorageModeShared];
                state_->contactContributionBuffer = [device
                    newBufferWithBytes:state_->contactContributions.data()
                    length:state_->contactContributions.size() *
                        sizeof(NMNumiHumanFEMContactContributionGPU)
                    options:MTLResourceStorageModeShared];
                state_->contactRangeBuffer = [device
                    newBufferWithBytes:state_->contactRanges.data()
                    length:state_->contactRanges.size() *
                        sizeof(NMIncidenceRangeGPU)
                    options:MTLResourceStorageModeShared];
            }
            if (!state_->femBodyContactSamples.empty()) {
                state_->femBodyContactSampleBuffer = [device
                    newBufferWithBytes:state_->femBodyContactSamples.data()
                    length:state_->femBodyContactSamples.size() *
                        sizeof(NMNumiHumanFEMBodyContactSampleGPU)
                    options:MTLResourceStorageModeShared];
                state_->femBodyContactWrenchBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        pass.bodyPoseStride *
                        sizeof(NMNumiHumanBodyWrenchGPU)
                    options:MTLResourceStorageModePrivate];
                state_->femBodyContactAuditBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        sizeof(NMNumiHumanArticularContactAuditGPU)
                    options:MTLResourceStorageModeShared];
                state_->femBodyContactAuditHistoryBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS *
                        sizeof(NMNumiHumanArticularContactAuditGPU)
                    options:MTLResourceStorageModeShared];
                if (state_->femBodyContactAuditBuffer != nil) {
                    std::memset(
                        state_->femBodyContactAuditBuffer.contents, 0,
                        state_->femBodyContactAuditBuffer.length);
                }
                if (state_->femBodyContactAuditHistoryBuffer != nil) {
                    std::memcpy(
                        state_->femBodyContactAuditHistoryBuffer.contents,
                        state_->femBodyContactAcceptedHistory.data(),
                        state_->femBodyContactAuditHistoryBuffer.length);
                }
            }
            if (!state_->articularContactSamples.empty()) {
                state_->articularContactSampleBuffer = [device
                    newBufferWithBytes:state_->articularContactSamples.data()
                    length:state_->articularContactSamples.size() *
                        sizeof(NMNumiHumanArticularContactSampleGPU)
                    options:MTLResourceStorageModeShared];
                state_->articularBodyWrenchBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        pass.bodyPoseStride *
                        sizeof(NMNumiHumanBodyWrenchGPU)
                    options:MTLResourceStorageModePrivate];
                state_->articularContactAuditBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        sizeof(NMNumiHumanArticularContactAuditGPU)
                    options:MTLResourceStorageModeShared];
                state_->articularContactAuditHistoryBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS *
                        sizeof(NMNumiHumanArticularContactAuditGPU)
                    options:MTLResourceStorageModeShared];
                if (state_->articularContactAuditBuffer != nil) {
                    std::memset(
                        state_->articularContactAuditBuffer.contents, 0,
                        state_->articularContactAuditBuffer.length);
                }
                if (state_->articularContactAuditHistoryBuffer != nil) {
                    std::memcpy(
                        state_->articularContactAuditHistoryBuffer.contents,
                        state_->articularContactAcceptedHistory.data(),
                        state_->articularContactAuditHistoryBuffer.length);
                }
                if (state_->articularContactPairCount != 0u) {
                    state_->articularContactPairRangeBuffer = [device
                        newBufferWithBytes:
                            state_->articularContactPairRanges.data()
                        length:state_->articularContactPairRanges.size() *
                            sizeof(NMIncidenceRangeGPU)
                        options:MTLResourceStorageModeShared];
                    const std::size_t pairStride =
                        static_cast<std::size_t>(
                            state_->articularContactPairCount) + 1u;
                    state_->articularContactPairAuditBuffer = [device
                        newBufferWithLength:state_->environmentCount *
                            pairStride * sizeof(float)
                        options:MTLResourceStorageModeShared];
                    state_->articularContactPairAuditHistoryBuffer = [device
                        newBufferWithLength:
                            state_->articularContactPairForceAcceptedHistory
                                .size() * sizeof(float)
                        options:MTLResourceStorageModeShared];
                    if (state_->articularContactPairAuditBuffer != nil) {
                        std::memset(
                            state_->articularContactPairAuditBuffer.contents,
                            0,
                            state_->articularContactPairAuditBuffer.length);
                    }
                    if (state_->articularContactPairAuditHistoryBuffer != nil) {
                        std::memcpy(
                            state_->articularContactPairAuditHistoryBuffer
                                .contents,
                            state_->articularContactPairForceAcceptedHistory
                                .data(),
                            state_->articularContactPairAuditHistoryBuffer
                                .length);
                    }
                }
            }
            if (!state_->passiveLigaments.empty()) {
                state_->passiveLigamentBuffer = [device
                    newBufferWithBytes:state_->passiveLigaments.data()
                    length:state_->passiveLigaments.size() *
                        sizeof(NMNumiHumanPassiveLigamentGPU)
                    options:MTLResourceStorageModeShared];
                state_->passiveLigamentAuditBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        sizeof(NMNumiHumanPassiveLigamentAuditGPU)
                    options:MTLResourceStorageModeShared];
                if (state_->passiveLigamentAuditBuffer != nil) {
                    std::memcpy(
                        state_->passiveLigamentAuditBuffer.contents,
                        state_->passiveLigamentAcceptedState.data(),
                        state_->passiveLigamentAuditBuffer.length);
                }
            }
            if (!state_->passiveRoutedBands.empty()) {
                state_->passiveRoutedBandBuffer = [device
                    newBufferWithBytes:state_->passiveRoutedBands.data()
                    length:state_->passiveRoutedBands.size() *
                        sizeof(NMNumiHumanPassiveRoutedBandGPU)
                    options:MTLResourceStorageModeShared];
                state_->passiveRoutedBandAuditBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        sizeof(NMNumiHumanPassiveRoutedBandAuditGPU)
                    options:MTLResourceStorageModeShared];
                if (state_->passiveRoutedBandAuditBuffer != nil) {
                    std::memcpy(
                        state_->passiveRoutedBandAuditBuffer.contents,
                        state_->passiveRoutedBandAcceptedState.data(),
                        state_->passiveRoutedBandAuditBuffer.length);
                }
            }
            state_->externalForceBuffer = [device
                newBufferWithLength:externalForceBytes
                options:MTLResourceStorageModePrivate];
            state_->externalForceAuditBuffer = [device
                newBufferWithLength:state_->environmentCount * sizeof(nm_float4)
                options:MTLResourceStorageModeShared];
            if (state_->executionMode ==
                    NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime) {
                state_->anchorReactionAuditBuffer = [device
                    newBufferWithLength:state_->environmentCount * sizeof(nm_float4)
                    options:MTLResourceStorageModeShared];
                state_->anchorReactionAuditHistoryBuffer = [device
                    newBufferWithLength:state_->environmentCount *
                        NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS *
                        sizeof(nm_float4)
                    options:MTLResourceStorageModeShared];
                if (state_->anchorReactionAuditHistoryBuffer != nil) {
                    std::memcpy(
                        state_->anchorReactionAuditHistoryBuffer.contents,
                        state_->anchorReactionAcceptedHistory.data(),
                        state_->anchorReactionAuditHistoryBuffer.length);
                }
                state_->kinematicTargetBuffer = [device
                    newBufferWithLength:externalForceBytes
                    options:MTLResourceStorageModePrivate];
            }
            state_->worldStatusBuffer = [device
                newBufferWithLength:statusBytes
                options:MTLResourceStorageModeShared];
            if (state_->provisionalStatusPipeline == nil ||
                state_->forcePipeline == nil ||
                state_->forceAuditPipeline == nil ||
                (!state_->contactSamples.empty() &&
                 state_->contactForcePipeline == nil) ||
                (!state_->femBodyContactSamples.empty() &&
                 (state_->femBodyContactForcePipeline == nil ||
                  state_->femBodyContactWrenchPipeline == nil ||
                  state_->femBodyContactAuditPipeline == nil ||
                  state_->articularContactAuditCommitPipeline == nil)) ||
                (!state_->articularContactSamples.empty() &&
                 (state_->articularContactPipeline == nil ||
                  state_->articularContactAuditPipeline == nil ||
                  state_->articularContactAuditCommitPipeline == nil)) ||
                (state_->articularContactPairCount != 0u &&
                 (state_->articularContactPairAuditPipeline == nil ||
                  state_->articularContactPairAuditCommitPipeline == nil)) ||
                (!state_->passiveLigaments.empty() &&
                 (state_->passiveLigamentAuditPipeline == nil ||
                  state_->passiveLigamentAuditCommitPipeline == nil)) ||
                (!state_->passiveRoutedBands.empty() &&
                 (state_->passiveRoutedBandAuditPipeline == nil ||
                  state_->passiveRoutedBandAuditCommitPipeline == nil)) ||
                (state_->executionMode ==
                        NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime &&
                 (state_->targetPipeline == nil ||
                  state_->statusPipeline == nil ||
                  state_->reactionAuditPipeline == nil ||
                  state_->reactionAuditCommitPipeline == nil ||
                  state_->reactionPipeline == nil)) ||
                (state_->executionMode ==
                        NumiHumanTendonFEMLoadExecutionMode::numanXDeferred &&
                 state_->preSourceCorrectionPipeline == nil)) {
                if (state_->message == "initialized") {
                    state_->message = "Human tendon/FEM pipeline is unavailable";
                }
                return false;
            }
            if (state_->nodeLoadBuffer == nil || state_->nodeAnchorBuffer == nil ||
                state_->replacementBuffer == nil) {
                state_->message = "Human tendon/FEM immutable mapping buffer is unavailable";
                return false;
            }
            if (!state_->contactSamples.empty() &&
                (state_->contactSampleBuffer == nil ||
                 state_->contactContributionBuffer == nil ||
                 state_->contactRangeBuffer == nil)) {
                state_->message =
                    "Human FEM contact immutable mapping buffer is unavailable";
                return false;
            }
            if (!state_->femBodyContactSamples.empty() &&
                (state_->femBodyContactSampleBuffer == nil ||
                 state_->femBodyContactWrenchBuffer == nil ||
                 state_->femBodyContactAuditBuffer == nil ||
                 state_->femBodyContactAuditHistoryBuffer == nil)) {
                state_->message =
                    "Human FEM/body contact buffer is unavailable";
                return false;
            }
            if (!state_->articularContactSamples.empty() &&
                (state_->articularContactSampleBuffer == nil ||
                 state_->articularBodyWrenchBuffer == nil ||
                 state_->articularContactAuditBuffer == nil ||
                 state_->articularContactAuditHistoryBuffer == nil)) {
                state_->message =
                    "Human articular contact buffer is unavailable";
                return false;
            }
            if (state_->articularContactPairCount != 0u &&
                (state_->articularContactPairRangeBuffer == nil ||
                 state_->articularContactPairAuditBuffer == nil ||
                 state_->articularContactPairAuditHistoryBuffer == nil)) {
                state_->message =
                    "Human articular pair-audit buffer is unavailable";
                return false;
            }
            if (!state_->passiveLigaments.empty() &&
                (state_->passiveLigamentBuffer == nil ||
                 state_->passiveLigamentAuditBuffer == nil)) {
                state_->message =
                    "Human passive ligament buffer is unavailable";
                return false;
            }
            if (!state_->passiveRoutedBands.empty() &&
                (state_->passiveRoutedBandBuffer == nil ||
                 state_->passiveRoutedBandAuditBuffer == nil)) {
                state_->message =
                    "Human passive routed-band buffer is unavailable";
                return false;
            }
            if (state_->externalForceBuffer == nil ||
                state_->externalForceAuditBuffer == nil ||
                (state_->executionMode ==
                        NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime &&
                 (state_->anchorReactionAuditBuffer == nil ||
                  state_->anchorReactionAuditHistoryBuffer == nil ||
                  state_->kinematicTargetBuffer == nil))) {
                state_->message = "Human tendon/FEM force/target buffer is unavailable";
                return false;
            }
            if (state_->worldStatusBuffer == nil) {
                state_->message = "Human tendon/FEM world-status buffer is unavailable";
                return false;
            }
        } else if (state_->device.registryID != device.registryID) {
            state_->message = "Human tendon/FEM adapter changed Metal devices";
            return false;
        }
        if ((!state_->articularContactSamples.empty() ||
             !state_->femBodyContactSamples.empty()) &&
            (state_->allocatedArticularBodyPoseStride != pass.bodyPoseStride ||
             (state_->articularBodyPoseStride != 0u &&
              state_->articularBodyPoseStride != pass.bodyPoseStride))) {
            state_->message =
                "Human tendon/FEM adapter changed its bound body-pose stride";
            return false;
        }
        if (state_->articularBodyPoseStride == 0u) {
            state_->articularBodyPoseStride = pass.bodyPoseStride;
        }

        const NMNumiHumanTendonFEMLoadDispatchGPU dispatch{
            .abiVersion = NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION,
            .environmentCount = state_->environmentCount,
            .femNodeCount = static_cast<std::uint32_t>(state_->nodeLoads.size()),
            .endpointCount = state_->endpointCount,
            .transferStride = pass.endpointCount,
            .stepIndex = pass.stepIndex,
            .replacementCount = static_cast<std::uint32_t>(
                state_->replacements.size()),
            .dofCount = pass.dofCount,
            .bodyPoseStride = pass.bodyPoseStride,
            .articulationFirstBody = pass.articulationFirstBody,
            .pointJacobianStride = pass.pointJacobianStride,
            .bodyJacobianPointOffset = pass.bodyJacobianPointOffset,
            .generalizedForceStride = pass.generalizedForceStride,
            .generalizedForceOffset = pass.generalizedForceOffset,
            .muscleCount = pass.muscleCount,
            .femContactSampleCount = static_cast<std::uint32_t>(
                state_->contactSamples.size()),
            .articularContactSampleCount = static_cast<std::uint32_t>(
                state_->articularContactSamples.size()),
            .passiveLigamentCount = static_cast<std::uint32_t>(
                state_->passiveLigaments.size()),
            .femBodyContactSampleCount = static_cast<std::uint32_t>(
                state_->femBodyContactSamples.size()),
            .passiveRoutedBandCount = static_cast<std::uint32_t>(
                state_->passiveRoutedBands.size()),
            .articularContactPairCount =
                state_->articularContactPairCount,
        };
        const auto encodeKernel = [&](id<MTLComputePipelineState> pipeline,
                                      const NSUInteger count,
                                      const auto& bind) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (encoder == nil) return false;
            [encoder setComputePipelineState:pipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            bind(encoder);
            const NSUInteger width = std::min(
                count,
                std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256u)
            );
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(std::max<NSUInteger>(width, 1u), 1u, 1u)];
            [encoder endEncoding];
            return true;
        };
        if (!encodeKernel(
                state_->provisionalStatusPipeline, state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeLoadBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:state_->nodeAnchorBuffer offset:0u atIndex:2u];
                    [encoder setBuffer:state_->replacementBuffer offset:0u atIndex:3u];
                    [encoder setBuffer:bindings offset:0u atIndex:4u];
                    [encoder setBuffer:transfers offset:0u atIndex:5u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:6u];
                    [encoder setBuffer:state_->worldStatusBuffer
                                offset:0u atIndex:7u];
                    if (state_->articularContactSampleBuffer != nil) {
                        [encoder setBuffer:state_->articularContactSampleBuffer offset:0u atIndex:8u];
                    } else {
                        // Metal requires a typed binding even when the dispatch
                        // count is zero. This is never a physical sample.
                        const NMNumiHumanArticularContactSampleGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:8u];
                    }
                    if (state_->passiveLigamentBuffer != nil) {
                        [encoder setBuffer:state_->passiveLigamentBuffer offset:0u atIndex:9u];
                    } else {
                        // Metal requires a typed binding even when the dispatch
                        // count is zero. This is never a physical sample.
                        const NMNumiHumanPassiveLigamentGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:9u];
                    }
                    if (state_->femBodyContactSampleBuffer != nil) {
                        [encoder setBuffer:state_->femBodyContactSampleBuffer offset:0u atIndex:10u];
                    } else {
                        // Metal requires a typed binding even when the dispatch
                        // count is zero. This is never a physical sample.
                        const NMNumiHumanFEMBodyContactSampleGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:10u];
                    }
                    if (state_->passiveRoutedBandBuffer != nil) {
                        [encoder setBuffer:state_->passiveRoutedBandBuffer offset:0u atIndex:11u];
                    } else {
                        // Metal requires a typed binding even when the dispatch
                        // count is zero. This is never a physical sample.
                        const NMNumiHumanPassiveRoutedBandGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:11u];
                    }
                }
            ) || !encodeKernel(
                state_->forcePipeline,
                state_->environmentCount * state_->nodeLoads.size(),
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeLoadBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:transfers offset:0u atIndex:2u];
                    [encoder setBuffer:state_->externalForceBuffer
                                offset:0u atIndex:3u];
                }
            )) {
            state_->message = "Human tendon/FEM assembly kernel encoding failed";
            return false;
        }
        id<MTLBuffer> acceptedNodes = nil;
        if (!state_->contactSamples.empty() ||
            !state_->femBodyContactSamples.empty()) {
            acceptedNodes = (__bridge id<MTLBuffer>)
                state_->runtime->femAcceptedNodeBuffer();
            const std::uint64_t expectedAcceptedNodeBytes =
                static_cast<std::uint64_t>(state_->environmentCount) *
                state_->nodeLoads.size() * sizeof(NMFEMNodeStateGPU);
            if (acceptedNodes == nil ||
                acceptedNodes.device.registryID != state_->device.registryID ||
                acceptedNodes.length < expectedAcceptedNodeBytes) {
                state_->message =
                    "Human FEM contact accepted-node arena is unavailable";
                return false;
            }
        }
        if (!state_->contactSamples.empty() &&
            !encodeKernel(
                    state_->contactForcePipeline,
                    state_->environmentCount * state_->nodeLoads.size(),
                    [&](id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:state_->contactSampleBuffer
                                    offset:0u atIndex:1u];
                        [encoder setBuffer:state_->contactContributionBuffer
                                    offset:0u atIndex:2u];
                        [encoder setBuffer:state_->contactRangeBuffer
                                    offset:0u atIndex:3u];
                        [encoder setBuffer:acceptedNodes offset:0u atIndex:4u];
                        [encoder setBuffer:state_->externalForceBuffer
                                    offset:0u atIndex:5u];
                    })) {
                state_->message =
                    "Human internal FEM contact kernel encoding failed";
                return false;
        }
        if (!state_->femBodyContactSamples.empty() &&
            (!encodeKernel(
                state_->femBodyContactForcePipeline,
                state_->environmentCount * state_->nodeLoads.size(),
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->femBodyContactSampleBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:acceptedNodes offset:0u atIndex:2u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:3u];
                    [encoder setBuffer:state_->externalForceBuffer
                                offset:0u atIndex:4u];
                }) || !encodeKernel(
                state_->femBodyContactWrenchPipeline,
                state_->environmentCount * pass.bodyPoseStride,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->femBodyContactSampleBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:acceptedNodes offset:0u atIndex:2u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:3u];
                    [encoder setBuffer:state_->femBodyContactWrenchBuffer
                                offset:0u atIndex:4u];
                }) || !encodeKernel(
                state_->femBodyContactAuditPipeline,
                state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->femBodyContactSampleBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:acceptedNodes offset:0u atIndex:2u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:3u];
                    [encoder setBuffer:state_->femBodyContactAuditBuffer
                                offset:0u atIndex:4u];
                }))) {
            state_->message = "Human FEM/body contact encoding failed";
            return false;
        }
        if (!state_->articularContactSamples.empty() &&
            !encodeKernel(
                state_->articularContactPipeline,
                state_->environmentCount * pass.bodyPoseStride,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->articularContactSampleBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:2u];
                    [encoder setBuffer:state_->articularBodyWrenchBuffer
                                offset:0u atIndex:3u];
                })) {
            state_->message =
                "Human articular contact wrench encoding failed";
            return false;
        }
        if (!state_->passiveLigaments.empty() &&
            !encodeKernel(
                state_->passiveLigamentAuditPipeline,
                state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->passiveLigamentBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:2u];
                    [encoder setBuffer:state_->passiveLigamentAuditBuffer
                                offset:0u atIndex:3u];
                })) {
            state_->message = "Human passive ligament audit encoding failed";
            return false;
        }
        if (!state_->passiveRoutedBands.empty() &&
            !encodeKernel(
                state_->passiveRoutedBandAuditPipeline,
                state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->passiveRoutedBandBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:2u];
                    [encoder setBuffer:state_->passiveRoutedBandAuditBuffer
                                offset:0u atIndex:3u];
                })) {
            state_->message = "Human passive routed-band audit encoding failed";
            return false;
        }
        if (!state_->articularContactSamples.empty() &&
            !encodeKernel(
                state_->articularContactAuditPipeline,
                state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->articularContactSampleBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:2u];
                    [encoder setBuffer:state_->articularBodyWrenchBuffer
                                offset:0u atIndex:3u];
                    [encoder setBuffer:state_->articularContactAuditBuffer
                                offset:0u atIndex:4u];
                })) {
            state_->message =
                "Human articular contact audit encoding failed";
            return false;
        }
        if (state_->articularContactPairCount != 0u &&
            !encodeKernel(
                state_->articularContactPairAuditPipeline,
                state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->articularContactSampleBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:state_->articularContactPairRangeBuffer
                                offset:0u atIndex:2u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:3u];
                    [encoder setBuffer:state_->articularContactPairAuditBuffer
                                offset:0u atIndex:4u];
                })) {
            state_->message =
                "Human articular contact pair audit encoding failed";
            return false;
        }
        if (!encodeKernel(
                state_->forceAuditPipeline, state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->externalForceBuffer
                                offset:0u atIndex:1u];
                    [encoder setBuffer:state_->externalForceAuditBuffer
                                offset:0u atIndex:2u];
                })) {
            state_->message = "Human tendon/FEM assembly kernel encoding failed";
            return false;
        }
        if (state_->executionMode ==
                NumiHumanTendonFEMLoadExecutionMode::numanXDeferred) {
            if (!encodeKernel(
                    state_->preSourceCorrectionPipeline,
                    state_->environmentCount * pass.dofCount,
                    [&](id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:state_->replacementBuffer
                                    offset:0u atIndex:1u];
                        [encoder setBuffer:bindings offset:0u atIndex:2u];
                        [encoder setBuffer:transfers offset:0u atIndex:3u];
                        [encoder setBuffer:bodyPoses offset:0u atIndex:4u];
                        [encoder setBuffer:pointJacobians offset:0u atIndex:5u];
                        [encoder setBuffer:generalizedForces offset:0u atIndex:6u];
                        if (state_->articularBodyWrenchBuffer != nil) {
                            [encoder setBuffer:state_->articularBodyWrenchBuffer
                                        offset:0u atIndex:7u];
                        } else {
                            const NMNumiHumanBodyWrenchGPU empty{};
                            [encoder setBytes:&empty length:sizeof(empty)
                                      atIndex:7u];
                        }
                        if (state_->passiveLigamentBuffer != nil) {
                            [encoder setBuffer:state_->passiveLigamentBuffer
                                        offset:0u atIndex:8u];
                        } else {
                            const NMNumiHumanPassiveLigamentGPU empty{};
                            [encoder setBytes:&empty length:sizeof(empty)
                                      atIndex:8u];
                        }
                        if (state_->femBodyContactWrenchBuffer != nil) {
                            [encoder setBuffer:state_->femBodyContactWrenchBuffer
                                        offset:0u atIndex:9u];
                        } else {
                            const NMNumiHumanBodyWrenchGPU empty{};
                            [encoder setBytes:&empty length:sizeof(empty)
                                      atIndex:9u];
                        }
                        if (state_->passiveRoutedBandBuffer != nil) {
                            [encoder setBuffer:state_->passiveRoutedBandBuffer
                                        offset:0u atIndex:10u];
                        } else {
                            const NMNumiHumanPassiveRoutedBandGPU empty{};
                            [encoder setBytes:&empty length:sizeof(empty)
                                      atIndex:10u];
                        }
                        [encoder setBuffer:state_->worldStatusBuffer
                                    offset:0u atIndex:11u];
                    })) {
                state_->message =
                    "Human tendon/FEM pre-source correction encoding failed";
                return false;
            }
            ++state_->preSourceCorrectionEncodeCount;
            state_->deferredCommandBufferIdentity =
                reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
            state_->deferredStepIndex = pass.stepIndex;
            state_->deferredExternalForcesAvailable = true;
            state_->articularAttemptedStepCount = std::max(
                state_->articularAttemptedStepCount, pass.stepIndex + 1u);
            state_->message = "deferred pre-source loads encoded";
            return true;
        }
        if (!encodeKernel(
                state_->targetPipeline,
                state_->environmentCount * state_->nodeAnchors.size(),
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeAnchorBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:2u];
                    [encoder setBuffer:state_->kinematicTargetBuffer
                                offset:0u atIndex:3u];
                })) {
            state_->message = "Human tendon/FEM target encoding failed";
            return false;
        }

        EncodeRequest request{};
        request.commandBuffer = pass.commandBuffer;
        request.environmentStatuses =
            (__bridge void*)state_->worldStatusBuffer;
        request.femExternalForces =
            (__bridge void*)state_->externalForceBuffer;
        request.femExternalForceCount = static_cast<std::uint32_t>(
            state_->environmentCount * state_->nodeLoads.size()
        );
        request.femKinematicTargets =
            (__bridge void*)state_->kinematicTargetBuffer;
        request.femKinematicTargetCount = request.femExternalForceCount;
        request.controlStep = pass.stepIndex;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = state_->runtime->timestepSeconds();
        request.phase = EncodePhase::preDynamics;
        const auto pre = state_->runtime->encode(request);
        ++state_->runtimePreDynamicsEncodeCount;
        if (!pre.encoded) {
            state_->message = "Human tendon/FEM Matter pre-dynamics failed: " + pre.message;
            return false;
        }
        id<MTLBuffer> reactions = (__bridge id<MTLBuffer>)
            state_->runtime->femConstraintReactionBuffer();
        id<MTLBuffer> matterStatuses =
            (__bridge id<MTLBuffer>)state_->runtime->statusBuffer();
        if (reactions == nil || matterStatuses == nil ||
            !encodeKernel(
                state_->reactionAuditPipeline, state_->environmentCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeAnchorBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:reactions offset:0u atIndex:2u];
                    [encoder setBuffer:matterStatuses offset:0u atIndex:3u];
                    [encoder setBuffer:state_->anchorReactionAuditBuffer
                                offset:0u atIndex:4u];
                }
            ) || !encodeKernel(
                state_->reactionPipeline,
                state_->environmentCount * pass.dofCount,
                [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:state_->nodeAnchorBuffer offset:0u atIndex:1u];
                    [encoder setBuffer:state_->replacementBuffer offset:0u atIndex:2u];
                    [encoder setBuffer:bindings offset:0u atIndex:3u];
                    [encoder setBuffer:transfers offset:0u atIndex:4u];
                    [encoder setBuffer:reactions offset:0u atIndex:5u];
                    [encoder setBuffer:matterStatuses offset:0u atIndex:6u];
                    [encoder setBuffer:bodyPoses offset:0u atIndex:7u];
                    [encoder setBuffer:pointJacobians offset:0u atIndex:8u];
                    [encoder setBuffer:generalizedForces offset:0u atIndex:9u];
                    if (state_->articularBodyWrenchBuffer != nil) {
                        [encoder setBuffer:state_->articularBodyWrenchBuffer offset:0u atIndex:10u];
                    } else {
                        const NMNumiHumanBodyWrenchGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:10u];
                    }
                    if (state_->passiveLigamentBuffer != nil) {
                        [encoder setBuffer:state_->passiveLigamentBuffer offset:0u atIndex:11u];
                    } else {
                        const NMNumiHumanPassiveLigamentGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:11u];
                    }
                    if (state_->femBodyContactWrenchBuffer != nil) {
                        [encoder setBuffer:state_->femBodyContactWrenchBuffer offset:0u atIndex:12u];
                    } else {
                        const NMNumiHumanBodyWrenchGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:12u];
                    }
                    if (state_->passiveRoutedBandBuffer != nil) {
                        [encoder setBuffer:state_->passiveRoutedBandBuffer offset:0u atIndex:13u];
                    } else {
                        const NMNumiHumanPassiveRoutedBandGPU empty{};
                        [encoder setBytes:&empty length:sizeof(empty) atIndex:13u];
                    }
                }
            )) {
            state_->message = "Human tendon/FEM anchor-reaction encoding failed";
            return false;
        }
        state_->articularAttemptedStepCount = std::max(
            state_->articularAttemptedStepCount, pass.stepIndex + 1u);
        state_->message = "pre-dynamics encoded";
        return true;
    }
}

bool NumiHumanTendonFEMLoadAdapter::encodePostValidation(
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    @autoreleasepool {
        if (state_ == nullptr || state_->runtime == nullptr ||
            state_->device == nil ||
            (state_->executionMode ==
                 NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime &&
             state_->statusPipeline == nil) ||
            pass.commandBuffer == nullptr || pass.standStatuses == nullptr ||
            pass.environmentCount != state_->environmentCount ||
            pass.endpointCount != state_->endpointCount ||
            pass.dofCount == 0u || pass.muscleCount == 0u ||
            pass.stepIndex >=
                NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS) {
            if (state_ != nullptr)
                state_->message = "invalid borrowed Human post-validation pass";
            return false;
        }
        const bool deferred = state_->executionMode ==
            NumiHumanTendonFEMLoadExecutionMode::numanXDeferred;
        if (deferred &&
            (!state_->deferredExternalForcesAvailable ||
             state_->deferredCommandBufferIdentity !=
                reinterpret_cast<std::uintptr_t>(pass.commandBuffer) ||
             state_->deferredStepIndex != pass.stepIndex)) {
            state_->message =
                "deferred tendon/FEM post-validation lost its exact pre-source pass";
            return false;
        }
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        id<MTLBuffer> standStatuses =
            (__bridge id<MTLBuffer>)pass.standStatuses;
        id<MTLBuffer> matterStatuses = deferred ? nil :
            (__bridge id<MTLBuffer>)state_->runtime->statusBuffer();
        if (command == nil || standStatuses == nil ||
            standStatuses.device.registryID != state_->device.registryID ||
            (!deferred &&
             (matterStatuses == nil ||
              matterStatuses.device.registryID != state_->device.registryID))) {
            state_->message =
                "borrowed Human or Matter status is unavailable";
            return false;
        }
        const NMNumiHumanTendonFEMLoadDispatchGPU dispatch{
            .abiVersion = NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION,
            .environmentCount = state_->environmentCount,
            .femNodeCount = static_cast<std::uint32_t>(state_->nodeLoads.size()),
            .endpointCount = state_->endpointCount,
            .transferStride = pass.endpointCount,
            .stepIndex = pass.stepIndex,
            .replacementCount = static_cast<std::uint32_t>(
                state_->replacements.size()),
            .dofCount = pass.dofCount,
            .bodyPoseStride = pass.bodyPoseStride,
            .articulationFirstBody = pass.articulationFirstBody,
            .pointJacobianStride = pass.pointJacobianStride,
            .bodyJacobianPointOffset = pass.bodyJacobianPointOffset,
            .generalizedForceStride = pass.generalizedForceStride,
            .generalizedForceOffset = pass.generalizedForceOffset,
            .muscleCount = pass.muscleCount,
            .femContactSampleCount = static_cast<std::uint32_t>(
                state_->contactSamples.size()),
            .articularContactSampleCount = static_cast<std::uint32_t>(
                state_->articularContactSamples.size()),
            .passiveLigamentCount = static_cast<std::uint32_t>(
                state_->passiveLigaments.size()),
            .femBodyContactSampleCount = static_cast<std::uint32_t>(
                state_->femBodyContactSamples.size()),
            .passiveRoutedBandCount = static_cast<std::uint32_t>(
                state_->passiveRoutedBands.size()),
            .articularContactPairCount =
                state_->articularContactPairCount,
        };
        const NSUInteger count = state_->environmentCount;
        if (!deferred) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (encoder == nil) {
                state_->message =
                    "Human tendon/FEM status encoder is unavailable";
                return false;
            }
            [encoder setComputePipelineState:state_->statusPipeline];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [encoder setBuffer:standStatuses offset:0u atIndex:1u];
            [encoder setBuffer:state_->worldStatusBuffer offset:0u atIndex:2u];
            [encoder setBuffer:matterStatuses offset:0u atIndex:3u];
            const NSUInteger width = std::min<NSUInteger>(
                count, std::min<NSUInteger>(
                    state_->statusPipeline.maxTotalThreadsPerThreadgroup,
                    256u));
            [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    std::max<NSUInteger>(width, 1u), 1u, 1u)];
            [encoder endEncoding];

            EncodeRequest request{};
            request.commandBuffer = pass.commandBuffer;
            request.environmentStatuses =
                (__bridge void*)state_->worldStatusBuffer;
            request.femExternalForces =
                (__bridge void*)state_->externalForceBuffer;
            request.femExternalForceCount = static_cast<std::uint32_t>(
                state_->environmentCount * state_->nodeLoads.size());
            request.femKinematicTargets =
                (__bridge void*)state_->kinematicTargetBuffer;
            request.femKinematicTargetCount = request.femExternalForceCount;
            request.controlStep = pass.stepIndex;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = state_->runtime->timestepSeconds();
            request.phase = EncodePhase::postCommit;
            const auto post = state_->runtime->encode(request);
            ++state_->runtimePostCommitEncodeCount;
            if (!post.encoded) {
                state_->message =
                    "Human tendon/FEM Matter post-commit failed: " +
                    post.message;
                return false;
            }
        }
        id<MTLComputeCommandEncoder> commitEncoder =
            [command computeCommandEncoder];
        if (commitEncoder == nil) {
            state_->message =
                "Human FEM audit commit encoder is unavailable";
            return false;
        }
        if (!deferred) {
            [commitEncoder setComputePipelineState:
                state_->reactionAuditCommitPipeline];
            [commitEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [commitEncoder setBuffer:standStatuses offset:0u atIndex:1u];
            [commitEncoder setBuffer:state_->anchorReactionAuditBuffer
                              offset:0u atIndex:2u];
            [commitEncoder setBuffer:state_->anchorReactionAuditHistoryBuffer
                              offset:0u atIndex:3u];
            const NSUInteger reactionCommitWidth = std::min<NSUInteger>(
                count, std::min<NSUInteger>(
                    state_->reactionAuditCommitPipeline
                        .maxTotalThreadsPerThreadgroup,
                    256u));
            [commitEncoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    std::max<NSUInteger>(reactionCommitWidth, 1u), 1u, 1u)];
        }
        if (!state_->articularContactSamples.empty()) {
            [commitEncoder setComputePipelineState:
                state_->articularContactAuditCommitPipeline];
            [commitEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [commitEncoder setBuffer:standStatuses offset:0u atIndex:1u];
            [commitEncoder setBuffer:state_->articularContactAuditBuffer
                              offset:0u atIndex:2u];
            [commitEncoder setBuffer:state_->articularContactAuditHistoryBuffer
                              offset:0u atIndex:3u];
            const NSUInteger commitWidth = std::min<NSUInteger>(
                count, std::min<NSUInteger>(
                    state_->articularContactAuditCommitPipeline
                        .maxTotalThreadsPerThreadgroup,
                    256u));
            [commitEncoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    std::max<NSUInteger>(commitWidth, 1u), 1u, 1u)];
        }
        if (state_->articularContactPairCount != 0u) {
            [commitEncoder setComputePipelineState:
                state_->articularContactPairAuditCommitPipeline];
            [commitEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [commitEncoder setBuffer:standStatuses offset:0u atIndex:1u];
            [commitEncoder setBuffer:state_->articularContactPairAuditBuffer
                              offset:0u atIndex:2u];
            [commitEncoder setBuffer:
                state_->articularContactPairAuditHistoryBuffer
                              offset:0u atIndex:3u];
            const NSUInteger pairCommitWidth = std::min<NSUInteger>(
                count, std::min<NSUInteger>(
                    state_->articularContactPairAuditCommitPipeline
                        .maxTotalThreadsPerThreadgroup,
                    256u));
            [commitEncoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    std::max<NSUInteger>(pairCommitWidth, 1u), 1u, 1u)];
        }
        if (!state_->femBodyContactSamples.empty()) {
            [commitEncoder setComputePipelineState:
                state_->articularContactAuditCommitPipeline];
            [commitEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [commitEncoder setBuffer:standStatuses offset:0u atIndex:1u];
            [commitEncoder setBuffer:state_->femBodyContactAuditBuffer
                              offset:0u atIndex:2u];
            [commitEncoder setBuffer:state_->femBodyContactAuditHistoryBuffer
                              offset:0u atIndex:3u];
            const NSUInteger contactCommitWidth = std::min<NSUInteger>(
                count, std::min<NSUInteger>(
                    state_->articularContactAuditCommitPipeline
                        .maxTotalThreadsPerThreadgroup,
                    256u));
            [commitEncoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    std::max<NSUInteger>(contactCommitWidth, 1u), 1u, 1u)];
        }
        if (!state_->passiveLigaments.empty()) {
            [commitEncoder setComputePipelineState:
                state_->passiveLigamentAuditCommitPipeline];
            [commitEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [commitEncoder setBuffer:standStatuses offset:0u atIndex:1u];
            [commitEncoder setBuffer:state_->passiveLigamentAuditBuffer
                          offset:0u atIndex:2u];
            const NSUInteger passiveWidth = std::min<NSUInteger>(
                count, std::min<NSUInteger>(
                    state_->passiveLigamentAuditCommitPipeline
                        .maxTotalThreadsPerThreadgroup,
                    256u));
            [commitEncoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    std::max<NSUInteger>(passiveWidth, 1u), 1u, 1u)];
        }
        if (!state_->passiveRoutedBands.empty()) {
            [commitEncoder setComputePipelineState:
                state_->passiveRoutedBandAuditCommitPipeline];
            [commitEncoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0u];
            [commitEncoder setBuffer:standStatuses offset:0u atIndex:1u];
            [commitEncoder setBuffer:state_->passiveRoutedBandAuditBuffer
                          offset:0u atIndex:2u];
            const NSUInteger routedWidth = std::min<NSUInteger>(
                count, std::min<NSUInteger>(
                    state_->passiveRoutedBandAuditCommitPipeline
                        .maxTotalThreadsPerThreadgroup,
                    256u));
            [commitEncoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(
                    std::max<NSUInteger>(routedWidth, 1u), 1u, 1u)];
        }
        [commitEncoder endEncoding];
        ++state_->encodedPassCount;
        if (deferred) {
            state_->deferredExternalForcesAvailable = false;
            state_->deferredCommandBufferIdentity = 0u;
            state_->message = "deferred audit reconciliation encoded";
        } else {
            state_->message = "two-way transaction encoded";
        }
        return true;
    }
}

void NumiHumanTendonFEMLoadAdapter::abort(void* commandBuffer) noexcept {
    if (state_ != nullptr && state_->runtime != nullptr &&
        commandBuffer != nullptr) {
        ++state_->abortCount;
        if (state_->executionMode ==
                NumiHumanTendonFEMLoadExecutionMode::numanXDeferred) {
            if (state_->deferredCommandBufferIdentity ==
                    reinterpret_cast<std::uintptr_t>(commandBuffer)) {
                state_->deferredExternalForcesAvailable = false;
                state_->deferredCommandBufferIdentity = 0u;
            }
        } else {
            state_->runtime->cancel(commandBuffer);
        }
    }
}

bool NumiHumanTendonFEMLoadAdapter::borrowDeferredExternalForces(
    void* commandBuffer,
    const std::uint32_t stepIndex,
    const std::uint32_t environmentCount,
    NumiHumanTendonFEMDeferredExternalForceView& output
) noexcept {
    @autoreleasepool {
        if (state_ == nullptr || state_->runtime == nullptr ||
            state_->executionMode !=
                NumiHumanTendonFEMLoadExecutionMode::numanXDeferred ||
            !state_->deferredExternalForcesAvailable ||
            commandBuffer == nullptr ||
            state_->deferredCommandBufferIdentity !=
                reinterpret_cast<std::uintptr_t>(commandBuffer) ||
            stepIndex != state_->deferredStepIndex ||
            environmentCount != state_->environmentCount ||
            state_->fingerprint == 0u ||
            state_->runtimeDeviceProgramFingerprint == 0u ||
            state_->runtime->deviceProgramFingerprint() !=
                state_->runtimeDeviceProgramFingerprint ||
            state_->runtime->environmentCount() != state_->environmentCount ||
            state_->runtime->femNodeCount() != state_->nodeLoads.size() ||
            state_->runtime->femHumanAttachmentCount() !=
                state_->activeAnchorCount) {
            return false;
        }
        __unsafe_unretained id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)commandBuffer;
        __unsafe_unretained id<MTLBuffer> forces =
            state_->externalForceBuffer;
        __unsafe_unretained id<MTLBuffer> statuses =
            state_->worldStatusBuffer;
        const std::uint64_t forceElements =
            static_cast<std::uint64_t>(state_->environmentCount) *
            state_->nodeLoads.size();
        const std::uint64_t forceBytes =
            forceElements * sizeof(nm_float4);
        const std::uint64_t statusBytes =
            static_cast<std::uint64_t>(state_->environmentCount) *
            sizeof(MRMetalWorldStatusGPU);
        if (command == nil || command.commandQueue == nil ||
            command.status != MTLCommandBufferStatusNotEnqueued ||
            forces == nil || statuses == nil || state_->device == nil ||
            command.commandQueue.device.registryID !=
                state_->device.registryID ||
            forces.device.registryID != state_->device.registryID ||
            statuses.device.registryID != state_->device.registryID ||
            forces.gpuAddress == 0u || statuses.gpuAddress == 0u ||
            forces.length < forceBytes || statuses.length < statusBytes ||
            state_->nodeLoads.size() >
                std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        NumiHumanTendonFEMDeferredExternalForceView result;
        result.externalForces = (__bridge void*)forces;
        result.validationStatuses = (__bridge void*)statuses;
        result.externalForcesGPUAddress = forces.gpuAddress;
        result.validationStatusesGPUAddress = statuses.gpuAddress;
        result.externalForceElementCount = forceElements;
        result.validationStatusElementCount = state_->environmentCount;
        result.deviceRegistryID = state_->device.registryID;
        result.programFingerprint = state_->fingerprint;
        result.environmentCount = state_->environmentCount;
        result.femNodeCount = static_cast<std::uint32_t>(
            state_->nodeLoads.size());
        result.externalForceStride = result.femNodeCount;
        result.validationStatusStride = 1u;
        result.stepIndex = stepIndex;
        output = result;
        ++state_->deferredExternalForceBorrowCount;
        return true;
    }
}

NumiHumanTendonFEMDeferredLoadProgram
NumiHumanTendonFEMLoadAdapter::deferredProgram() noexcept {
    NumiHumanTendonFEMDeferredLoadProgram result{};
    if (state_ == nullptr || state_->executionMode !=
            NumiHumanTendonFEMLoadExecutionMode::numanXDeferred ||
        state_->fingerprint == 0u ||
        state_->runtimeDeviceProgramFingerprint == 0u) {
        return result;
    }
    result.context = this;
    result.encodePreSourceCorrections = [](
        void* context,
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    ) {
        return context != nullptr &&
            static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->
                encodePreDynamics(pass);
    };
    result.borrowExternalForces = [](
        void* context,
        void* commandBuffer,
        const std::uint32_t stepIndex,
        const std::uint32_t environmentCount,
        NumiHumanTendonFEMDeferredExternalForceView& output
    ) noexcept {
        return context != nullptr &&
            static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->
                borrowDeferredExternalForces(
                    commandBuffer, stepIndex, environmentCount, output);
    };
    result.fingerprint = state_->fingerprint;
    result.runtimeDeviceProgramFingerprint =
        state_->runtimeDeviceProgramFingerprint;
    result.environmentCount = state_->environmentCount;
    result.femNodeCount = static_cast<std::uint32_t>(
        state_->nodeLoads.size());
    result.activeAnchorCount = state_->activeAnchorCount;
    return result;
}

metalrobo::MetalNumiHumanTendonLoadProgram
NumiHumanTendonFEMLoadAdapter::program() noexcept {
    metalrobo::MetalNumiHumanTendonLoadProgram result{};
    if (state_ == nullptr || state_->fingerprint == 0u) return result;
    result.context = this;
    if (state_->executionMode ==
            NumiHumanTendonFEMLoadExecutionMode::numanXDeferred) {
        const auto deferred = deferredProgram();
        if (!deferred.valid()) return {};
        result.encodePreDynamics = deferred.encodePreSourceCorrections;
    } else {
        result.encodePreDynamics = [](
            void* context,
            const metalrobo::MetalNumiHumanTendonLoadPass& pass
        ) {
            return context != nullptr &&
                static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->
                    encodePreDynamics(pass);
        };
    }
    result.encodePostValidation = [](
        void* context,
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    ) {
        return context != nullptr &&
            static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->
                encodePostValidation(pass);
    };
    result.abort = [](void* context, void* commandBuffer) {
        if (context != nullptr) {
            static_cast<NumiHumanTendonFEMLoadAdapter*>(context)->abort(
                commandBuffer
            );
        }
    };
    result.fingerprint = state_->fingerprint;
    return result;
}

NumiHumanTendonFEMLoadAdapterSnapshotV1
NumiHumanTendonFEMLoadAdapter::snapshot() const noexcept {
    NumiHumanTendonFEMLoadAdapterSnapshotV1 result;
    try {
        if (state_ == nullptr || state_->runtime == nullptr ||
            state_->fingerprint == 0u || !state_->runtime->valid()) {
            result.message =
                "Numi Human tendon/FEM adapter is not initialized";
            return result;
        }
        result.adapterFingerprint = state_->fingerprint;
        result.runtimeDeviceProgramFingerprint =
            state_->runtime->deviceProgramFingerprint();
        result.environmentCount = state_->environmentCount;
        result.endpointCount = state_->endpointCount;
        result.femNodeCount = static_cast<std::uint32_t>(
            state_->nodeLoads.size());
        result.contactSampleCount = static_cast<std::uint32_t>(
            state_->contactSamples.size());
        result.femBodyContactSampleCount = static_cast<std::uint32_t>(
            state_->femBodyContactSamples.size());
        result.articularContactSampleCount = static_cast<std::uint32_t>(
            state_->articularContactSamples.size());
        result.articularContactPairCount =
            state_->articularContactPairCount;
        result.passiveLigamentCount = static_cast<std::uint32_t>(
            state_->passiveLigaments.size());
        result.passiveRoutedBandCount = static_cast<std::uint32_t>(
            state_->passiveRoutedBands.size());
        result.boundBodyPoseStride = state_->articularBodyPoseStride;
        result.encodedPassCount = state_->encodedPassCount;
        result.abortCount = state_->abortCount;
        result.articularAttemptedStepCount =
            state_->articularAttemptedStepCount;
        result.anchorReactionAcceptedHistory =
            state_->anchorReactionAcceptedHistory;
        result.femBodyContactAcceptedHistory =
            state_->femBodyContactAcceptedHistory;
        result.articularContactAcceptedHistory =
            state_->articularContactAcceptedHistory;
        result.articularContactPairForceAcceptedHistory =
            state_->articularContactPairForceAcceptedHistory;
        result.passiveLigamentAcceptedState =
            state_->passiveLigamentAcceptedState;
        result.passiveRoutedBandAcceptedState =
            state_->passiveRoutedBandAcceptedState;

        const auto readAuthority = [&](id<MTLBuffer> buffer, auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            const std::size_t expectedBytes = values.size() * sizeof(Value);
            if (buffer == nil) return true;
            if (buffer.contents == nullptr || buffer.length != expectedBytes) {
                return false;
            }
            if (expectedBytes != 0u) {
                std::memcpy(values.data(), buffer.contents, expectedBytes);
            }
            return true;
        };
        if (!readAuthority(
                state_->anchorReactionAuditHistoryBuffer,
                result.anchorReactionAcceptedHistory) ||
            !readAuthority(
                state_->femBodyContactAuditHistoryBuffer,
                result.femBodyContactAcceptedHistory) ||
            !readAuthority(
                state_->articularContactAuditHistoryBuffer,
                result.articularContactAcceptedHistory) ||
            !readAuthority(
                state_->articularContactPairAuditHistoryBuffer,
                result.articularContactPairForceAcceptedHistory) ||
            !readAuthority(
                state_->passiveLigamentAuditBuffer,
                result.passiveLigamentAcceptedState) ||
            !readAuthority(
                state_->passiveRoutedBandAuditBuffer,
                result.passiveRoutedBandAcceptedState)) {
            result.message =
                "Numi Human tendon/FEM adapter authority buffer is unavailable";
            return result;
        }
        result.available = true;
        std::string error;
        if (!digestNumiHumanTendonFEMLoadAdapterSnapshotV1(
                result, result.authoritySHA256, error)) {
            result.available = false;
            result.message = std::move(error);
            return result;
        }
        result.message = "authoritative adapter snapshot captured";
        return result;
    } catch (const std::exception& exception) {
        result.available = false;
        result.message = std::string(
            "Numi Human tendon/FEM adapter snapshot failed: ") +
            exception.what();
        return result;
    } catch (...) {
        result.available = false;
        result.message = "Numi Human tendon/FEM adapter snapshot failed";
        return result;
    }
}

NumiHumanTendonFEMLoadAdapterRestoreDiagnostics
NumiHumanTendonFEMLoadAdapter::restore(
    const NumiHumanTendonFEMLoadAdapterSnapshotV1& snapshotValue
) noexcept {
    NumiHumanTendonFEMLoadAdapterRestoreDiagnostics result;
    const auto reject = [&](std::string message) {
        result.message = std::move(message);
        return result;
    };
    try {
        if (state_ == nullptr || state_->runtime == nullptr ||
            state_->fingerprint == 0u || !state_->runtime->valid()) {
            return reject(
                "Numi Human tendon/FEM adapter restore requires an initialized adapter");
        }
        if (!snapshotValue.available ||
            snapshotValue.formatVersion !=
                kNumiHumanTendonFEMLoadAdapterSnapshotVersionV1) {
            return reject(
                "Numi Human tendon/FEM adapter restore rejected an unavailable or unsupported snapshot");
        }
        if (snapshotValue.adapterFingerprint != state_->fingerprint ||
            snapshotValue.runtimeDeviceProgramFingerprint !=
                state_->runtime->deviceProgramFingerprint() ||
            snapshotValue.environmentCount != state_->environmentCount ||
            snapshotValue.endpointCount != state_->endpointCount ||
            snapshotValue.femNodeCount != state_->nodeLoads.size() ||
            snapshotValue.contactSampleCount !=
                state_->contactSamples.size() ||
            snapshotValue.femBodyContactSampleCount !=
                state_->femBodyContactSamples.size() ||
            snapshotValue.articularContactSampleCount !=
                state_->articularContactSamples.size() ||
            snapshotValue.articularContactPairCount !=
                state_->articularContactPairCount ||
            snapshotValue.passiveLigamentCount !=
                state_->passiveLigaments.size() ||
            snapshotValue.passiveRoutedBandCount !=
                state_->passiveRoutedBands.size()) {
            return reject(
                "Numi Human tendon/FEM adapter restore identity does not match this adapter");
        }
        if (snapshotValue.articularAttemptedStepCount >
                NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS ||
            ((!state_->articularContactSamples.empty() ||
              !state_->femBodyContactSamples.empty()) &&
             state_->allocatedArticularBodyPoseStride != 0u &&
             snapshotValue.boundBodyPoseStride != 0u &&
             snapshotValue.boundBodyPoseStride !=
                state_->allocatedArticularBodyPoseStride)) {
            return reject(
                "Numi Human tendon/FEM adapter restore cursor or body-pose binding is invalid");
        }
        const std::size_t historyElements =
            static_cast<std::size_t>(state_->environmentCount) *
            NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS;
        const std::size_t femBodyHistoryElements =
            state_->femBodyContactSamples.empty() ? 0u : historyElements;
        const std::size_t articularHistoryElements =
            state_->articularContactSamples.empty() ? 0u : historyElements;
        const std::size_t articularPairHistoryElements =
            state_->articularContactPairCount == 0u
                ? 0u
                : historyElements *
                    (static_cast<std::size_t>(
                         state_->articularContactPairCount) + 1u);
        const std::size_t passiveLigamentElements =
            state_->passiveLigaments.empty()
                ? 0u : state_->environmentCount;
        const std::size_t passiveRoutedBandElements =
            state_->passiveRoutedBands.empty()
                ? 0u : state_->environmentCount;
        if (snapshotValue.anchorReactionAcceptedHistory.size() !=
                historyElements ||
            snapshotValue.femBodyContactAcceptedHistory.size() !=
                femBodyHistoryElements ||
            snapshotValue.articularContactAcceptedHistory.size() !=
                articularHistoryElements ||
            snapshotValue.articularContactPairForceAcceptedHistory.size() !=
                articularPairHistoryElements ||
            snapshotValue.passiveLigamentAcceptedState.size() !=
                passiveLigamentElements ||
            snapshotValue.passiveRoutedBandAcceptedState.size() !=
                passiveRoutedBandElements) {
            return reject(
                "Numi Human tendon/FEM adapter restore vector extent is invalid");
        }
        NumiHumanTendonFEMLoadAdapterDigest computedDigest{};
        std::string digestError;
        if (!digestNumiHumanTendonFEMLoadAdapterSnapshotV1(
                snapshotValue, computedDigest, digestError) ||
            computedDigest != snapshotValue.authoritySHA256) {
            return reject(digestError.empty()
                ? "Numi Human tendon/FEM adapter restore digest mismatch"
                : std::move(digestError));
        }

        // All fallible allocations and all buffer-shape checks complete before
        // the first state byte is changed.
        auto stagedAnchor = snapshotValue.anchorReactionAcceptedHistory;
        auto stagedFEMBody = snapshotValue.femBodyContactAcceptedHistory;
        auto stagedArticular = snapshotValue.articularContactAcceptedHistory;
        auto stagedArticularPairs =
            snapshotValue.articularContactPairForceAcceptedHistory;
        auto stagedLigament = snapshotValue.passiveLigamentAcceptedState;
        auto stagedRouted = snapshotValue.passiveRoutedBandAcceptedState;
        const auto writable = [&](id<MTLBuffer> buffer,
                                  const std::size_t expectedBytes) {
            if (buffer == nil) {
                return expectedBytes == 0u || state_->device == nil;
            }
            return buffer.contents != nullptr &&
                buffer.length == expectedBytes;
        };
        if ((state_->executionMode !=
                    NumiHumanTendonFEMLoadExecutionMode::numanXDeferred &&
             !writable(
                 state_->anchorReactionAuditHistoryBuffer,
                 stagedAnchor.size() * sizeof(stagedAnchor.front()))) ||
            !writable(
                state_->femBodyContactAuditHistoryBuffer,
                stagedFEMBody.size() *
                    sizeof(NMNumiHumanArticularContactAuditGPU)) ||
            !writable(
                state_->articularContactAuditHistoryBuffer,
                stagedArticular.size() *
                    sizeof(NMNumiHumanArticularContactAuditGPU)) ||
            !writable(
                state_->articularContactPairAuditHistoryBuffer,
                stagedArticularPairs.size() * sizeof(float)) ||
            !writable(
                state_->passiveLigamentAuditBuffer,
                stagedLigament.size() *
                    sizeof(NMNumiHumanPassiveLigamentAuditGPU)) ||
            !writable(
                state_->passiveRoutedBandAuditBuffer,
                stagedRouted.size() *
                    sizeof(NMNumiHumanPassiveRoutedBandAuditGPU))) {
            return reject(
                "Numi Human tendon/FEM adapter restore target buffer is unavailable");
        }
        const auto writeAuthority = [](id<MTLBuffer> buffer,
                                       const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            const std::size_t bytes = values.size() * sizeof(Value);
            if (buffer != nil && bytes != 0u) {
                std::memcpy(buffer.contents, values.data(), bytes);
            }
        };
        writeAuthority(
            state_->anchorReactionAuditHistoryBuffer, stagedAnchor);
        writeAuthority(
            state_->femBodyContactAuditHistoryBuffer, stagedFEMBody);
        writeAuthority(
            state_->articularContactAuditHistoryBuffer, stagedArticular);
        writeAuthority(
            state_->articularContactPairAuditHistoryBuffer,
            stagedArticularPairs);
        writeAuthority(state_->passiveLigamentAuditBuffer, stagedLigament);
        writeAuthority(state_->passiveRoutedBandAuditBuffer, stagedRouted);
        state_->anchorReactionAcceptedHistory = std::move(stagedAnchor);
        state_->femBodyContactAcceptedHistory = std::move(stagedFEMBody);
        state_->articularContactAcceptedHistory = std::move(stagedArticular);
        state_->articularContactPairForceAcceptedHistory =
            std::move(stagedArticularPairs);
        state_->passiveLigamentAcceptedState = std::move(stagedLigament);
        state_->passiveRoutedBandAcceptedState = std::move(stagedRouted);
        state_->articularBodyPoseStride =
            snapshotValue.boundBodyPoseStride;
        state_->encodedPassCount = snapshotValue.encodedPassCount;
        state_->abortCount = snapshotValue.abortCount;
        state_->articularAttemptedStepCount =
            snapshotValue.articularAttemptedStepCount;
        state_->message = "authoritative adapter state restored";
        result.restored = true;
        result.message = state_->message;
        return result;
    } catch (const std::exception& exception) {
        return reject(std::string(
            "Numi Human tendon/FEM adapter restore failed before mutation: ") +
            exception.what());
    } catch (...) {
        return reject(
            "Numi Human tendon/FEM adapter restore failed before mutation");
    }
}

NumiHumanTendonFEMLoadDiagnostics
NumiHumanTendonFEMLoadAdapter::diagnostics() const noexcept {
    NumiHumanTendonFEMLoadDiagnostics result{};
    if (state_ == nullptr) return result;
    result.initialized = state_->runtime != nullptr && state_->fingerprint != 0u;
    result.encodedPassCount = state_->encodedPassCount;
    result.abortCount = state_->abortCount;
    result.preSourceCorrectionEncodeCount =
        state_->preSourceCorrectionEncodeCount;
    result.deferredExternalForceBorrowCount =
        state_->deferredExternalForceBorrowCount;
    result.runtimePreDynamicsEncodeCount =
        state_->runtimePreDynamicsEncodeCount;
    result.runtimePostCommitEncodeCount =
        state_->runtimePostCommitEncodeCount;
    result.executionMode = state_->executionMode;
    result.fingerprint = state_->fingerprint;
    result.contactSampleCount = static_cast<std::uint32_t>(
        state_->contactSamples.size());
    result.femBodyContactSampleCount = static_cast<std::uint32_t>(
        state_->femBodyContactSamples.size());
    result.articularContactSampleCount = static_cast<std::uint32_t>(
        state_->articularContactSamples.size());
    result.articularContactPairCount =
        state_->articularContactPairCount;
    result.articularMechanicalSampleCount =
        state_->articularMechanicalSampleCount;
    result.articularInternalSameBodySampleCount =
        state_->articularInternalSameBodySampleCount;
    result.passiveLigamentCount = static_cast<std::uint32_t>(
        state_->passiveLigaments.size());
    if (state_->passiveLigamentAuditBuffer != nil) {
        const auto* audits =
            static_cast<const NMNumiHumanPassiveLigamentAuditGPU*>(
                state_->passiveLigamentAuditBuffer.contents);
        double forceX = 0.0;
        double forceY = 0.0;
        double forceZ = 0.0;
        double momentX = 0.0;
        double momentY = 0.0;
        double momentZ = 0.0;
        double minimumStretch = std::numeric_limits<double>::infinity();
        bool accepted = true;
        for (std::uint32_t environment = 0u;
             environment < state_->environmentCount; ++environment) {
            const auto& audit = audits[environment];
            const nm_float4 force = audit.forceResidualAndL1;
            const nm_float4 moment = audit.momentResidualAndMaximumTension;
            const nm_float4 stretch = audit.stretchCountAndAccepted;
            if (!finiteScale(force) || !finiteScale(moment) ||
                !finiteScale(stretch) || force.w < 0.0f || moment.w < 0.0f ||
                stretch.x <= 0.0f || stretch.y < stretch.x ||
                static_cast<std::uint32_t>(stretch.z) !=
                    state_->passiveLigaments.size() ||
                (stretch.w != 0.0f && stretch.w != 1.0f)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                result.passiveLigamentEndpointForceL1Newtons = nan;
                result.passiveLigamentMaximumTensionNewtons = nan;
                result.passiveLigamentMinimumEffectiveStretch = nan;
                result.passiveLigamentMaximumEffectiveStretch = nan;
                result.passiveLigamentForceResidualNewtons = nan;
                result.passiveLigamentMomentResidualNewtonMeters = nan;
                return result;
            }
            accepted = accepted && stretch.w == 1.0f;
            forceX += force.x;
            forceY += force.y;
            forceZ += force.z;
            momentX += moment.x;
            momentY += moment.y;
            momentZ += moment.z;
            result.passiveLigamentEndpointForceL1Newtons += force.w;
            result.passiveLigamentMaximumTensionNewtons = std::max(
                result.passiveLigamentMaximumTensionNewtons,
                static_cast<double>(moment.w));
            minimumStretch = std::min(
                minimumStretch, static_cast<double>(stretch.x));
            result.passiveLigamentMaximumEffectiveStretch = std::max(
                result.passiveLigamentMaximumEffectiveStretch,
                static_cast<double>(stretch.y));
        }
        result.passiveLigamentMinimumEffectiveStretch = minimumStretch;
        result.passiveLigamentForceResidualNewtons = std::sqrt(
            forceX * forceX + forceY * forceY + forceZ * forceZ);
        result.passiveLigamentMomentResidualNewtonMeters = std::sqrt(
            momentX * momentX + momentY * momentY + momentZ * momentZ);
        result.passiveLigamentLatestTransactionAccepted = accepted;
    }
    result.passiveRoutedBandCount = static_cast<std::uint32_t>(
        state_->passiveRoutedBands.size());
    if (state_->passiveRoutedBandAuditBuffer != nil) {
        const auto* audits =
            static_cast<const NMNumiHumanPassiveRoutedBandAuditGPU*>(
                state_->passiveRoutedBandAuditBuffer.contents);
        double forceX = 0.0;
        double forceY = 0.0;
        double forceZ = 0.0;
        double momentX = 0.0;
        double momentY = 0.0;
        double momentZ = 0.0;
        double minimumStrain = std::numeric_limits<double>::infinity();
        bool accepted = true;
        for (std::uint32_t environment = 0u;
             environment < state_->environmentCount; ++environment) {
            const auto& audit = audits[environment];
            const nm_float4 force = audit.forceResidualAndL1;
            const nm_float4 moment = audit.momentResidualAndMaximumTension;
            const nm_float4 strain = audit.strainCountAndAccepted;
            const nm_float4 energy = audit.energyAndMaximumExtension;
            if (!finiteScale(force) || !finiteScale(moment) ||
                !finiteScale(strain) || !finiteScale(energy) ||
                force.w < 0.0f || moment.w < 0.0f ||
                strain.y < strain.x ||
                static_cast<std::uint32_t>(strain.z) !=
                    state_->passiveRoutedBands.size() ||
                (strain.w != 0.0f && strain.w != 1.0f) ||
                energy.x < 0.0f || energy.y < 0.0f ||
                energy.z != 0.0f || energy.w != 0.0f) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                result.passiveRoutedBandEndpointForceL1Newtons = nan;
                result.passiveRoutedBandMaximumTensionNewtons = nan;
                result.passiveRoutedBandMinimumStrain = nan;
                result.passiveRoutedBandMaximumStrain = nan;
                result.passiveRoutedBandForceResidualNewtons = nan;
                result.passiveRoutedBandMomentResidualNewtonMeters = nan;
                result.passiveRoutedBandStoredEnergyJoules = nan;
                result.passiveRoutedBandMaximumExtensionMeters = nan;
                return result;
            }
            accepted = accepted && strain.w == 1.0f;
            forceX += force.x;
            forceY += force.y;
            forceZ += force.z;
            momentX += moment.x;
            momentY += moment.y;
            momentZ += moment.z;
            result.passiveRoutedBandEndpointForceL1Newtons += force.w;
            result.passiveRoutedBandMaximumTensionNewtons = std::max(
                result.passiveRoutedBandMaximumTensionNewtons,
                static_cast<double>(moment.w));
            minimumStrain = std::min(
                minimumStrain, static_cast<double>(strain.x));
            result.passiveRoutedBandMaximumStrain = std::max(
                result.passiveRoutedBandMaximumStrain,
                static_cast<double>(strain.y));
            result.passiveRoutedBandStoredEnergyJoules += energy.x;
            result.passiveRoutedBandMaximumExtensionMeters = std::max(
                result.passiveRoutedBandMaximumExtensionMeters,
                static_cast<double>(energy.y));
        }
        result.passiveRoutedBandMinimumStrain = minimumStrain;
        result.passiveRoutedBandForceResidualNewtons = std::sqrt(
            forceX * forceX + forceY * forceY + forceZ * forceZ);
        result.passiveRoutedBandMomentResidualNewtonMeters = std::sqrt(
            momentX * momentX + momentY * momentY + momentZ * momentZ);
        result.passiveRoutedBandLatestTransactionAccepted = accepted;
    }
    if (state_->externalForceAuditBuffer != nil) {
        const auto* audits = static_cast<const nm_float4*>(
            state_->externalForceAuditBuffer.contents);
        double resultantX = 0.0;
        double resultantY = 0.0;
        double resultantZ = 0.0;
        for (std::uint32_t environment = 0u;
             environment < state_->environmentCount; ++environment) {
            const nm_float4 audit = audits[environment];
            if (!finiteScale(audit) || audit.w < 0.0f) {
                result.assembledExternalForceL1Newtons =
                    std::numeric_limits<double>::quiet_NaN();
                result.assembledExternalForceResultantNewtons =
                    std::numeric_limits<double>::quiet_NaN();
                break;
            }
            resultantX += audit.x;
            resultantY += audit.y;
            resultantZ += audit.z;
            result.assembledExternalForceL1Newtons += audit.w;
            result.assembledExternalForceResultantNewtons = std::sqrt(
                resultantX * resultantX + resultantY * resultantY +
                resultantZ * resultantZ);
        }
    }
    if (state_->anchorReactionAuditBuffer != nil) {
        const auto* audits = static_cast<const nm_float4*>(
            state_->anchorReactionAuditBuffer.contents);
        double resultantX = 0.0;
        double resultantY = 0.0;
        double resultantZ = 0.0;
        for (std::uint32_t environment = 0u;
             environment < state_->environmentCount; ++environment) {
            const nm_float4 audit = audits[environment];
            if (!finiteScale(audit) || audit.w < 0.0f) {
                result.anchorReactionL1Newtons =
                    std::numeric_limits<double>::quiet_NaN();
                result.anchorReactionResultantNewtons =
                    std::numeric_limits<double>::quiet_NaN();
                break;
            }
            resultantX += audit.x;
            resultantY += audit.y;
            resultantZ += audit.z;
            result.anchorReactionL1Newtons += audit.w;
            result.anchorReactionResultantNewtons = std::sqrt(
                resultantX * resultantX + resultantY * resultantY +
                resultantZ * resultantZ);
        }
    }
    if (state_->anchorReactionAuditHistoryBuffer != nil) {
        const auto* history = static_cast<const nm_float4*>(
            state_->anchorReactionAuditHistoryBuffer.contents);
        const std::uint32_t attemptedSteps = std::min(
            state_->articularAttemptedStepCount,
            static_cast<std::uint32_t>(
                NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS));
        result.anchorReactionAuditedStepCount = attemptedSteps;
        double minimumL1 = std::numeric_limits<double>::infinity();
        for (std::uint32_t step = 0u; step < attemptedSteps; ++step) {
            double l1 = 0.0;
            double resultantX = 0.0;
            double resultantY = 0.0;
            double resultantZ = 0.0;
            for (std::uint32_t environment = 0u;
                 environment < state_->environmentCount; ++environment) {
                const nm_float4 audit = history[
                    environment *
                        NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS + step];
                if (!finiteScale(audit) || audit.w < 0.0f) {
                    const double nan =
                        std::numeric_limits<double>::quiet_NaN();
                    result.anchorReactionTrajectoryMinimumL1Newtons = nan;
                    result.anchorReactionTrajectoryMaximumL1Newtons = nan;
                    result.anchorReactionTrajectoryMaximumResultantNewtons = nan;
                    return result;
                }
                resultantX += audit.x;
                resultantY += audit.y;
                resultantZ += audit.z;
                l1 += audit.w;
            }
            minimumL1 = std::min(minimumL1, l1);
            result.anchorReactionTrajectoryMaximumL1Newtons = std::max(
                result.anchorReactionTrajectoryMaximumL1Newtons, l1);
            result.anchorReactionTrajectoryMaximumResultantNewtons = std::max(
                result.anchorReactionTrajectoryMaximumResultantNewtons,
                std::sqrt(resultantX * resultantX +
                          resultantY * resultantY +
                          resultantZ * resultantZ));
        }
        if (attemptedSteps > 0u) {
            result.anchorReactionTrajectoryMinimumL1Newtons = minimumL1;
        }
    }
    if (state_->femBodyContactAuditHistoryBuffer != nil) {
        const auto* history =
            static_cast<const NMNumiHumanArticularContactAuditGPU*>(
                state_->femBodyContactAuditHistoryBuffer.contents);
        const std::uint32_t attemptedSteps = std::min(
            state_->articularAttemptedStepCount,
            NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS);
        for (std::uint32_t step = 0u; step < attemptedSteps; ++step) {
            double forceX = 0.0;
            double forceY = 0.0;
            double forceZ = 0.0;
            double momentX = 0.0;
            double momentY = 0.0;
            double momentZ = 0.0;
            double maximumTangentialSlip = 0.0;
            double maximumPressure = 0.0;
            double normalForce = 0.0;
            double contactArea = 0.0;
            double storedEnergy = 0.0;
            double maximumNormalStrain = 0.0;
            double maximumClosure = 0.0;
            std::uint32_t closedSamples = 0u;
            bool accepted = true;
            bool malformed = false;
            for (std::uint32_t environment = 0u;
                 environment < state_->environmentCount; ++environment) {
                const auto& audit = history[
                    environment *
                        NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS +
                    step];
                const nm_float4 force = audit.forceResidualAndL1;
                const nm_float4 moment =
                    audit.momentResidualAndMaximumPressure;
                const nm_float4 contact = audit.normalForceAreaAndCounts;
                const nm_float4 energy =
                    audit.energyStrainClosureAndAccepted;
                if (energy.w == 0.0f) {
                    accepted = false;
                    break;
                }
                malformed = !finiteScale(force) || !finiteScale(moment) ||
                    !finiteScale(contact) || !finiteScale(energy) ||
                    energy.w != 1.0f || force.w < 0.0f || moment.w < 0.0f ||
                    contact.x < 0.0f || contact.y < 0.0f || contact.z < 0.0f ||
                    static_cast<std::uint32_t>(contact.w) !=
                        state_->femBodyContactSamples.size() ||
                    energy.x < 0.0f || energy.y < 0.0f || energy.z < 0.0f;
                if (malformed) break;
                forceX += force.x;
                forceY += force.y;
                forceZ += force.z;
                momentX += moment.x;
                momentY += moment.y;
                momentZ += moment.z;
                maximumTangentialSlip = std::max(
                    maximumTangentialSlip, static_cast<double>(force.w));
                maximumPressure = std::max(
                    maximumPressure, static_cast<double>(moment.w));
                normalForce += contact.x;
                contactArea += contact.y;
                closedSamples += static_cast<std::uint32_t>(contact.z);
                storedEnergy += energy.x;
                maximumNormalStrain = std::max(
                    maximumNormalStrain, static_cast<double>(energy.y));
                maximumClosure = std::max(
                    maximumClosure, static_cast<double>(energy.z));
            }
            if (malformed) {
                const double nan =
                    std::numeric_limits<double>::quiet_NaN();
                result.femBodyContactAreaSquareMeters = nan;
                result.femBodyContactNormalForceNewtons = nan;
                result.femBodyContactMaximumPressurePascals = nan;
                result.femBodyContactForceResidualNewtons = nan;
                result.femBodyContactMomentResidualNewtonMeters = nan;
                result.femBodyContactStoredEnergyJoules = nan;
                result.femBodyContactMaximumNormalStrain = nan;
                result.femBodyContactMaximumClosureMeters = nan;
                result.femBodyContactMaximumTangentialSlipMeters = nan;
                result.femBodyContactTrajectoryMinimumNormalForceNewtons = nan;
                result.femBodyContactTrajectoryMaximumNormalForceNewtons = nan;
                result.femBodyContactTrajectoryMaximumForceResidualNewtons = nan;
                result.femBodyContactTrajectoryMaximumMomentResidualNewtonMeters = nan;
                result.femBodyContactTrajectoryMaximumTangentialSlipMeters = nan;
                break;
            }
            if (!accepted) break;
            const double forceResidual = std::sqrt(
                forceX * forceX + forceY * forceY + forceZ * forceZ);
            const double momentResidual = std::sqrt(
                momentX * momentX + momentY * momentY + momentZ * momentZ);
            result.femBodyContactClosedSampleCount = closedSamples;
            result.femBodyContactAreaSquareMeters = contactArea;
            result.femBodyContactNormalForceNewtons = normalForce;
            result.femBodyContactMaximumPressurePascals = maximumPressure;
            result.femBodyContactForceResidualNewtons = forceResidual;
            result.femBodyContactMomentResidualNewtonMeters = momentResidual;
            result.femBodyContactStoredEnergyJoules = storedEnergy;
            result.femBodyContactMaximumNormalStrain = maximumNormalStrain;
            result.femBodyContactMaximumClosureMeters = maximumClosure;
            result.femBodyContactMaximumTangentialSlipMeters =
                maximumTangentialSlip;
            if (result.femBodyContactAuditedStepCount == 0u) {
                result.femBodyContactTrajectoryMinimumNormalForceNewtons =
                    normalForce;
            } else {
                result.femBodyContactTrajectoryMinimumNormalForceNewtons =
                    std::min(
                        result.femBodyContactTrajectoryMinimumNormalForceNewtons,
                        normalForce);
            }
            ++result.femBodyContactAuditedStepCount;
            result.femBodyContactTrajectoryMaximumNormalForceNewtons =
                std::max(
                    result.femBodyContactTrajectoryMaximumNormalForceNewtons,
                    normalForce);
            result.femBodyContactTrajectoryMaximumForceResidualNewtons =
                std::max(
                    result.femBodyContactTrajectoryMaximumForceResidualNewtons,
                    forceResidual);
            result.femBodyContactTrajectoryMaximumMomentResidualNewtonMeters =
                std::max(
                    result.femBodyContactTrajectoryMaximumMomentResidualNewtonMeters,
                    momentResidual);
            result.femBodyContactTrajectoryMaximumTangentialSlipMeters =
                std::max(
                    result.femBodyContactTrajectoryMaximumTangentialSlipMeters,
                    maximumTangentialSlip);
        }
    }
    if (state_->articularContactAuditHistoryBuffer != nil) {
        const auto* history =
            static_cast<const NMNumiHumanArticularContactAuditGPU*>(
                state_->articularContactAuditHistoryBuffer.contents);
        const auto poisonArticularDiagnostics = [&]() {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            result.articularContactAreaSquareMeters = nan;
            result.articularNormalForceNewtons = nan;
            result.articularMaximumPressurePascals = nan;
            result.articularBodyForceL1Newtons = nan;
            result.articularForceResidualNewtons = nan;
            result.articularMomentResidualNewtonMeters = nan;
            result.articularStoredEnergyJoules = nan;
            result.articularMaximumNormalStrain = nan;
            result.articularMaximumClosureMeters = nan;
            result.articularTrajectoryMinimumNormalForceNewtons = nan;
            result.articularTrajectoryMaximumNormalForceNewtons = nan;
            result.articularTrajectoryMaximumPressurePascals = nan;
            result.articularTrajectoryMaximumStoredEnergyJoules = nan;
            result.articularTrajectoryMaximumNormalStrain = nan;
            result.articularTrajectoryMaximumClosureMeters = nan;
            result.articularTrajectoryMaximumForceResidualNewtons = nan;
            result.articularTrajectoryMaximumMomentResidualNewtonMeters = nan;
        };
        const std::uint32_t attemptedSteps = std::min(
            state_->articularAttemptedStepCount,
            NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS);
        for (std::uint32_t step = 0u; step < attemptedSteps; ++step) {
            double forceX = 0.0;
            double forceY = 0.0;
            double forceZ = 0.0;
            double momentX = 0.0;
            double momentY = 0.0;
            double momentZ = 0.0;
            double bodyForceL1 = 0.0;
            double maximumPressure = 0.0;
            double normalForce = 0.0;
            double contactArea = 0.0;
            double storedEnergy = 0.0;
            double maximumNormalStrain = 0.0;
            double maximumClosure = 0.0;
            std::uint32_t closedSamples = 0u;
            bool accepted = true;
            bool malformed = false;
            for (std::uint32_t environment = 0u;
                 environment < state_->environmentCount; ++environment) {
                const auto& audit = history[
                    environment *
                        NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS +
                    step];
                const nm_float4 force = audit.forceResidualAndL1;
                const nm_float4 moment = audit.momentResidualAndMaximumPressure;
                const nm_float4 contact = audit.normalForceAreaAndCounts;
                const nm_float4 energy =
                    audit.energyStrainClosureAndAccepted;
                if (energy.w == 0.0f) {
                    accepted = false;
                    break;
                }
                malformed = !finiteScale(force) || !finiteScale(moment) ||
                    !finiteScale(contact) || !finiteScale(energy) ||
                    energy.w != 1.0f || force.w < 0.0f || moment.w < 0.0f ||
                    contact.x < 0.0f || contact.y < 0.0f || contact.z < 0.0f ||
                    contact.w < 0.0f || energy.x < 0.0f || energy.y < 0.0f ||
                    energy.z < 0.0f ||
                    static_cast<std::uint32_t>(contact.w) !=
                        state_->articularInternalSameBodySampleCount;
                if (malformed) break;
                forceX += force.x;
                forceY += force.y;
                forceZ += force.z;
                momentX += moment.x;
                momentY += moment.y;
                momentZ += moment.z;
                bodyForceL1 += force.w;
                maximumPressure = std::max(
                    maximumPressure, static_cast<double>(moment.w));
                normalForce += contact.x;
                contactArea += contact.y;
                closedSamples += static_cast<std::uint32_t>(contact.z);
                storedEnergy += energy.x;
                maximumNormalStrain = std::max(
                    maximumNormalStrain, static_cast<double>(energy.y));
                maximumClosure = std::max(
                    maximumClosure, static_cast<double>(energy.z));
            }
            if (malformed) {
                poisonArticularDiagnostics();
                break;
            }
            // Only a contiguous prefix of wholly accepted multi-environment
            // steps is trajectory evidence. A rejected step terminates it.
            if (!accepted) break;
            const double forceResidual = std::sqrt(
                forceX * forceX + forceY * forceY + forceZ * forceZ);
            const double momentResidual = std::sqrt(
                momentX * momentX + momentY * momentY + momentZ * momentZ);
            result.articularClosedSampleCount = closedSamples;
            result.articularContactAreaSquareMeters = contactArea;
            result.articularNormalForceNewtons = normalForce;
            result.articularMaximumPressurePascals = maximumPressure;
            result.articularBodyForceL1Newtons = bodyForceL1;
            result.articularForceResidualNewtons = forceResidual;
            result.articularMomentResidualNewtonMeters = momentResidual;
            result.articularStoredEnergyJoules = storedEnergy;
            result.articularMaximumNormalStrain = maximumNormalStrain;
            result.articularMaximumClosureMeters = maximumClosure;
            if (result.articularAuditedStepCount == 0u) {
                result.articularTrajectoryMinimumClosedSampleCount =
                    closedSamples;
                result.articularTrajectoryMinimumNormalForceNewtons =
                    normalForce;
            } else {
                result.articularTrajectoryMinimumClosedSampleCount = std::min(
                    result.articularTrajectoryMinimumClosedSampleCount,
                    closedSamples);
                result.articularTrajectoryMinimumNormalForceNewtons = std::min(
                    result.articularTrajectoryMinimumNormalForceNewtons,
                    normalForce);
            }
            ++result.articularAuditedStepCount;
            result.articularTrajectoryMaximumClosedSampleCount = std::max(
                result.articularTrajectoryMaximumClosedSampleCount,
                closedSamples);
            result.articularTrajectoryMaximumNormalForceNewtons = std::max(
                result.articularTrajectoryMaximumNormalForceNewtons,
                normalForce);
            result.articularTrajectoryMaximumPressurePascals = std::max(
                result.articularTrajectoryMaximumPressurePascals,
                maximumPressure);
            result.articularTrajectoryMaximumStoredEnergyJoules = std::max(
                result.articularTrajectoryMaximumStoredEnergyJoules,
                storedEnergy);
            result.articularTrajectoryMaximumNormalStrain = std::max(
                result.articularTrajectoryMaximumNormalStrain,
                maximumNormalStrain);
            result.articularTrajectoryMaximumClosureMeters = std::max(
                result.articularTrajectoryMaximumClosureMeters,
                maximumClosure);
            result.articularTrajectoryMaximumForceResidualNewtons = std::max(
                result.articularTrajectoryMaximumForceResidualNewtons,
                forceResidual);
            result.articularTrajectoryMaximumMomentResidualNewtonMeters =
                std::max(
                    result.articularTrajectoryMaximumMomentResidualNewtonMeters,
                    momentResidual);
        }
    }
    result.message = state_->message;
    return result;
}

} // namespace numi::matter
