#pragma once

// Same-build completion-boundary RuntimeStateSnapshot persistence for
// checkpoint-linked shipping protocols. Runtime::restore remains the final
// authority and requires the exact cooked device-program fingerprint.

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace numi::cardboard {
namespace shipping_snapshot_detail {

inline constexpr std::array<char, 8> kMagic{'N', 'M', 'C', 'B', 'S', 'N', 'P', '1'};
inline constexpr std::uint32_t kVersion = 1u;
inline constexpr std::uint32_t kEndianMarker = 0x01020304u;

template<class T>
void writePod(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) throw std::runtime_error("shipping snapshot write failed");
}

template<class T>
T readPod(std::istream& input) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("shipping snapshot is truncated");
    return value;
}

inline std::uint64_t remainingInputBytes(std::istream& input) {
    const auto current = input.tellg();
    if (current < std::streampos(0))
        throw std::runtime_error("shipping snapshot stream is not seekable");
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    input.seekg(current);
    if (!input || end < current)
        throw std::runtime_error("shipping snapshot stream bounds are unavailable");
    return static_cast<std::uint64_t>(end - current);
}

inline void writeString(std::ostream& output, const std::string& value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("shipping snapshot string is too large");
    writePod(output, static_cast<std::uint32_t>(value.size()));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output) throw std::runtime_error("shipping snapshot string write failed");
}

inline std::string readString(std::istream& input, const std::uint64_t maximumBytes) {
    const std::uint32_t count = readPod<std::uint32_t>(input);
    if (count > maximumBytes || count > remainingInputBytes(input))
        throw std::length_error("shipping snapshot string exceeds its bound");
    std::string value(count, '\0');
    input.read(value.data(), static_cast<std::streamsize>(count));
    if (!input) throw std::runtime_error("shipping snapshot string is truncated");
    return value;
}

template<class T>
void writeVector(std::ostream& output, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    writePod(output, static_cast<std::uint64_t>(values.size()));
    if (values.empty()) return;
    if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::length_error("shipping snapshot vector size overflow");
    const std::size_t bytes = values.size() * sizeof(T);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::length_error("shipping snapshot vector exceeds stream limits");
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(bytes));
    if (!output) throw std::runtime_error("shipping snapshot vector write failed");
}

template<class T>
std::vector<T> readVector(std::istream& input, const std::uint64_t maximumBytes) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t count = readPod<std::uint64_t>(input);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
        count * sizeof(T) > maximumBytes ||
        count * sizeof(T) > remainingInputBytes(input) ||
        count * sizeof(T) > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamsize>::max()))
        throw std::length_error("shipping snapshot vector exceeds its bound");
    std::vector<T> values(static_cast<std::size_t>(count));
    if (!values.empty()) {
        input.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!input) throw std::runtime_error("shipping snapshot vector is truncated");
    }
    return values;
}

inline void validateAcceptedSnapshot(const numi::matter::RuntimeStateSnapshot& state) {
    if (!state.available || state.statuses.empty() ||
        state.statuses.size() != state.solverCertificates.size())
        throw std::runtime_error("shipping snapshot lacks per-environment accepted certificates");
    for (std::size_t environment = 0u; environment < state.statuses.size(); ++environment) {
        const auto& status = state.statuses[environment];
        const auto& certificate = state.solverCertificates[environment];
        if (status.code != NM_STATUS_SUCCESS || !(certificate.validity.w > 0.5f) ||
            !std::isfinite(certificate.nonlinear.x) ||
            !std::isfinite(certificate.nonlinear.y) ||
            !std::isfinite(certificate.nonlinear.z) ||
            !std::isfinite(certificate.nonlinear.w))
            throw std::runtime_error("shipping snapshot contains a failed or uncertified environment");
    }
}

inline std::size_t shippingSnapshotExpectedCount(
    const std::uint32_t environmentCount,
    const std::size_t perEnvironmentCount,
    const char* label) {
    if (perEnvironmentCount != 0u &&
        environmentCount > std::numeric_limits<std::size_t>::max() /
            perEnvironmentCount)
        throw std::length_error(std::string("shipping snapshot expected count overflow: ") + label);
    return static_cast<std::size_t>(environmentCount) * perEnvironmentCount;
}

inline std::size_t shippingSnapshotProduct(
    const std::size_t first,
    const std::size_t second,
    const char* label) {
    if (second != 0u && first > std::numeric_limits<std::size_t>::max() / second)
        throw std::length_error(std::string("shipping snapshot product overflow: ") + label);
    return first * second;
}

template<class T>
void requireShippingSnapshotSize(
    const std::vector<T>& values,
    const std::size_t expected,
    const char* label) {
    if (values.size() != expected)
        throw std::runtime_error(std::string("shipping checkpoint is incomplete: ") + label);
}

// Runtime buffers preserve a typed sentinel even when the logical arena is
// empty, and pad sub-16-byte payloads to the Metal binding minimum. Snapshot()
// reads the full physical allocation for exactArena-backed fields, while
// Runtime::restore requires byte-exact arena size. This helper mirrors
// runtime.mm checkedBytes() rather than accepting arbitrary extra records.
template<class T>
std::size_t shippingSnapshotExpectedPhysicalRecordCount(
    const std::size_t logicalCount,
    const char* label) {
    static_assert(std::is_trivially_copyable_v<T>);
    constexpr std::size_t elementSize = sizeof(T);
    if (logicalCount > std::numeric_limits<std::size_t>::max() / elementSize)
        throw std::length_error(
            std::string("shipping snapshot physical byte count overflow: ") + label);
    const std::size_t logicalBytes = logicalCount * elementSize;
    const std::size_t physicalBytes =
        std::max({logicalBytes, elementSize, std::size_t{16u}});
    if (physicalBytes % elementSize != 0u)
        throw std::length_error(
            std::string("shipping snapshot sentinel is not record-aligned: ") + label);
    return physicalBytes / elementSize;
}

template<class T>
void requireShippingSnapshotPhysicalSize(
    const std::vector<T>& values,
    const std::size_t logicalCount,
    const char* label) {
    requireShippingSnapshotSize(
        values, shippingSnapshotExpectedPhysicalRecordCount<T>(logicalCount, label),
        label);
}

// Runtime::restore is the final device-side layout authority, but a shipping
// checkpoint should never be published as accepted when that later restore
// will reject it for missing arenas. Validate the state that carries physical
// history against the exact compiled world before the file is written.
inline void validateShippingCheckpointShape(
    const numi::matter::RuntimeStateSnapshot& state,
    const numi::matter::CompiledWorld& world,
    const std::uint32_t environmentCount,
    const bool captureDiagnostics) {
    const auto& dispatch = world.dispatch;
    if (environmentCount == 0u || dispatch.environmentCount != environmentCount ||
        dispatch.femNodeCount != world.fem.nodes.size() ||
        dispatch.tetrahedronCount != world.fem.tetrahedra.size() ||
        dispatch.rigidProxyCount != world.contact.rigidProxies.size() ||
        dispatch.contactPairCount != world.contact.pairs.size() ||
        dispatch.cohesiveFaceCount != world.fem.cohesiveFaces.size() ||
        dispatch.punctureChannelCount != world.fem.punctureChannels.size() ||
        dispatch.objectCount != world.objects.size() ||
        state.materialStateStride != dispatch.materialStateStride)
        throw std::runtime_error("shipping checkpoint compiled layout is internally inconsistent");

    const auto perEnvironment = [&](const std::size_t count) {
        return shippingSnapshotExpectedCount(environmentCount, count, "environment arena");
    };
    const auto exact = [&](const auto& values, const std::size_t expected,
                           const char* label) {
        requireShippingSnapshotSize(values, expected, label);
    };
    const auto exactPhysical = [&](const auto& values,
                                   const std::size_t logicalCount,
                                   const char* label) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        requireShippingSnapshotPhysicalSize<Value>(values, logicalCount, label);
    };
    exact(state.statuses, environmentCount, "environment status arena");
    exactPhysical(state.solverCertificates,
        perEnvironment(world.objects.size()), "solver certificate arena");
    exactPhysical(state.femNodes, perEnvironment(world.fem.nodes.size()), "FEM node state");
    exactPhysical(state.femFields, perEnvironment(world.fem.fields.size()), "FEM field state");
    exactPhysical(state.femMaterialState,
        perEnvironment(shippingSnapshotProduct(
            world.fem.tetrahedra.size(),
            static_cast<std::size_t>(dispatch.materialStateStride),
            "FEM material state")),
        "FEM material history");
    exactPhysical(state.femTopologyNodes,
        perEnvironment(world.fem.topologyNodes.size()), "FEM topology node history");
    exactPhysical(state.femTopologyTetrahedra,
        perEnvironment(world.fem.tetrahedra.size()), "FEM topology tetrahedron history");
    exactPhysical(state.cohesiveFaces,
        perEnvironment(world.fem.cohesiveFaces.size()), "cohesive interface history");
    exactPhysical(state.punctureChannels,
        perEnvironment(world.fem.punctureChannels.size()), "puncture channel history");
    exactPhysical(state.topologyStates,
        perEnvironment(world.objects.size()), "topology generation history");
    exactPhysical(state.particles,
        perEnvironment(world.mpm.particles.size()), "particle state");
    exactPhysical(state.particleMaterialState,
        perEnvironment(shippingSnapshotProduct(
            world.mpm.particles.size(),
            static_cast<std::size_t>(dispatch.materialStateStride),
            "particle material state")),
        "particle material history");
    exactPhysical(state.environmentParameters,
        perEnvironment(world.parameters.size()), "environment parameter state");
    exact(state.vascularState,
        perEnvironment(world.vascular.layout.ranges.z), "vascular state");
    exact(state.vascularClock,
        world.vascular.layout.ranges.z == 0u ? 0u : environmentCount,
        "vascular clock state");
    exact(state.rigidStates,
        perEnvironment(world.contact.rigidProxies.size()), "rigid proxy state");
    exact(state.rigidGeneralizedCandidate,
        perEnvironment(dispatch.rigidGeneralizedCapacity), "rigid generalized state");
    exactPhysical(state.reactions,
        perEnvironment(world.contact.rigidProxies.size()), "rigid reaction state");
    exact(state.contactHistories,
        perEnvironment(world.contact.pairs.size()), "rigid contact history");
    exact(state.deformableContactHistories,
        perEnvironment(dispatch.deformableContactCapacity), "deformable contact history");
    exactPhysical(state.adaptive,
        perEnvironment(world.adaptive.size()), "adaptive object state");
    exactPhysical(state.schedulers,
        perEnvironment(world.schedulers.size()), "scheduler state");
    exactPhysical(state.learnedWeights,
        world.learnedWeights.size(), "learned material state");
    exactPhysical(state.identification,
        world.identification.size(), "identification state");
    exact(state.mpmActiveNodeIndices,
        perEnvironment(dispatch.mpmActiveNodeCapacity), "MPM active-node index state");
    exact(state.mpmNodeToActive,
        perEnvironment(dispatch.gridNodeCount), "MPM node-to-active state");
    exact(state.mpmActiveNodeCounts,
        environmentCount, "MPM active-node count state");
    if (captureDiagnostics) {
        exact(state.contactSamples,
            perEnvironment(world.contact.pairs.size()), "contact diagnostic sample state");
        exactPhysical(state.deformableContactFailures,
            environmentCount, "deformable contact failure diagnostics");
    } else if (!state.contactSamples.empty() ||
               !state.deformableContactFailures.empty()) {
        throw std::runtime_error("shipping checkpoint contains diagnostics not enabled by its runtime");
    }
    // These optional facilities are not authored by the shipping driver. A
    // populated arena here would require a corresponding source/runtime
    // identity in the paired checkpoint contract.
    if (!state.sutureProxyEdges.empty() || !state.humanSupportHistories.empty() ||
        !state.humanSupportConsequences.empty())
        throw std::runtime_error("shipping checkpoint contains an unbound optional history arena");
}

inline void writeScalarFields(std::ostream& output,
                              const numi::matter::RuntimeStateSnapshot& state) {
    writePod(output, state.sourcePhysicsFingerprint);
    writePod(output, state.deviceProgramFingerprint);
    writePod(output, state.controlStep);
    writePod(output, state.physicsSubstep);
    writePod(output, state.identificationGeneration);
    writePod(output, state.identificationCheckpoint);
    writePod(output, static_cast<std::uint8_t>(state.identificationAdvanced));
    writePod(output, state.sutureProxyBindingRevision);
    writePod(output, state.coupledTimestepMultiplier);
    writePod(output, state.coupledTimestepDivisor);
    writePod(output, state.fgmresIterationBudgetOverride);
    writePod(output, state.newtonIterationBudgetOverride);
    writePod(output, state.allocationGeneration);
    writePod(output, state.learnedWeightRevision);
    writePod(output, state.materialStateStride);
}

inline void readScalarFields(std::istream& input,
                             numi::matter::RuntimeStateSnapshot& state) {
    state.sourcePhysicsFingerprint = readPod<std::uint64_t>(input);
    state.deviceProgramFingerprint = readPod<std::uint64_t>(input);
    state.controlStep = readPod<std::uint32_t>(input);
    state.physicsSubstep = readPod<std::uint32_t>(input);
    state.identificationGeneration = readPod<std::uint32_t>(input);
    state.identificationCheckpoint = readPod<std::uint32_t>(input);
    const std::uint8_t advanced = readPod<std::uint8_t>(input);
    if (advanced > 1u) throw std::runtime_error("shipping snapshot has invalid boolean");
    state.identificationAdvanced = advanced != 0u;
    state.sutureProxyBindingRevision = readPod<std::uint64_t>(input);
    state.coupledTimestepMultiplier = readPod<std::uint32_t>(input);
    state.coupledTimestepDivisor = readPod<std::uint32_t>(input);
    state.fgmresIterationBudgetOverride = readPod<std::uint32_t>(input);
    state.newtonIterationBudgetOverride = readPod<std::uint32_t>(input);
    state.allocationGeneration = readPod<std::uint32_t>(input);
    state.learnedWeightRevision = readPod<std::uint32_t>(input);
    state.materialStateStride = readPod<std::uint32_t>(input);
}

#define NUMI_CARDBOARD_SNAPSHOT_VECTORS(X) \
    X(sutureProxyEdges) \
    X(particles) \
    X(femNodes) \
    X(femFields) \
    X(vascularState) \
    X(vascularClock) \
    X(femTopologyNodes) \
    X(femTopologyTetrahedra) \
    X(cohesiveFaces) \
    X(punctureChannels) \
    X(topologyStates) \
    X(statuses) \
    X(solverCertificates) \
    X(mpmActiveNodeIndices) \
    X(mpmNodeToActive) \
    X(mpmActiveNodeCounts) \
    X(rigidGeneralizedCandidate) \
    X(learnedWeights) \
    X(adaptive) \
    X(schedulers) \
    X(reactions) \
    X(rigidStates) \
    X(contactSamples) \
    X(deformableContactFailures) \
    X(contactHistories) \
    X(humanSupportHistories) \
    X(humanSupportConsequences) \
    X(deformableContactHistories) \
    X(particleMaterialState) \
    X(femMaterialState) \
    X(identification) \
    X(environmentParameters)

inline void writeSnapshotPayload(std::ostream& output,
                                 const numi::matter::RuntimeStateSnapshot& state) {
    writeString(output, state.message);
    writeScalarFields(output, state);
#define WRITE_FIELD(field) writeVector(output, state.field);
    NUMI_CARDBOARD_SNAPSHOT_VECTORS(WRITE_FIELD)
#undef WRITE_FIELD
}

class SnapshotCountingBuffer final : public std::streambuf {
public:
    std::uint64_t bytes() const noexcept { return bytes_; }
    bool overflowed() const noexcept { return overflowed_; }

protected:
    std::streamsize xsputn(const char*, const std::streamsize count) override {
        if (count < 0 || !add(static_cast<std::uint64_t>(count))) return 0;
        return count;
    }

    int_type overflow(const int_type character) override {
        if (traits_type::eq_int_type(character, traits_type::eof()))
            return traits_type::not_eof(character);
        if (!add(1u)) return traits_type::eof();
        return character;
    }

private:
    bool add(const std::uint64_t count) noexcept {
        if (count > std::numeric_limits<std::uint64_t>::max() - bytes_) {
            overflowed_ = true;
            return false;
        }
        bytes_ += count;
        return true;
    }

    std::uint64_t bytes_ = 0u;
    bool overflowed_ = false;
};

inline std::uint64_t serializedAcceptedSnapshotBytes(
    const numi::matter::RuntimeStateSnapshot& state) {
    validateAcceptedSnapshot(state);
    SnapshotCountingBuffer buffer;
    std::ostream output(&buffer);
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    writePod(output, kVersion);
    writePod(output, kEndianMarker);
    writePod(output, static_cast<std::uint32_t>(
        sizeof(numi::matter::RuntimeStateSnapshot)));
    writePod(output, static_cast<std::uint8_t>(state.available));
    writeSnapshotPayload(output, state);
    if (!output || buffer.overflowed())
        throw std::length_error("shipping snapshot serialized size overflow");
    return buffer.bytes();
}

inline void readSnapshotVectors(std::istream& input,
                                numi::matter::RuntimeStateSnapshot& state,
                                const std::uint64_t maximumBytes) {
#define READ_FIELD(field) state.field = readVector<decltype(state.field)::value_type>(input, maximumBytes);
    NUMI_CARDBOARD_SNAPSHOT_VECTORS(READ_FIELD)
#undef READ_FIELD
    validateAcceptedSnapshot(state);
}

#undef NUMI_CARDBOARD_SNAPSHOT_VECTORS

} // namespace shipping_snapshot_detail

inline void writeShippingAcceptedSnapshot(
    const std::filesystem::path& path,
    const numi::matter::RuntimeStateSnapshot& state) {
    using namespace shipping_snapshot_detail;
    validateAcceptedSnapshot(state);
    const std::filesystem::path partial = path.string() + ".partial";
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create shipping snapshot");
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    writePod(output, kVersion);
    writePod(output, kEndianMarker);
    writePod(output, static_cast<std::uint32_t>(sizeof(numi::matter::RuntimeStateSnapshot)));
    writePod(output, static_cast<std::uint8_t>(state.available));
    writeSnapshotPayload(output, state);
    output.flush();
    if (!output) throw std::runtime_error("failed flushing shipping snapshot");
    output.close();
    std::error_code error;
    std::filesystem::rename(partial, path, error);
    if (error) throw std::runtime_error("cannot publish shipping snapshot: " + error.message());
}

inline void writeShippingAcceptedSnapshot(
    const std::filesystem::path& path,
    const numi::matter::RuntimeStateSnapshot& state,
    const numi::matter::CompiledWorld& world,
    const std::uint32_t environmentCount,
    const bool captureDiagnostics) {
    shipping_snapshot_detail::validateAcceptedSnapshot(state);
    shipping_snapshot_detail::validateShippingCheckpointShape(
        state, world, environmentCount, captureDiagnostics);
    writeShippingAcceptedSnapshot(path, state);
}

inline numi::matter::RuntimeStateSnapshot readShippingAcceptedSnapshot(
    const std::filesystem::path& path,
    const std::uint64_t expectedSourcePhysicsFingerprint,
    const std::uint64_t expectedDeviceProgramFingerprint,
    const std::uint64_t maximumBytes =
        16ull * 1024ull * 1024ull * 1024ull) {
    using namespace shipping_snapshot_detail;
    std::error_code error;
    const std::uint64_t fileBytes = std::filesystem::file_size(path, error);
    if (error || fileBytes > maximumBytes)
        throw std::runtime_error("shipping snapshot file is unavailable or exceeds its byte limit");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open shipping snapshot");
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != kMagic || readPod<std::uint32_t>(input) != kVersion ||
        readPod<std::uint32_t>(input) != kEndianMarker ||
        readPod<std::uint32_t>(input) != sizeof(numi::matter::RuntimeStateSnapshot))
        throw std::runtime_error("shipping snapshot format or native ABI does not match");
    const std::uint8_t available = readPod<std::uint8_t>(input);
    if (available != 1u)
        throw std::runtime_error("shipping snapshot is not an available accepted state");
    numi::matter::RuntimeStateSnapshot state;
    state.available = true;
    state.message = readString(input, maximumBytes);
    readScalarFields(input, state);
    if (state.sourcePhysicsFingerprint != expectedSourcePhysicsFingerprint ||
        state.deviceProgramFingerprint != expectedDeviceProgramFingerprint)
        throw std::runtime_error("shipping snapshot fingerprint does not match the current compiled world");
    // Only allocate variable-sized arrays after cheap identity validation. Each
    // subsequent allocation is additionally bounded by the actual remaining
    // bytes in the file, not merely by the declared vector count.
    readSnapshotVectors(input, state, maximumBytes);
    validateAcceptedSnapshot(state);
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("shipping snapshot has trailing bytes");
    return state;
}

struct ShippingSnapshotIdentity {
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t deviceProgramFingerprint = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t materialStateStride = 0u;
    bool available = false;
};

// Read only the fixed identity header when a checkpoint is used as provenance
// for a later run. This does not restore state or allocate any FEM/material
// arenas; the file SHA-256 and these stored fingerprints must both be checked
// by the caller. Full state restoration still uses
// readShippingAcceptedSnapshot with the exact current world fingerprints.
inline ShippingSnapshotIdentity readShippingAcceptedSnapshotIdentity(
    const std::filesystem::path& path,
    const std::uint64_t maximumBytes =
        16ull * 1024ull * 1024ull * 1024ull) {
    using namespace shipping_snapshot_detail;
    std::error_code error;
    const std::uint64_t fileBytes = std::filesystem::file_size(path, error);
    if (error || fileBytes > maximumBytes)
        throw std::runtime_error(
            "shipping snapshot identity file is unavailable or exceeds its byte limit");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open shipping snapshot identity");
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic != kMagic || readPod<std::uint32_t>(input) != kVersion ||
        readPod<std::uint32_t>(input) != kEndianMarker ||
        readPod<std::uint32_t>(input) != sizeof(numi::matter::RuntimeStateSnapshot))
        throw std::runtime_error(
            "shipping snapshot identity format or native ABI does not match");
    const std::uint8_t available = readPod<std::uint8_t>(input);
    if (available != 1u)
        throw std::runtime_error("shipping snapshot identity is not an available state");
    numi::matter::RuntimeStateSnapshot state;
    state.available = true;
    state.message = readString(input, maximumBytes);
    readScalarFields(input, state);
    if (state.sourcePhysicsFingerprint == 0u ||
        state.deviceProgramFingerprint == 0u ||
        state.materialStateStride == 0u)
        throw std::runtime_error("shipping snapshot identity has invalid source fingerprints");
    return {state.sourcePhysicsFingerprint, state.deviceProgramFingerprint,
            state.controlStep, state.physicsSubstep, state.materialStateStride,
            state.available};
}

} // namespace numi::cardboard
