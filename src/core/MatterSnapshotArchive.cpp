#include "metalrobo/MatterSnapshotArchive.hpp"

#include "numi/matter/shared.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kEndianMarker = 0x01020304u;
constexpr std::array<char, 8u> kMagic{
    'N', 'M', 'S', 'N', 'A', 'P', '0', '1',
};

struct MatterSnapshotFileHeader {
    std::array<char, 8u> magic{};
    std::uint32_t formatVersion = 0u;
    std::uint32_t endianMarker = 0u;
    std::uint32_t matterAbiVersion = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint64_t payloadBytes = 0u;
    std::uint64_t contentHash = 0u;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t deviceProgramFingerprint = 0u;
    std::array<std::uint64_t, 4u> reserved{};
};

static_assert(std::is_trivially_copyable_v<MatterSnapshotFileHeader>);
static_assert(sizeof(MatterSnapshotFileHeader) == 88u);

MatterSnapshotArchiveResult fail(
    const MatterSnapshotArchiveStatus status,
    std::string message
) {
    return {
        .status = status,
        .message = std::move(message),
    };
}

std::uint64_t hashBytes(const std::span<const std::byte> bytes) {
    std::uint64_t hash = kFnvOffset;
    for (const std::byte value : bytes) {
        hash ^= std::to_integer<unsigned char>(value);
        hash *= kFnvPrime;
    }
    return hash == 0u ? 1u : hash;
}

template <typename T>
bool equalBytes(
    const std::vector<T>& left,
    const std::vector<T>& right
) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(
            left.data(),
            right.data(),
            left.size() * sizeof(T)
        ) == 0);
}

class PayloadWriter {
public:
    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* begin = reinterpret_cast<const std::byte*>(&value);
        bytes_.insert(bytes_.end(), begin, begin + sizeof(T));
    }

    template <typename T>
    void podVector(const std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint32_t elementBytes = sizeof(T);
        const std::uint64_t count = values.size();
        pod(elementBytes);
        pod(count);
        if (values.empty()) {
            return;
        }
        const auto* begin =
            reinterpret_cast<const std::byte*>(values.data());
        bytes_.insert(
            bytes_.end(),
            begin,
            begin + values.size() * sizeof(T)
        );
    }

    [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept {
        return bytes_;
    }

private:
    std::vector<std::byte> bytes_;
};

class PayloadReader {
public:
    explicit PayloadReader(const std::span<const std::byte> bytes)
        : bytes_(bytes) {}

    template <typename T>
    bool pod(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (remaining() < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, bytes_.data() + cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return true;
    }

    template <typename T>
    bool podVector(std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::uint32_t elementBytes = 0u;
        std::uint64_t count = 0u;
        if (!pod(elementBytes) || !pod(count) ||
            elementBytes != sizeof(T) ||
            count > std::numeric_limits<std::size_t>::max() ||
            (count != 0u && sizeof(T) >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(count))) {
            return false;
        }
        const std::size_t size = static_cast<std::size_t>(count);
        const std::size_t bytes = size * sizeof(T);
        if (bytes > remaining()) {
            return false;
        }
        values.resize(size);
        if (bytes != 0u) {
            std::memcpy(values.data(), bytes_.data() + cursor_, bytes);
        }
        cursor_ += bytes;
        return true;
    }

    [[nodiscard]] bool finished() const noexcept {
        return cursor_ == bytes_.size();
    }

private:
    [[nodiscard]] std::size_t remaining() const noexcept {
        return bytes_.size() - cursor_;
    }

    std::span<const std::byte> bytes_;
    std::size_t cursor_ = 0u;
};

std::vector<std::byte> serialize(
    const numi::matter::RuntimeStateSnapshot& snapshot
) {
    PayloadWriter writer;
    writer.pod(snapshot.sourcePhysicsFingerprint);
    writer.pod(snapshot.deviceProgramFingerprint);
    writer.pod(snapshot.controlStep);
    writer.pod(snapshot.physicsSubstep);
    writer.pod(snapshot.identificationGeneration);
    writer.pod(snapshot.identificationCheckpoint);
    const std::uint32_t identificationAdvanced =
        snapshot.identificationAdvanced ? 1u : 0u;
    writer.pod(identificationAdvanced);
    writer.pod(snapshot.sutureProxyBindingRevision);
    writer.pod(snapshot.coupledTimestepMultiplier);
    writer.pod(snapshot.coupledTimestepDivisor);
    writer.pod(snapshot.fgmresIterationBudgetOverride);
    writer.pod(snapshot.newtonIterationBudgetOverride);
    writer.pod(snapshot.allocationGeneration);
    writer.pod(snapshot.learnedWeightRevision);
    writer.pod(snapshot.materialStateStride);

    writer.podVector(snapshot.sutureProxyEdges);
    writer.podVector(snapshot.particles);
    writer.podVector(snapshot.femNodes);
    writer.podVector(snapshot.femFields);
    writer.podVector(snapshot.femTopologyNodes);
    writer.podVector(snapshot.femTopologyTetrahedra);
    writer.podVector(snapshot.cohesiveFaces);
    writer.podVector(snapshot.punctureChannels);
    writer.podVector(snapshot.topologyStates);
    writer.podVector(snapshot.statuses);
    writer.podVector(snapshot.solverCertificates);
    writer.podVector(snapshot.mpmActiveNodeIndices);
    writer.podVector(snapshot.mpmNodeToActive);
    writer.podVector(snapshot.mpmActiveNodeCounts);
    writer.podVector(snapshot.rigidGeneralizedCandidate);
    writer.podVector(snapshot.learnedWeights);
    writer.podVector(snapshot.adaptive);
    writer.podVector(snapshot.schedulers);
    writer.podVector(snapshot.reactions);
    writer.podVector(snapshot.rigidStates);
    writer.podVector(snapshot.contactSamples);
    writer.podVector(snapshot.contactHistories);
    writer.podVector(snapshot.humanSupportHistories);
    writer.podVector(snapshot.humanSupportConsequences);
    writer.podVector(snapshot.deformableContactHistories);
    writer.podVector(snapshot.particleMaterialState);
    writer.podVector(snapshot.femMaterialState);
    writer.podVector(snapshot.identification);
    writer.podVector(snapshot.environmentParameters);
    return writer.bytes();
}

bool deserialize(
    const std::span<const std::byte> payload,
    numi::matter::RuntimeStateSnapshot& snapshot,
    const std::uint32_t formatVersion
) {
    PayloadReader reader(payload);
    std::uint32_t identificationAdvanced = 0u;
    if (!reader.pod(snapshot.sourcePhysicsFingerprint) ||
        !reader.pod(snapshot.deviceProgramFingerprint) ||
        !reader.pod(snapshot.controlStep) ||
        !reader.pod(snapshot.physicsSubstep) ||
        !reader.pod(snapshot.identificationGeneration) ||
        !reader.pod(snapshot.identificationCheckpoint) ||
        !reader.pod(identificationAdvanced) ||
        identificationAdvanced > 1u ||
        !reader.pod(snapshot.sutureProxyBindingRevision) ||
        !reader.pod(snapshot.coupledTimestepMultiplier) ||
        !reader.pod(snapshot.coupledTimestepDivisor)) {
        return false;
    }
    // v1 snapshots predate the phase-local FGMRES selector and v1/v2 predate
    // the phase-local Newton selector. Zero is their exact legacy meaning: use
    // the corresponding budget fingerprinted into the cooked device program.
    snapshot.fgmresIterationBudgetOverride = 0u;
    snapshot.newtonIterationBudgetOverride = 0u;
    if (formatVersion >= 2u &&
        !reader.pod(snapshot.fgmresIterationBudgetOverride)) {
        return false;
    }
    if (formatVersion >= 3u &&
        !reader.pod(snapshot.newtonIterationBudgetOverride)) {
        return false;
    }
    if (!reader.pod(snapshot.allocationGeneration) ||
        !reader.pod(snapshot.learnedWeightRevision) ||
        !reader.pod(snapshot.materialStateStride) ||
        !reader.podVector(snapshot.sutureProxyEdges) ||
        !reader.podVector(snapshot.particles) ||
        !reader.podVector(snapshot.femNodes) ||
        !reader.podVector(snapshot.femFields) ||
        !reader.podVector(snapshot.femTopologyNodes) ||
        !reader.podVector(snapshot.femTopologyTetrahedra) ||
        !reader.podVector(snapshot.cohesiveFaces) ||
        !reader.podVector(snapshot.punctureChannels) ||
        !reader.podVector(snapshot.topologyStates) ||
        !reader.podVector(snapshot.statuses) ||
        !reader.podVector(snapshot.solverCertificates) ||
        !reader.podVector(snapshot.mpmActiveNodeIndices) ||
        !reader.podVector(snapshot.mpmNodeToActive) ||
        !reader.podVector(snapshot.mpmActiveNodeCounts) ||
        !reader.podVector(snapshot.rigidGeneralizedCandidate) ||
        !reader.podVector(snapshot.learnedWeights) ||
        !reader.podVector(snapshot.adaptive) ||
        !reader.podVector(snapshot.schedulers) ||
        !reader.podVector(snapshot.reactions) ||
        !reader.podVector(snapshot.rigidStates) ||
        !reader.podVector(snapshot.contactSamples) ||
        !reader.podVector(snapshot.contactHistories)) {
        return false;
    }
    if (formatVersion >= 4u &&
        (!reader.podVector(snapshot.humanSupportHistories) ||
         !reader.podVector(snapshot.humanSupportConsequences))) {
        return false;
    }
    if (
        !reader.podVector(snapshot.deformableContactHistories) ||
        !reader.podVector(snapshot.particleMaterialState) ||
        !reader.podVector(snapshot.femMaterialState) ||
        !reader.podVector(snapshot.identification) ||
        !reader.podVector(snapshot.environmentParameters) ||
        !reader.finished()) {
        return false;
    }
    snapshot.identificationAdvanced = identificationAdvanced != 0u;
    snapshot.available = true;
    snapshot.message = "Matter snapshot archive decoded";
    return snapshot.sourcePhysicsFingerprint != 0u &&
        snapshot.deviceProgramFingerprint != 0u &&
        snapshot.coupledTimestepMultiplier != 0u &&
        snapshot.coupledTimestepDivisor != 0u;
}

} // namespace

MatterSnapshotArchiveResult writeMatterSnapshotArchive(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    const std::filesystem::path& path
) {
    if (path.empty() || !snapshot.available ||
        snapshot.sourcePhysicsFingerprint == 0u ||
        snapshot.deviceProgramFingerprint == 0u) {
        return fail(
            MatterSnapshotArchiveStatus::invalidSnapshot,
            path.empty()
                ? "Matter snapshot archive path is empty"
                : "Matter snapshot is unavailable or has no program identity"
        );
    }
    try {
        const std::vector<std::byte> payload = serialize(snapshot);
        MatterSnapshotFileHeader header;
        header.magic = kMagic;
        header.formatVersion = kMatterSnapshotArchiveVersion;
        header.endianMarker = kEndianMarker;
        header.matterAbiVersion = NM_MATTER_ABI_VERSION;
        header.payloadBytes = payload.size();
        header.contentHash = hashBytes(payload);
        header.sourcePhysicsFingerprint =
            snapshot.sourcePhysicsFingerprint;
        header.deviceProgramFingerprint =
            snapshot.deviceProgramFingerprint;

        std::filesystem::path temporary = path;
        temporary += "." + std::to_string(header.contentHash) + ".tmp";
        {
            std::ofstream stream(
                temporary,
                std::ios::binary | std::ios::trunc
            );
            if (!stream) {
                return fail(
                    MatterSnapshotArchiveStatus::ioFailure,
                    "could not open temporary Matter snapshot archive"
                );
            }
            stream.write(
                reinterpret_cast<const char*>(&header),
                sizeof(header)
            );
            if (!payload.empty()) {
                stream.write(
                    reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size())
                );
            }
            stream.flush();
            if (!stream) {
                stream.close();
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                return fail(
                    MatterSnapshotArchiveStatus::ioFailure,
                    "failed while writing Matter snapshot archive"
                );
            }
        }
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return fail(
                MatterSnapshotArchiveStatus::ioFailure,
                "could not publish Matter snapshot archive"
            );
        }
        return {
            .contentHash = header.contentHash,
            .payloadBytes = header.payloadBytes,
        };
    } catch (const std::bad_alloc&) {
        return fail(
            MatterSnapshotArchiveStatus::capacityOverflow,
            "host allocation failed while writing Matter snapshot archive"
        );
    } catch (const std::exception& exception) {
        return fail(
            MatterSnapshotArchiveStatus::ioFailure,
            exception.what()
        );
    }
}

MatterSnapshotArchiveResult readMatterSnapshotArchive(
    const std::filesystem::path& path,
    numi::matter::RuntimeStateSnapshot& output
) {
    if (path.empty()) {
        return fail(
            MatterSnapshotArchiveStatus::ioFailure,
            "Matter snapshot archive path is empty"
        );
    }
    try {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            return fail(
                MatterSnapshotArchiveStatus::ioFailure,
                "could not open Matter snapshot archive"
            );
        }
        const std::streamoff fileSize = stream.tellg();
        if (fileSize < static_cast<std::streamoff>(
                sizeof(MatterSnapshotFileHeader)
            )) {
            return fail(
                MatterSnapshotArchiveStatus::corruptPayload,
                "Matter snapshot archive is shorter than its header"
            );
        }
        stream.seekg(0);
        MatterSnapshotFileHeader header;
        stream.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!stream || header.magic != kMagic) {
            return fail(
                MatterSnapshotArchiveStatus::corruptPayload,
                "Matter snapshot archive magic is invalid"
            );
        }
        if ((header.formatVersion != 1u &&
             header.formatVersion != 2u &&
             header.formatVersion != 3u &&
             header.formatVersion != kMatterSnapshotArchiveVersion) ||
            header.endianMarker != kEndianMarker ||
            header.matterAbiVersion != NM_MATTER_ABI_VERSION) {
            return fail(
                MatterSnapshotArchiveStatus::unsupportedVersion,
                "Matter snapshot archive format or ABI is unsupported"
            );
        }
        const std::uint64_t available =
            static_cast<std::uint64_t>(fileSize) - sizeof(header);
        if (header.payloadBytes != available ||
            header.payloadBytes >
                std::numeric_limits<std::size_t>::max() ||
            header.payloadBytes >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamsize>::max()
                )) {
            return fail(
                MatterSnapshotArchiveStatus::corruptPayload,
                "Matter snapshot archive payload length is invalid"
            );
        }
        std::vector<std::byte> payload(
            static_cast<std::size_t>(header.payloadBytes)
        );
        if (!payload.empty()) {
            stream.read(
                reinterpret_cast<char*>(payload.data()),
                static_cast<std::streamsize>(payload.size())
            );
        }
        if (!stream || hashBytes(payload) != header.contentHash) {
            return fail(
                MatterSnapshotArchiveStatus::corruptPayload,
                "Matter snapshot archive payload hash is invalid"
            );
        }
        numi::matter::RuntimeStateSnapshot candidate;
        if (!deserialize(payload, candidate, header.formatVersion) ||
            candidate.sourcePhysicsFingerprint !=
                header.sourcePhysicsFingerprint ||
            candidate.deviceProgramFingerprint !=
                header.deviceProgramFingerprint) {
            return fail(
                MatterSnapshotArchiveStatus::corruptPayload,
                "Matter snapshot archive payload identity is inconsistent"
            );
        }
        output = std::move(candidate);
        return {
            .contentHash = header.contentHash,
            .payloadBytes = header.payloadBytes,
        };
    } catch (const std::bad_alloc&) {
        return fail(
            MatterSnapshotArchiveStatus::capacityOverflow,
            "host allocation failed while reading Matter snapshot archive"
        );
    } catch (const std::exception& exception) {
        return fail(
            MatterSnapshotArchiveStatus::ioFailure,
            exception.what()
        );
    }
}

bool sameMatterSnapshotAuthority(
    const numi::matter::RuntimeStateSnapshot& left,
    const numi::matter::RuntimeStateSnapshot& right
) noexcept {
    return left.available == right.available &&
        left.sourcePhysicsFingerprint == right.sourcePhysicsFingerprint &&
        left.deviceProgramFingerprint == right.deviceProgramFingerprint &&
        left.controlStep == right.controlStep &&
        left.physicsSubstep == right.physicsSubstep &&
        left.identificationGeneration == right.identificationGeneration &&
        left.identificationCheckpoint == right.identificationCheckpoint &&
        left.identificationAdvanced == right.identificationAdvanced &&
        left.sutureProxyBindingRevision ==
            right.sutureProxyBindingRevision &&
        left.coupledTimestepMultiplier ==
            right.coupledTimestepMultiplier &&
        left.coupledTimestepDivisor == right.coupledTimestepDivisor &&
        left.fgmresIterationBudgetOverride ==
            right.fgmresIterationBudgetOverride &&
        left.newtonIterationBudgetOverride ==
            right.newtonIterationBudgetOverride &&
        left.allocationGeneration == right.allocationGeneration &&
        left.learnedWeightRevision == right.learnedWeightRevision &&
        left.materialStateStride == right.materialStateStride &&
        equalBytes(left.sutureProxyEdges, right.sutureProxyEdges) &&
        equalBytes(left.particles, right.particles) &&
        equalBytes(left.femNodes, right.femNodes) &&
        equalBytes(left.femFields, right.femFields) &&
        equalBytes(left.femTopologyNodes, right.femTopologyNodes) &&
        equalBytes(
            left.femTopologyTetrahedra,
            right.femTopologyTetrahedra
        ) &&
        equalBytes(left.cohesiveFaces, right.cohesiveFaces) &&
        equalBytes(left.punctureChannels, right.punctureChannels) &&
        equalBytes(left.topologyStates, right.topologyStates) &&
        equalBytes(left.statuses, right.statuses) &&
        equalBytes(left.solverCertificates, right.solverCertificates) &&
        equalBytes(left.mpmActiveNodeIndices, right.mpmActiveNodeIndices) &&
        equalBytes(left.mpmNodeToActive, right.mpmNodeToActive) &&
        equalBytes(left.mpmActiveNodeCounts, right.mpmActiveNodeCounts) &&
        equalBytes(
            left.rigidGeneralizedCandidate,
            right.rigidGeneralizedCandidate
        ) &&
        equalBytes(left.learnedWeights, right.learnedWeights) &&
        equalBytes(left.adaptive, right.adaptive) &&
        equalBytes(left.schedulers, right.schedulers) &&
        equalBytes(left.reactions, right.reactions) &&
        equalBytes(left.rigidStates, right.rigidStates) &&
        equalBytes(left.contactSamples, right.contactSamples) &&
        equalBytes(left.contactHistories, right.contactHistories) &&
        equalBytes(
            left.humanSupportHistories,
            right.humanSupportHistories) &&
        equalBytes(
            left.humanSupportConsequences,
            right.humanSupportConsequences) &&
        equalBytes(
            left.deformableContactHistories,
            right.deformableContactHistories
        ) &&
        equalBytes(left.particleMaterialState, right.particleMaterialState) &&
        equalBytes(left.femMaterialState, right.femMaterialState) &&
        equalBytes(left.identification, right.identification) &&
        equalBytes(left.environmentParameters, right.environmentParameters);
}

const char* matterSnapshotArchiveStatusName(
    const MatterSnapshotArchiveStatus status
) noexcept {
    switch (status) {
    case MatterSnapshotArchiveStatus::success:
        return "success";
    case MatterSnapshotArchiveStatus::invalidSnapshot:
        return "invalid_snapshot";
    case MatterSnapshotArchiveStatus::unsupportedVersion:
        return "unsupported_version";
    case MatterSnapshotArchiveStatus::corruptPayload:
        return "corrupt_payload";
    case MatterSnapshotArchiveStatus::capacityOverflow:
        return "capacity_overflow";
    case MatterSnapshotArchiveStatus::ioFailure:
        return "io_failure";
    }
    return "unknown";
}

} // namespace metalrobo
