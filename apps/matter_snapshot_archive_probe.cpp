#include "metalrobo/MatterSnapshotArchive.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
bool equalBytes(const std::vector<T>& left, const std::vector<T>& right) {
    static_assert(std::is_trivially_copyable_v<T>);
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(
            left.data(),
            right.data(),
            left.size() * sizeof(T)
        ) == 0);
}

bool equal(
    const numi::matter::RuntimeStateSnapshot& left,
    const numi::matter::RuntimeStateSnapshot& right
) {
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
        equalBytes(
            left.mpmActiveNodeIndices,
            right.mpmActiveNodeIndices
        ) &&
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
            left.deformableContactHistories,
            right.deformableContactHistories
        ) &&
        equalBytes(
            left.particleMaterialState,
            right.particleMaterialState
        ) &&
        equalBytes(left.femMaterialState, right.femMaterialState) &&
        equalBytes(left.identification, right.identification) &&
        equalBytes(
            left.environmentParameters,
            right.environmentParameters
        );
}

numi::matter::RuntimeStateSnapshot fixture() {
    numi::matter::RuntimeStateSnapshot result;
    result.available = true;
    result.message = "fixture-only message is not archived";
    result.sourcePhysicsFingerprint = 0x1020304050607080ull;
    result.deviceProgramFingerprint = 0x8877665544332211ull;
    result.controlStep = 37u;
    result.physicsSubstep = 5u;
    result.identificationGeneration = 9u;
    result.identificationCheckpoint = 8u;
    result.identificationAdvanced = true;
    result.sutureProxyEdges = {41u, 42u, 43u, 44u};
    result.sutureProxyBindingRevision = 17u;
    result.coupledTimestepMultiplier = 1u;
    result.coupledTimestepDivisor = 4u;
    result.allocationGeneration = 3u;
    result.learnedWeightRevision = 12u;
    result.materialStateStride = 2u;

    result.particles.resize(2u);
    result.femNodes.resize(3u);
    result.femFields.resize(2u);
    result.femTopologyNodes.resize(3u);
    result.femTopologyTetrahedra.resize(2u);
    result.cohesiveFaces.resize(2u);
    result.punctureChannels.resize(3u);
    result.topologyStates.resize(2u);
    result.statuses.resize(2u);
    result.solverCertificates.resize(2u);
    result.mpmActiveNodeIndices = {3u, 7u, 11u};
    result.mpmNodeToActive = {1u, 0u, 2u};
    result.mpmActiveNodeCounts = {2u, 1u};
    result.rigidGeneralizedCandidate = {0.5f, -0.25f, 1.5f};
    result.learnedWeights = {0.125f, -0.75f};
    result.adaptive.resize(2u);
    result.schedulers.resize(2u);
    result.reactions.resize(2u);
    result.rigidStates.resize(2u);
    result.contactSamples.resize(2u);
    result.contactHistories.resize(3u);
    result.deformableContactHistories.resize(2u);
    result.particleMaterialState = {0.1f, 0.2f, 0.3f, 0.4f};
    result.femMaterialState = {0.5f, 0.6f, 0.7f, 0.8f};
    result.identification.resize(2u);
    result.environmentParameters = {690.0f, 81.2f, 72.4f};

    result.femNodes[0u].positionAndMass = {1.0f, 2.0f, 3.0f, 4.0f};
    result.femNodes[1u].velocityAndInverseMass = {
        -1.0f, 0.25f, 0.5f, 2.0f,
    };
    result.punctureChannels[0u].identity = {1u, 2u, 3u, 4u};
    return result;
}

} // namespace

int main(int argc, const char* argv[]) {
    try {
        require(
            argc == 2,
            "usage: metalrobo_matter_snapshot_archive_probe OUTPUT"
        );
        const std::filesystem::path path = argv[1];
        const auto source = fixture();
        const auto written = metalrobo::writeMatterSnapshotArchive(
            source,
            path
        );
        require(written.succeeded(), written.message);
        require(
            written.contentHash != 0u && written.payloadBytes != 0u,
            "Matter snapshot archive did not publish content identity"
        );

        numi::matter::RuntimeStateSnapshot decoded;
        const auto read = metalrobo::readMatterSnapshotArchive(path, decoded);
        require(read.succeeded(), read.message);
        require(
            read.contentHash == written.contentHash &&
                read.payloadBytes == written.payloadBytes &&
                equal(source, decoded),
            "Matter snapshot archive round trip changed state bytes"
        );

        {
            std::fstream stream(
                path,
                std::ios::binary | std::ios::in | std::ios::out
            );
            require(stream.good(), "could not reopen Matter snapshot archive");
            stream.seekg(-1, std::ios::end);
            char value = 0;
            stream.read(&value, 1);
            require(stream.good(), "could not read archive corruption byte");
            value = static_cast<char>(value ^ 0x5a);
            stream.seekp(-1, std::ios::end);
            stream.write(&value, 1);
            stream.flush();
            require(stream.good(), "could not write archive corruption byte");
        }
        numi::matter::RuntimeStateSnapshot unchanged = fixture();
        const auto corrupt = metalrobo::readMatterSnapshotArchive(
            path,
            unchanged
        );
        require(
            corrupt.status ==
                    metalrobo::MatterSnapshotArchiveStatus::corruptPayload &&
                equal(unchanged, source),
            "corrupt Matter snapshot archive did not fail closed"
        );

        std::error_code error;
        std::filesystem::remove(path, error);
        require(!error, "could not remove Matter snapshot archive fixture");
        std::cout
            << "matter_snapshot_archive=ok"
            << " version="
            << metalrobo::kMatterSnapshotArchiveVersion
            << " content_hash=0x" << std::hex << written.contentHash
            << std::dec
            << " payload_bytes=" << written.payloadBytes
            << " corrupt_payload_rejected=yes\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "matter_snapshot_archive=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
