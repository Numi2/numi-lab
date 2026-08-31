#pragma once

#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human_shared.h"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace numi::matter {

// Complete load and bone-anchor maps for one Numi Human FEM consumer. Node
// records are immutable and indexed by cooked global FEM-node index.
//
// Active tendon replacement mode requires at least one endpoint replacement and
// a positive productionForceOwnerFraction. Endpoint mode removes the declared
// anchor-endpoint J^T share. Full-muscle-row mode removes the selected source
// muscle's complete generalized-force row and restores only its load-endpoint
// reaction, so internal source wrap forces cannot duplicate a continuum path.
// A declared distal force couple additionally maps the other terminal across
// two opposing exact attachment patches; its signed nodal weights sum to zero
// and its absolute weights sum to twice the owned terminal force.
// Both modes return solved fixed-node reactions through the owning body
// Jacobians.
//
// Passive attachment-only mode uses no endpoint replacements, no active node
// loads, and a zero productionForceOwnerFraction. Active anchors may reference
// different articulated bodies; their fixed-node reactions are returned through
// each owning body Jacobian in the same accepted command-buffer transaction.
struct NumiHumanTendonFEMLoadSource {
    std::span<const NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads{};
    std::span<const NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors{};
    std::span<const NMNumiHumanTendonFEMEndpointReplacementGPU>
        endpointReplacements{};
    // Optional internal cartilage/meniscus contact. Ranges exactly cover the
    // cooked FEM-node array and index contactContributions; every sample must
    // have one slave and three master contributions.
    std::span<const NMNumiHumanFEMContactSampleGPU> contactSamples{};
    std::span<const NMNumiHumanFEMContactContributionGPU>
        contactContributions{};
    std::span<const NMIncidenceRangeGPU> contactRanges{};
    std::uint32_t endpointCount = 0u;
    std::uint32_t environmentCount = 1u;
    float productionForceOwnerFraction = 0.0f;
};

struct NumiHumanTendonFEMLoadConfiguration {
    std::filesystem::path metallib;
};

struct NumiHumanTendonFEMLoadDiagnostics {
    bool initialized = false;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint64_t fingerprint = 0u;
    // Valid after the enclosing borrowed command buffer has completed. L1 is
    // the sum of nodal force magnitudes; resultant is the magnitude of their
    // vector sum for the most recently encoded pass across all environments.
    double assembledExternalForceL1Newtons = 0.0;
    double assembledExternalForceResultantNewtons = 0.0;
    std::uint32_t contactSampleCount = 0u;
    std::string message;
};

class NumiHumanTendonFEMLoadAdapter {
public:
    NumiHumanTendonFEMLoadAdapter();
    ~NumiHumanTendonFEMLoadAdapter();
    NumiHumanTendonFEMLoadAdapter(NumiHumanTendonFEMLoadAdapter&&) noexcept;
    NumiHumanTendonFEMLoadAdapter& operator=(
        NumiHumanTendonFEMLoadAdapter&&
    ) noexcept;
    NumiHumanTendonFEMLoadAdapter(
        const NumiHumanTendonFEMLoadAdapter&
    ) = delete;
    NumiHumanTendonFEMLoadAdapter& operator=(
        const NumiHumanTendonFEMLoadAdapter&
    ) = delete;

    [[nodiscard]] bool initialize(
        Runtime& runtime,
        const NumiHumanTendonFEMLoadSource& source,
        const NumiHumanTendonFEMLoadConfiguration& configuration
    );
    [[nodiscard]] metalrobo::MetalNumiHumanTendonLoadProgram
    program() noexcept;
    [[nodiscard]] NumiHumanTendonFEMLoadDiagnostics diagnostics() const noexcept;

private:
    [[nodiscard]] bool encodePreDynamics(
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    );
    [[nodiscard]] bool encodePostValidation(
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    );
    void abort(void* commandBuffer) noexcept;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace numi::matter
