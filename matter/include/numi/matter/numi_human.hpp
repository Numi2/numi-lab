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
// a positive productionForceOwnerFraction. Each replacement removes exactly the
// declared source J^T share and returns solved fixed-node reactions through the
// owning body Jacobian.
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
