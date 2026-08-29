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

// A complete FEM-node load map for one Numi Human tendon consumer. Every
// record is immutable and indexed by cooked global FEM-node index. This first
// adapter is deliberately non-owning: it may qualify a load-driven continuum
// in the same command buffer, but it cannot replace rigid J^T or return anchor
// reactions until that ownership path is explicitly implemented.
struct NumiHumanTendonFEMLoadSource {
    std::span<const NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads{};
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
    [[nodiscard]] bool encode(
        const metalrobo::MetalNumiHumanTendonLoadPass& pass
    );
    void abort(void* commandBuffer) noexcept;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace numi::matter
