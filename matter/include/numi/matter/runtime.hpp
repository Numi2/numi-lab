#pragma once

#include "numi/matter/ir.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace numi::matter {

struct RuntimeConfiguration {
    std::filesystem::path metallib;
    std::uint32_t environmentCount = 0u;
    bool captureEvents = true;
    bool captureDiagnostics = false;
};

struct BorrowedRigidWorldBuffers {
    void* currentBodies = nullptr;       // id<MTLBuffer>, MRBodyStateGPU
    void* articulatedWrenches = nullptr; // id<MTLBuffer>, MRABABodyWrenchGPU
    void* sceneBodies = nullptr;         // id<MTLBuffer>, MRBodyStateGPU
    std::uint32_t currentBodyCount = 0u;
    std::uint32_t currentBodyStride = 0u;
    std::uint32_t articulatedBodyCount = 0u;
    std::uint32_t articulatedStride = 0u;
    std::uint32_t sceneBodyCount = 0u;
    std::uint32_t sceneStride = 0u;
};

struct EncodeRequest {
    void* commandBuffer = nullptr; // borrowed id<MTLCommandBuffer>
    BorrowedRigidWorldBuffers rigid{};
    void* adaptiveTriggers = nullptr; // optional id<MTLBuffer>
    std::uint32_t controlStep = 0u;
    std::uint64_t seed = 0u;
    // NM_INVALID_INDEX encodes a complete frame. A concrete microtick encodes
    // one slice so NumiSolver can interleave Matter with ABA/contact.
    std::uint32_t microtick = NM_INVALID_INDEX;
    bool beginFrame = true;
    bool endFrame = true;
    bool runIdentification = false;
    bool runAdaptiveTransfer = true;
};

struct RuntimeDiagnostics {
    bool encoded = false;
    std::size_t residentBytes = 0u;
    std::string device;
    std::string message;
};

class MatterRuntime {
public:
    MatterRuntime();
    ~MatterRuntime();
    MatterRuntime(MatterRuntime&&) noexcept;
    MatterRuntime& operator=(MatterRuntime&&) noexcept;
    MatterRuntime(const MatterRuntime&) = delete;
    MatterRuntime& operator=(const MatterRuntime&) = delete;

    [[nodiscard]] RuntimeDiagnostics initialize(
        const CompiledMatterWorld& world,
        const RuntimeConfiguration& configuration = {}
    );
    // Encodes into a borrowed command buffer; never commits, waits, or reads a
    // device count. Initialization may perform one synchronous immutable upload.
    [[nodiscard]] RuntimeDiagnostics encode(const EncodeRequest& request);

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] void* eventBuffer() const noexcept;
    [[nodiscard]] void* statusBuffer() const noexcept;
    [[nodiscard]] void* parameterBuffer() const noexcept;
    [[nodiscard]] void* identificationLossBuffer() const noexcept;
    [[nodiscard]] void* adaptiveTriggerBuffer() const noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace numi::matter
