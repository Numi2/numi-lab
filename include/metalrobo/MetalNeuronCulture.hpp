#pragma once

#include "metalrobo/NeuronCulture.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace metalrobo {

enum class MetalNeuronCultureStatus : std::uint32_t {
    pending = 0u,
    success,
    invalidArgument,
    busy,
    commandFailure,
    internalFailure,
};

enum class MetalNeuronCultureAcceptedBuffer : std::uint32_t {
    membrane = 0u, refractory, preTrace, postTrace, weights, depression,
    spikes, spikeHistory, electrodeSpikeCounts, phase, tubulin,
};

struct MetalNeuronCultureBufferView {
    MetalNeuronCultureAcceptedBuffer kind{};
    void* metalBuffer = nullptr;
    std::uint64_t gpuAddress = 0u;
    std::uint64_t byteLength = 0u;
};

struct MetalNeuronCultureSupportRequest {
    std::uint64_t cultureFingerprint = 0u;
    std::uint64_t rootFingerprint = 0u;
    void* supportConsequencesBuffer = nullptr;
    std::uint64_t supportConsequencesGPUAddress = 0u;
    std::uint32_t supportCount = 10u;
    std::uint32_t supportStride = 10u;
    std::uint32_t tickCount = 100u;
    float physicsTimestepSeconds = 0.001f;
    float currentPerNewton = 1.0f;
};

using MetalNeuronCultureCompletion = void (*)(
    void* context, MetalNeuronCultureStatus status) noexcept;

class MetalNeuronCultureAcceptedView {
public:
    MetalNeuronCultureAcceptedView() = default;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t cultureFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] std::uint64_t tick() const noexcept;
    [[nodiscard]] std::uint64_t growthIteration() const noexcept;
    [[nodiscard]] void* completionEvent() const noexcept;
    [[nodiscard]] std::uint64_t completionValue() const noexcept;
    [[nodiscard]] std::span<const MetalNeuronCultureBufferView> buffers() const noexcept;
private:
    struct State;
    std::shared_ptr<State> state_;
    explicit MetalNeuronCultureAcceptedView(std::shared_ptr<State> state);
    friend class MetalNeuronCultureRuntime;
};

class MetalNeuronCultureTicket {
public:
    MetalNeuronCultureTicket() = default;
    MetalNeuronCultureTicket(MetalNeuronCultureTicket&&) noexcept;
    MetalNeuronCultureTicket& operator=(MetalNeuronCultureTicket&&) noexcept;
    ~MetalNeuronCultureTicket();

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool completed() const noexcept;
    [[nodiscard]] MetalNeuronCultureStatus wait() noexcept;
    [[nodiscard]] void* completionEvent() const noexcept;
    [[nodiscard]] std::uint64_t completionValue() const noexcept;
    [[nodiscard]] bool onCompleted(
        void* context, MetalNeuronCultureCompletion completion) noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
    explicit MetalNeuronCultureTicket(std::shared_ptr<State> state);
    friend class MetalNeuronCultureRuntime;
};

class MetalNeuronCultureRuntime {
public:
    MetalNeuronCultureRuntime();
    MetalNeuronCultureRuntime(MetalNeuronCultureRuntime&&) noexcept;
    MetalNeuronCultureRuntime& operator=(MetalNeuronCultureRuntime&&) noexcept;
    ~MetalNeuronCultureRuntime();

    [[nodiscard]] static MetalNeuronCultureRuntime create(
        const CompiledNeuronCulture& culture,
        void* metalDevice = nullptr
    );
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::string deviceName() const;
    [[nodiscard]] std::uint64_t residentBytes() const noexcept;
    [[nodiscard]] std::uint64_t peakResidentBytes() const noexcept;

    [[nodiscard]] MetalNeuronCultureTicket prepareTicks(
        std::uint32_t tickCount,
        std::uint32_t stimulationElectrode,
        float stimulationCurrent
    );
    [[nodiscard]] MetalNeuronCultureTicket prepareWindow(
        const NeuronCultureWindowRequest& request
    );
    [[nodiscard]] MetalNeuronCultureTicket prepareSupportWindow(
        const MetalNeuronCultureSupportRequest& request
    );
    [[nodiscard]] MetalNeuronCultureTicket prepareGrowth(
        std::uint32_t iterationCount
    );
    [[nodiscard]] MetalNeuronCultureStatus publishPrepared() noexcept;
    void rejectPrepared() noexcept;
    [[nodiscard]] MetalNeuronCultureStatus restoreAccepted(
        const NeuronCultureState& state
    ) noexcept;
    [[nodiscard]] MetalNeuronCultureAcceptedView acceptedView() const noexcept;
    [[nodiscard]] std::vector<std::uint32_t> acceptedElectrodeCountsTelemetry() const;

    // Qualification-only snapshot. Production embodiment consumes GPU buffers
    // directly and never uses this host publication path.
    [[nodiscard]] NeuronCultureState snapshotAcceptedForTesting() const;

private:
    // Pre-publication buffers are deliberately not public runtime authority.
    // The native aggregate bridge alone may retain them until the joint root
    // resolves and publishes.
    [[nodiscard]] MetalNeuronCultureAcceptedView preparedAcceptedView() const noexcept;
    friend struct MetalNeuronCultureRuntimeBridgeAccess;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* metalNeuronCultureStatusName(
    MetalNeuronCultureStatus status) noexcept;

} // namespace metalrobo
