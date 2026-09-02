#pragma once

#include "metalrobo/NeuronCulture.hpp"

#include <cstdint>
#include <memory>
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

class MetalNeuronCultureTicket {
public:
    MetalNeuronCultureTicket() = default;
    MetalNeuronCultureTicket(MetalNeuronCultureTicket&&) noexcept;
    MetalNeuronCultureTicket& operator=(MetalNeuronCultureTicket&&) noexcept;
    ~MetalNeuronCultureTicket();

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool completed() const noexcept;
    [[nodiscard]] MetalNeuronCultureStatus wait() noexcept;

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

    [[nodiscard]] MetalNeuronCultureTicket prepareTicks(
        std::uint32_t tickCount,
        std::uint32_t stimulationElectrode,
        float stimulationCurrent
    );
    [[nodiscard]] MetalNeuronCultureTicket prepareGrowth(
        std::uint32_t iterationCount
    );
    [[nodiscard]] MetalNeuronCultureStatus publishPrepared() noexcept;
    void rejectPrepared() noexcept;

    // Qualification-only snapshot. Production embodiment consumes GPU buffers
    // directly and never uses this host publication path.
    [[nodiscard]] NeuronCultureState snapshotAcceptedForTesting() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* metalNeuronCultureStatusName(
    MetalNeuronCultureStatus status) noexcept;

} // namespace metalrobo
