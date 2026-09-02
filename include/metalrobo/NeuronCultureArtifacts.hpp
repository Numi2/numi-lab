#pragma once

#include "metalrobo/NeuronCulture.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kNeuronCultureArtifactVersion = 1u;

enum class NeuronCultureArtifactStatus : std::uint32_t {
    success = 0u,
    invalidInput,
    unsupportedVersion,
    corruptPayload,
    identityMismatch,
    capacityOverflow,
    ioFailure,
};

struct NeuronCultureArtifactResult {
    NeuronCultureArtifactStatus status = NeuronCultureArtifactStatus::success;
    std::string message;
    std::array<std::uint8_t, 32u> sha256{};
    std::uint64_t payloadBytes = 0u;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NeuronCultureArtifactStatus::success;
    }
    [[nodiscard]] std::string sha256Hex() const;
};

struct NeuronCultureRunManifest {
    std::uint32_t version = 1u;
    std::uint64_t cultureFingerprint = 0u;
    std::uint64_t startingStateFingerprint = 0u;
    std::uint64_t acceptedStateFingerprint = 0u;
    std::string cultureSHA256;
    std::string startingCheckpointSHA256;
    std::string checkpointSHA256;
    std::string metalRoboRevision;
    std::string numiBrainRevision;
    std::string numanXRevision;
    std::string metallibSHA256;
    std::string device;
    std::string operatingSystem;
    std::string sdk;
    std::string command;
    std::string protocol;
    std::vector<std::string> checkpoints;
    std::vector<std::pair<std::string, double>> measurements;
    bool deterministicReplay = false;
    bool simulationOnly = true;
    std::vector<std::string> limitations;
};

[[nodiscard]] NeuronCultureArtifactResult writeCompiledNeuronCulture(
    const CompiledNeuronCulture& culture,
    const std::filesystem::path& path
);

[[nodiscard]] NeuronCultureArtifactResult readCompiledNeuronCulture(
    const std::filesystem::path& path,
    CompiledNeuronCulture& output
);

[[nodiscard]] NeuronCultureArtifactResult writeNeuronCultureCheckpoint(
    const CompiledNeuronCulture& culture,
    const NeuronCultureState& accepted,
    const std::filesystem::path& path
);

[[nodiscard]] NeuronCultureArtifactResult readNeuronCultureCheckpoint(
    const CompiledNeuronCulture& culture,
    const std::filesystem::path& path,
    NeuronCultureState& output
);

[[nodiscard]] NeuronCultureArtifactResult writeNeuronCultureRunManifest(
    const NeuronCultureRunManifest& manifest,
    const std::filesystem::path& path
);

// Validates the immutable identity fields of a canonical run manifest without
// treating it as simulation authority. The native embodiment may retain this
// provenance artifact, but its GPU schedule still comes from the exact root.
[[nodiscard]] NeuronCultureArtifactResult validateNeuronCultureRunManifest(
    const std::filesystem::path& path,
    std::uint64_t expectedCultureFingerprint,
    const std::string& expectedProtocol
);

[[nodiscard]] bool sameNeuronCultureState(
    const NeuronCultureState& left,
    const NeuronCultureState& right
) noexcept;

// Stable FNV-1a identity of the complete accepted-state payload used by the
// deterministic qualification rerun. This is not a substitute for the
// checkpoint's SHA-256 content identity.
[[nodiscard]] std::uint64_t fingerprintNeuronCultureState(
    const NeuronCultureState& state
) noexcept;

[[nodiscard]] const char* neuronCultureArtifactStatusName(
    NeuronCultureArtifactStatus status
) noexcept;

} // namespace metalrobo
