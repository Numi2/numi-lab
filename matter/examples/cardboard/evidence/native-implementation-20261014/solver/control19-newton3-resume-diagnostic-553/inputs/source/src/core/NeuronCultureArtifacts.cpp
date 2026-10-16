#include "metalrobo/NeuronCultureArtifacts.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint32_t kEndian = 0x01020304u;
constexpr std::uint32_t kKindCulture = 1u;
constexpr std::uint32_t kKindCheckpoint = 2u;
constexpr std::array<char, 8u> kMagic{'N','C','U','L','T','0','0','1'};

struct ArtifactHeader {
    std::array<char, 8u> magic{};
    std::uint32_t version = 0u;
    std::uint32_t endian = 0u;
    std::uint32_t kind = 0u;
    std::uint32_t abiVersion = 0u;
    std::uint64_t payloadBytes = 0u;
    std::uint64_t cultureFingerprint = 0u;
    std::array<std::uint8_t, 32u> sha256{};
    std::array<std::uint64_t, 4u> reserved{};
};
static_assert(sizeof(ArtifactHeader) == 104u);

class Writer {
public:
    template <typename T> void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* begin = reinterpret_cast<const std::byte*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }
    void string(const std::string& value) {
        const std::uint32_t count = static_cast<std::uint32_t>(value.size());
        pod(count);
        if (!value.empty()) {
            const auto* begin = reinterpret_cast<const std::byte*>(value.data());
            bytes.insert(bytes.end(), begin, begin + value.size());
        }
    }
    template <typename T> void vector(const std::vector<T>& values) {
        const std::uint32_t elementBytes = sizeof(T);
        const std::uint64_t count = values.size();
        pod(elementBytes);
        pod(count);
        if (!values.empty()) {
            const auto* begin = reinterpret_cast<const std::byte*>(values.data());
            bytes.insert(bytes.end(), begin, begin + values.size() * sizeof(T));
        }
    }
    std::vector<std::byte> bytes;
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> value) : bytes(value) {}
    template <typename T> bool pod(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (remaining() < sizeof(T)) return false;
        std::memcpy(&value, bytes.data() + cursor, sizeof(T));
        cursor += sizeof(T);
        return true;
    }
    bool string(std::string& value) {
        std::uint32_t count = 0u;
        if (!pod(count) || count > remaining() || count > 1024u * 1024u) return false;
        value.assign(reinterpret_cast<const char*>(bytes.data() + cursor), count);
        cursor += count;
        return true;
    }
    template <typename T> bool vector(std::vector<T>& values, std::uint64_t maximum) {
        std::uint32_t elementBytes = 0u;
        std::uint64_t count = 0u;
        if (!pod(elementBytes) || !pod(count) || elementBytes != sizeof(T) ||
            count > maximum || count > std::numeric_limits<std::size_t>::max() ||
            count > remaining() / sizeof(T)) return false;
        values.resize(static_cast<std::size_t>(count));
        const std::size_t byteCount = values.size() * sizeof(T);
        if (byteCount != 0u) std::memcpy(values.data(), bytes.data() + cursor, byteCount);
        cursor += byteCount;
        return true;
    }
    [[nodiscard]] bool finished() const noexcept { return cursor == bytes.size(); }
private:
    [[nodiscard]] std::size_t remaining() const noexcept { return bytes.size() - cursor; }
    std::span<const std::byte> bytes;
    std::size_t cursor = 0u;
};

std::array<std::uint8_t, 32u> sha256(std::span<const std::byte> bytes) {
    std::array<std::uint8_t, 32u> result{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), result.data());
    return result;
}

NeuronCultureArtifactResult failure(
    NeuronCultureArtifactStatus status, std::string message
) {
    return {.status = status, .message = std::move(message)};
}

NeuronCultureArtifactResult publish(
    const std::filesystem::path& path,
    std::uint32_t kind,
    std::uint64_t cultureFingerprint,
    const std::vector<std::byte>& payload
) {
    if (path.empty() || cultureFingerprint == 0u) {
        return failure(NeuronCultureArtifactStatus::invalidInput, "artifact path or identity is invalid");
    }
    ArtifactHeader header{
        .magic = kMagic,
        .version = kNeuronCultureArtifactVersion,
        .endian = kEndian,
        .kind = kind,
        .abiVersion = MR_NEURON_CULTURE_ABI_VERSION,
        .payloadBytes = payload.size(),
        .cultureFingerprint = cultureFingerprint,
        .sha256 = sha256(payload),
    };
    const auto temporary = path.string() + ".tmp." + std::to_string(getpid());
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) return failure(NeuronCultureArtifactStatus::ioFailure, "could not open temporary artifact");
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    stream.flush();
    if (!stream) {
        stream.close();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return failure(NeuronCultureArtifactStatus::ioFailure, "failed writing artifact payload");
    }
    stream.close();
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return failure(NeuronCultureArtifactStatus::ioFailure, "could not atomically publish artifact");
    }
    return {
        .status = NeuronCultureArtifactStatus::success,
        .message = "artifact published",
        .sha256 = header.sha256,
        .payloadBytes = header.payloadBytes,
    };
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream stream;
    for (const unsigned char byte : value) {
        switch (byte) {
            case '\"': stream << "\\\""; break;
            case '\\': stream << "\\\\"; break;
            case '\b': stream << "\\b"; break;
            case '\f': stream << "\\f"; break;
            case '\n': stream << "\\n"; break;
            case '\r': stream << "\\r"; break;
            case '\t': stream << "\\t"; break;
            default:
                if (byte < 0x20u) {
                    stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(byte) << std::dec;
                } else {
                    stream << static_cast<char>(byte);
                }
        }
    }
    return stream.str();
}

NeuronCultureArtifactResult publishText(
    const std::filesystem::path& path, const std::string& payload
) {
    if (path.empty() || payload.empty()) {
        return failure(NeuronCultureArtifactStatus::invalidInput,
                       "manifest path or payload is invalid");
    }
    const auto temporary = path.string() + ".tmp." + std::to_string(getpid());
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    stream.flush();
    if (!stream) {
        stream.close();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return failure(NeuronCultureArtifactStatus::ioFailure,
                       "failed writing run manifest");
    }
    stream.close();
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return failure(NeuronCultureArtifactStatus::ioFailure,
                       "could not atomically publish run manifest");
    }
    const auto bytes = std::as_bytes(std::span(payload.data(), payload.size()));
    return {.status = NeuronCultureArtifactStatus::success,
            .message = "run manifest published", .sha256 = sha256(bytes),
            .payloadBytes = payload.size()};
}

NeuronCultureArtifactResult load(
    const std::filesystem::path& path,
    std::uint32_t kind,
    ArtifactHeader& header,
    std::vector<std::byte>& payload
) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return failure(NeuronCultureArtifactStatus::ioFailure, "could not open artifact");
    const auto end = stream.tellg();
    if (end < static_cast<std::streamoff>(sizeof(header))) {
        return failure(NeuronCultureArtifactStatus::corruptPayload, "artifact is shorter than its header");
    }
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.magic != kMagic || header.version != kNeuronCultureArtifactVersion ||
        header.endian != kEndian || header.kind != kind ||
        header.abiVersion != MR_NEURON_CULTURE_ABI_VERSION) {
        return failure(NeuronCultureArtifactStatus::unsupportedVersion, "artifact format is unsupported");
    }
    const std::uint64_t total = static_cast<std::uint64_t>(end);
    if (header.payloadBytes != total - sizeof(header) ||
        header.payloadBytes > 2ull * 1024ull * 1024ull * 1024ull ||
        header.cultureFingerprint == 0u) {
        return failure(NeuronCultureArtifactStatus::capacityOverflow, "artifact payload length is invalid");
    }
    payload.resize(static_cast<std::size_t>(header.payloadBytes));
    stream.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!stream || sha256(payload) != header.sha256) {
        return failure(NeuronCultureArtifactStatus::corruptPayload, "artifact SHA-256 is invalid");
    }
    return {
        .status = NeuronCultureArtifactStatus::success,
        .message = "artifact loaded",
        .sha256 = header.sha256,
        .payloadBytes = header.payloadBytes,
    };
}

void writeState(Writer& writer, const NeuronCultureState& state) {
    writer.pod(state.generation);
    writer.pod(state.tick);
    writer.pod(state.growthIteration);
    writer.vector(state.membrane); writer.vector(state.refractory);
    writer.vector(state.preTrace); writer.vector(state.postTrace);
    writer.vector(state.weights); writer.vector(state.depression);
    writer.vector(state.spikes); writer.vector(state.spikeHistory);
    writer.vector(state.electrodeSpikeCounts);
    writer.vector(state.phase); writer.vector(state.tubulin);
}

bool readState(Reader& reader, const CompiledNeuronCulture& culture, NeuronCultureState& state) {
    const auto& h = culture.header();
    const std::uint64_t cells = static_cast<std::uint64_t>(h.growthWidth) * h.growthHeight;
    return reader.pod(state.generation) && reader.pod(state.tick) &&
        reader.pod(state.growthIteration) &&
        reader.vector(state.membrane, h.neuronCount) &&
        reader.vector(state.refractory, h.neuronCount) &&
        reader.vector(state.preTrace, h.neuronCount) &&
        reader.vector(state.postTrace, h.neuronCount) &&
        reader.vector(state.weights, h.synapseCount) &&
        reader.vector(state.depression, h.synapseCount) &&
        reader.vector(state.spikes, h.neuronCount) &&
        reader.vector(state.spikeHistory, static_cast<std::uint64_t>(h.neuronCount) * 256u) &&
        reader.vector(state.electrodeSpikeCounts, h.electrodeCount) &&
        reader.vector(state.phase, cells) && reader.vector(state.tubulin, cells) &&
        reader.finished();
}

template <typename T> bool sameVector(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

} // namespace

std::string NeuronCultureArtifactResult::sha256Hex() const {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto value : sha256) stream << std::setw(2) << static_cast<unsigned>(value);
    return stream.str();
}

NeuronCultureArtifactResult writeCompiledNeuronCulture(
    const CompiledNeuronCulture& culture, const std::filesystem::path& path
) {
    if (!culture.valid()) return failure(NeuronCultureArtifactStatus::invalidInput, "culture is invalid");
    Writer writer;
    writer.string(culture.id()); writer.string(culture.source());
    writer.string(culture.sourceRevision()); writer.string(culture.sourceLicense());
    writer.pod(culture.header()); writer.pod(culture.growth());
    writer.vector(std::vector(culture.neurons().begin(), culture.neurons().end()));
    writer.vector(std::vector(culture.synapses().begin(), culture.synapses().end()));
    writer.vector(std::vector(culture.electrodes().begin(), culture.electrodes().end()));
    return publish(path, kKindCulture, culture.fingerprint(), writer.bytes);
}

NeuronCultureArtifactResult readCompiledNeuronCulture(
    const std::filesystem::path& path, CompiledNeuronCulture& output
) {
    ArtifactHeader file{};
    std::vector<std::byte> payload;
    auto result = load(path, kKindCulture, file, payload);
    if (!result.succeeded()) return result;
    Reader reader(payload);
    NeuronCulturePack pack;
    MRNeuronCultureHeaderGPU header{};
    MRNeuronCultureGrowthGPU growth{};
    if (!reader.string(pack.id) || !reader.string(pack.source) ||
        !reader.string(pack.sourceRevision) || !reader.string(pack.sourceLicense) ||
        !reader.pod(header) || !reader.pod(growth) ||
        !reader.vector(pack.neurons, 1'000'000u) ||
        !reader.vector(pack.synapses, 16'000'000u) ||
        !reader.vector(pack.electrodes, MR_NEURON_CULTURE_MAX_ELECTRODES) ||
        !reader.finished()) {
        return failure(NeuronCultureArtifactStatus::corruptPayload, "culture payload structure is invalid");
    }
    pack.formatVersion = kNeuronCulturePackFormatVersion;
    pack.seed = header.seed;
    pack.growth = {
        .width = growth.width, .height = growth.height, .stage = growth.stage,
        .newtonIterations = growth.newtonIterations, .timestep = growth.timestep,
        .phaseMobility = growth.phaseMobility,
        .interfaceCoefficient = growth.interfaceCoefficient,
        .tubulinDiffusion = growth.tubulinDiffusion, .tubulinDecay = growth.tubulinDecay,
        .tubulinSource = growth.tubulinSource, .growthDrive = growth.growthDrive,
        .newtonTolerance = growth.newtonTolerance,
    };
    pack.network = {
        .neuralTimestepSeconds = header.neuralTimestepSeconds,
        .membraneTimeConstantSeconds = header.membraneTimeConstantSeconds,
        .restingPotential = header.restingPotential, .resetPotential = header.resetPotential,
        .thresholdPotential = header.thresholdPotential,
        .refractorySeconds = header.refractorySeconds,
        .traceTimeConstantSeconds = header.traceTimeConstantSeconds,
        .depressionRecoverySeconds = header.depressionRecoverySeconds,
        .stdpPotentiation = header.stdpPotentiation,
        .stdpDepression = header.stdpDepression,
        .minimumWeight = header.minimumWeight, .maximumWeight = header.maximumWeight,
        .synapticCurrentScale = header.synapticCurrentScale,
        .preSpikeSuppressionTimeConstantSeconds =
            header.preSpikeSuppressionTimeConstantSeconds,
        .postSpikeSuppressionTimeConstantSeconds =
            header.postSpikeSuppressionTimeConstantSeconds,
    };
    CompiledNeuronCulture candidate;
    const auto diagnostics = compileNeuronCulture(pack, candidate);
    if (!diagnostics.succeeded() || candidate.fingerprint() != header.cultureFingerprint ||
        candidate.fingerprint() != file.cultureFingerprint) {
        return failure(NeuronCultureArtifactStatus::identityMismatch, "culture identity does not recompute");
    }
    output = std::move(candidate);
    return result;
}

NeuronCultureArtifactResult writeNeuronCultureCheckpoint(
    const CompiledNeuronCulture& culture, const NeuronCultureState& accepted,
    const std::filesystem::path& path
) {
    NeuronCultureReference validator(culture);
    if (!validator.restoreAccepted(accepted)) {
        return failure(NeuronCultureArtifactStatus::invalidInput, "accepted culture state is invalid");
    }
    Writer writer;
    writeState(writer, accepted);
    return publish(path, kKindCheckpoint, culture.fingerprint(), writer.bytes);
}

NeuronCultureArtifactResult readNeuronCultureCheckpoint(
    const CompiledNeuronCulture& culture, const std::filesystem::path& path,
    NeuronCultureState& output
) {
    if (!culture.valid()) return failure(NeuronCultureArtifactStatus::invalidInput, "culture is invalid");
    ArtifactHeader file{};
    std::vector<std::byte> payload;
    auto result = load(path, kKindCheckpoint, file, payload);
    if (!result.succeeded()) return result;
    if (file.cultureFingerprint != culture.fingerprint()) {
        return failure(NeuronCultureArtifactStatus::identityMismatch, "checkpoint culture identity mismatches");
    }
    Reader reader(payload);
    NeuronCultureState candidate;
    if (!readState(reader, culture, candidate)) {
        return failure(NeuronCultureArtifactStatus::corruptPayload, "checkpoint state shape is invalid");
    }
    NeuronCultureReference validator(culture);
    if (!validator.restoreAccepted(candidate)) {
        return failure(NeuronCultureArtifactStatus::corruptPayload, "checkpoint state values are invalid");
    }
    output = std::move(candidate);
    return result;
}

NeuronCultureArtifactResult writeNeuronCultureRunManifest(
    const NeuronCultureRunManifest& manifest, const std::filesystem::path& path
) {
    const auto digest = [](const std::string& value) {
        return value.empty() || (value.size() == 64u && std::all_of(
            value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c); }));
    };
    const auto revision = [](const std::string& value) {
        return (value.size() == 40u || value.size() == 64u) && std::all_of(
            value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c); });
    };
    if (manifest.version != 1u || manifest.cultureFingerprint == 0u ||
        manifest.startingStateFingerprint == 0u ||
        manifest.acceptedStateFingerprint == 0u ||
        !digest(manifest.cultureSHA256) ||
        !digest(manifest.startingCheckpointSHA256) ||
        !digest(manifest.checkpointSHA256) ||
        !digest(manifest.metallibSHA256) ||
        !revision(manifest.metalRoboRevision) ||
        !revision(manifest.numiBrainRevision) ||
        !revision(manifest.numanXRevision) ||
        manifest.device.empty() || manifest.operatingSystem.empty() || manifest.sdk.empty() ||
        manifest.command.empty() || manifest.protocol.empty() || !manifest.simulationOnly ||
        manifest.limitations.empty()) {
        return failure(NeuronCultureArtifactStatus::invalidInput,
                       "run manifest identity or simulation boundary is invalid");
    }
    for (const auto& [name, value] : manifest.measurements) {
        if (name.empty() || !std::isfinite(value)) {
            return failure(NeuronCultureArtifactStatus::invalidInput,
                           "run manifest measurement is invalid");
        }
    }
    std::ostringstream json;
    json << std::setprecision(17)
         << "{\n  \"schema\": \"numi.neuron-culture.run.v1\",\n"
         << "  \"version\": 1,\n"
         << "  \"culture_fingerprint\": " << manifest.cultureFingerprint << ",\n"
         << "  \"starting_state_fingerprint\": "
         << manifest.startingStateFingerprint << ",\n"
         << "  \"accepted_state_fingerprint\": "
         << manifest.acceptedStateFingerprint << ",\n"
         << "  \"culture_sha256\": \"" << jsonEscape(manifest.cultureSHA256) << "\",\n"
         << "  \"starting_checkpoint_sha256\": \""
         << jsonEscape(manifest.startingCheckpointSHA256) << "\",\n"
         << "  \"checkpoint_sha256\": \"" << jsonEscape(manifest.checkpointSHA256) << "\",\n"
         << "  \"revisions\": {\"metalrobo\": \"" << jsonEscape(manifest.metalRoboRevision)
         << "\", \"numibrain\": \"" << jsonEscape(manifest.numiBrainRevision)
         << "\", \"numan_x\": \"" << jsonEscape(manifest.numanXRevision) << "\"},\n"
         << "  \"runtime\": {\"metallib_sha256\": \"" << jsonEscape(manifest.metallibSHA256)
         << "\", \"device\": \"" << jsonEscape(manifest.device)
         << "\", \"os\": \"" << jsonEscape(manifest.operatingSystem)
         << "\", \"sdk\": \"" << jsonEscape(manifest.sdk) << "\"},\n"
         << "  \"command\": \"" << jsonEscape(manifest.command) << "\",\n"
         << "  \"protocol\": \"" << jsonEscape(manifest.protocol) << "\",\n"
         << "  \"deterministic_replay\": " << (manifest.deterministicReplay ? "true" : "false") << ",\n"
         << "  \"simulation_only\": true,\n  \"checkpoints\": [";
    for (std::size_t index = 0u; index < manifest.checkpoints.size(); ++index) {
        if (index != 0u) json << ", ";
        json << "\"" << jsonEscape(manifest.checkpoints[index]) << "\"";
    }
    json << "],\n  \"measurements\": {";
    for (std::size_t index = 0u; index < manifest.measurements.size(); ++index) {
        if (index != 0u) json << ", ";
        json << "\"" << jsonEscape(manifest.measurements[index].first) << "\": "
             << manifest.measurements[index].second;
    }
    json << "},\n  \"limitations\": [";
    for (std::size_t index = 0u; index < manifest.limitations.size(); ++index) {
        if (index != 0u) json << ", ";
        json << "\"" << jsonEscape(manifest.limitations[index]) << "\"";
    }
    json << "]\n}\n";
    return publishText(path, json.str());
}

NeuronCultureArtifactResult validateNeuronCultureRunManifest(
    const std::filesystem::path& path,
    const std::uint64_t expectedCultureFingerprint,
    const std::string& expectedProtocol
) {
    if (path.empty() || expectedCultureFingerprint == 0u ||
        expectedProtocol.empty()) {
        return failure(NeuronCultureArtifactStatus::invalidInput,
                       "run manifest validation identity is invalid");
    }
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return failure(NeuronCultureArtifactStatus::ioFailure,
                       "could not open run manifest");
    }
    const auto end = stream.tellg();
    if (end <= 0 || end > 1024 * 1024) {
        return failure(NeuronCultureArtifactStatus::capacityOverflow,
                       "run manifest length is invalid");
    }
    std::string json(static_cast<std::size_t>(end), '\0');
    stream.seekg(0);
    stream.read(json.data(), static_cast<std::streamsize>(json.size()));
    if (!stream ||
        json.find("\"schema\": \"numi.neuron-culture.run.v1\"") ==
            std::string::npos ||
        json.find("\"version\": 1") == std::string::npos ||
        json.find("\"simulation_only\": true") == std::string::npos ||
        json.find("\"protocol\": \"" + expectedProtocol + "\"") ==
            std::string::npos) {
        return failure(NeuronCultureArtifactStatus::unsupportedVersion,
                       "run manifest schema or simulation boundary is invalid");
    }
    constexpr std::string_view key = "\"culture_fingerprint\":";
    const auto keyPosition = json.find(key);
    if (keyPosition == std::string::npos ||
        json.find(key, keyPosition + key.size()) != std::string::npos) {
        return failure(NeuronCultureArtifactStatus::corruptPayload,
                       "run manifest culture identity is absent or duplicated");
    }
    const char* begin = json.data() + keyPosition + key.size();
    const char* finish = json.data() + json.size();
    while (begin != finish && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
    std::uint64_t cultureFingerprint = 0u;
    const auto parsed = std::from_chars(begin, finish, cultureFingerprint);
    if (parsed.ec != std::errc{} || cultureFingerprint != expectedCultureFingerprint) {
        return failure(NeuronCultureArtifactStatus::identityMismatch,
                       "run manifest culture identity mismatches");
    }
    constexpr std::string_view startingKey = "\"starting_state_fingerprint\":";
    const auto startingPosition = json.find(startingKey);
    if (startingPosition != std::string::npos &&
        json.find(startingKey, startingPosition + startingKey.size()) !=
            std::string::npos) {
        return failure(NeuronCultureArtifactStatus::corruptPayload,
                       "run manifest starting-state identity is duplicated");
    }
    if (startingPosition != std::string::npos) {
        begin = json.data() + startingPosition + startingKey.size();
        while (begin != finish &&
               std::isspace(static_cast<unsigned char>(*begin))) ++begin;
        std::uint64_t startingStateFingerprint = 0u;
        const auto startingParsed = std::from_chars(
            begin, finish, startingStateFingerprint);
        if (startingParsed.ec != std::errc{} ||
            startingStateFingerprint == 0u) {
            return failure(NeuronCultureArtifactStatus::corruptPayload,
                           "run manifest starting-state identity is invalid");
        }
    }
    constexpr std::string_view stateKey = "\"accepted_state_fingerprint\":";
    const auto statePosition = json.find(stateKey);
    if (statePosition == std::string::npos ||
        json.find(stateKey, statePosition + stateKey.size()) != std::string::npos) {
        return failure(NeuronCultureArtifactStatus::corruptPayload,
                       "run manifest accepted-state identity is absent or duplicated");
    }
    begin = json.data() + statePosition + stateKey.size();
    while (begin != finish && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
    std::uint64_t acceptedStateFingerprint = 0u;
    const auto stateParsed = std::from_chars(
        begin, finish, acceptedStateFingerprint);
    if (stateParsed.ec != std::errc{} || acceptedStateFingerprint == 0u) {
        return failure(NeuronCultureArtifactStatus::corruptPayload,
                       "run manifest accepted-state identity is invalid");
    }
    const auto bytes = std::as_bytes(std::span(json.data(), json.size()));
    return {.status = NeuronCultureArtifactStatus::success,
            .message = "run manifest validated", .sha256 = sha256(bytes),
            .payloadBytes = json.size()};
}

bool sameNeuronCultureState(const NeuronCultureState& a, const NeuronCultureState& b) noexcept {
    return a.generation == b.generation && a.tick == b.tick &&
        a.growthIteration == b.growthIteration &&
        sameVector(a.membrane,b.membrane) && sameVector(a.refractory,b.refractory) &&
        sameVector(a.preTrace,b.preTrace) && sameVector(a.postTrace,b.postTrace) &&
        sameVector(a.weights,b.weights) && sameVector(a.depression,b.depression) &&
        sameVector(a.spikes,b.spikes) && sameVector(a.spikeHistory,b.spikeHistory) &&
        sameVector(a.electrodeSpikeCounts,b.electrodeSpikeCounts) &&
        sameVector(a.phase,b.phase) && sameVector(a.tubulin,b.tubulin);
}

std::uint64_t fingerprintNeuronCultureState(
    const NeuronCultureState& state
) noexcept {
    try {
        Writer writer;
        writeState(writer, state);
        std::uint64_t hash = 14695981039346656037ull;
        for (const std::byte value : writer.bytes) {
            hash ^= std::to_integer<std::uint8_t>(value);
            hash *= 1099511628211ull;
        }
        return hash == 0u ? 14695981039346656037ull : hash;
    } catch (...) {
        return 0u;
    }
}

const char* neuronCultureArtifactStatusName(NeuronCultureArtifactStatus status) noexcept {
    switch (status) {
        case NeuronCultureArtifactStatus::success: return "success";
        case NeuronCultureArtifactStatus::invalidInput: return "invalid_input";
        case NeuronCultureArtifactStatus::unsupportedVersion: return "unsupported_version";
        case NeuronCultureArtifactStatus::corruptPayload: return "corrupt_payload";
        case NeuronCultureArtifactStatus::identityMismatch: return "identity_mismatch";
        case NeuronCultureArtifactStatus::capacityOverflow: return "capacity_overflow";
        case NeuronCultureArtifactStatus::ioFailure: return "io_failure";
    }
    return "unknown";
}

} // namespace metalrobo
