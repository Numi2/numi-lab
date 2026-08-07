#include "metalrobo/HyperPolicyPacks.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <exception>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

namespace metalrobo {
namespace {

constexpr std::array<char, 8u> kMagic{
    'M', 'R', 'L', 'E', 'A', 'R', 'N', '\0',
};
constexpr std::uint32_t kHyperPolicyKind = 9u;
constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;
constexpr std::size_t kMaximumIOChunk = 64u * 1024u * 1024u;

struct Header {
    std::array<char, 8u> magic = kMagic;
    std::uint32_t formatVersion = kHyperPolicyPackFormatVersion;
    std::uint32_t kind = kHyperPolicyKind;
    std::uint64_t payloadBytes = 0u;
    std::uint64_t contentHash = 0u;
};
static_assert(sizeof(Header) == 32u);
static_assert(std::is_trivially_copyable_v<Header>);
static_assert(std::endian::native == std::endian::little);

LearningPackResult fail(LearningPackStatus status, std::string message) {
    return {.status = status, .message = std::move(message)};
}

std::uint64_t hashPayload(std::span<const std::byte> payload) noexcept {
    std::uint64_t hash = kFNVOffset;
    for (const std::byte value : payload) {
        hash ^= std::to_integer<unsigned char>(value);
        hash *= kFNVPrime;
    }
    return hash;
}

class Writer {
public:
    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* first = reinterpret_cast<const std::byte*>(&value);
        data_.insert(data_.end(), first, first + sizeof(value));
    }

    void string(std::string_view value) {
        pod<std::uint64_t>(value.size());
        if (!value.empty()) {
            const auto* first =
                reinterpret_cast<const std::byte*>(value.data());
            data_.insert(data_.end(), first, first + value.size());
        }
    }

    template <typename T>
    void vector(std::span<const T> values) {
        static_assert(std::is_trivially_copyable_v<T>);
        pod<std::uint64_t>(values.size());
        if (!values.empty()) {
            const auto* first =
                reinterpret_cast<const std::byte*>(values.data());
            data_.insert(data_.end(), first, first + values.size_bytes());
        }
    }

    [[nodiscard]] std::vector<std::byte> take() && {
        return std::move(data_);
    }

private:
    std::vector<std::byte> data_;
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> payload) : payload_(payload) {}

    template <typename T>
    bool pod(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (sizeof(T) > payload_.size() - offset_) {
            return false;
        }
        std::memcpy(&value, payload_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t count = 0u;
        if (!pod(count) || count > std::numeric_limits<std::uint32_t>::max() ||
            count > payload_.size() - offset_) {
            return false;
        }
        value.assign(
            reinterpret_cast<const char*>(payload_.data() + offset_),
            static_cast<std::size_t>(count)
        );
        offset_ += static_cast<std::size_t>(count);
        return true;
    }

    template <typename T>
    bool vector(std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::uint64_t count = 0u;
        if (!pod(count) || count > std::numeric_limits<std::uint32_t>::max() ||
            count > (payload_.size() - offset_) / sizeof(T)) {
            return false;
        }
        values.resize(static_cast<std::size_t>(count));
        const std::size_t bytes = values.size() * sizeof(T);
        if (bytes != 0u) {
            std::memcpy(values.data(), payload_.data() + offset_, bytes);
        }
        offset_ += bytes;
        return true;
    }

    [[nodiscard]] bool finished() const noexcept {
        return offset_ == payload_.size();
    }

private:
    std::span<const std::byte> payload_;
    std::size_t offset_ = 0u;
};

bool finite(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

bool strictlyIncreasingUnitPhases(std::span<const float> values) {
    if (values.size() < 2u ||
        std::abs(values.front()) > 1.0e-6f ||
        std::abs(values.back() - 1.0f) > 1.0e-6f) {
        return false;
    }
    return std::adjacent_find(
        values.begin(), values.end(),
        [](const float left, const float right) { return !(left < right); }
    ) == values.end();
}

bool supportedActivation(const PolicyActivation activation) {
    switch (activation) {
    case PolicyActivation::identity:
    case PolicyActivation::relu:
    case PolicyActivation::tanh:
    case PolicyActivation::elu:
    case PolicyActivation::silu:
        return true;
    }
    return false;
}

LearningPackResult validate(const HyperPolicyPack& pack) {
    std::uint64_t coefficientCount = 0u;
    std::uint32_t expectedInput = 0u;
    for (std::size_t index = 0u; index < pack.layers.size(); ++index) {
        const HyperPolicyLayer& layer = pack.layers[index];
        coefficientCount += layer.rank;
        const bool final = index + 1u == pack.layers.size();
        if (layer.inputCount == 0u || layer.outputCount == 0u ||
            layer.rank == 0u ||
            layer.rank > std::min(layer.inputCount, layer.outputCount) ||
            (index != 0u && layer.inputCount != expectedInput) ||
            !supportedActivation(layer.activation) ||
            (final && layer.activation != PolicyActivation::identity) ||
            (!final && layer.activation == PolicyActivation::identity) ||
            layer.weights.size() !=
                static_cast<std::size_t>(layer.inputCount) * layer.outputCount ||
            layer.bias.size() != layer.outputCount ||
            layer.adapterDown.size() !=
                static_cast<std::size_t>(layer.rank) * layer.inputCount ||
            layer.adapterUp.size() !=
                static_cast<std::size_t>(layer.outputCount) * layer.rank ||
            layer.adapterBiasBasis.size() != layer.adapterUp.size() ||
            !finite(layer.weights) || !finite(layer.bias) ||
            !finite(layer.adapterDown) || !finite(layer.adapterUp) ||
            !finite(layer.adapterBiasBasis)) {
            return fail(
                LearningPackStatus::invalidPack,
                "HyperPolicyPack contains an invalid low-rank actor layer"
            );
        }
        expectedInput = layer.outputCount;
    }
    const std::uint64_t knots = pack.knotPhases.size();
    const std::uint64_t references = pack.referencePhases.size();
    const std::uint64_t actions = pack.actionBias.size();
    if (pack.id.empty() || pack.revision == 0u ||
        !pack.contract.exact() || pack.sourceMotionFingerprint == 0u ||
        pack.layers.empty() || coefficientCount == 0u ||
        knots < 2u || references < 2u || actions == 0u ||
        pack.signatureCount == 0u || pack.contactTrackCount == 0u ||
        pack.contactTrackCount > 32u ||
        pack.contactGroupIndices.size() != pack.contactTrackCount ||
        pack.coefficientLimits.size() != coefficientCount ||
        pack.coefficientKnots.size() != knots * coefficientCount ||
        pack.coefficientTangents.size() != knots * coefficientCount ||
        pack.authorityKnots.size() != knots * actions ||
        pack.authorityTangents.size() != knots * actions ||
        pack.phaseRateKnots.size() != knots ||
        pack.phaseRateTangents.size() != knots ||
        pack.actionScale.size() != actions ||
        (!pack.actionLogStandardDeviation.empty() &&
         pack.actionLogStandardDeviation.size() != actions) ||
        pack.referenceActions.size() != references * actions ||
        pack.referenceSignatures.size() !=
            references * pack.signatureCount ||
        pack.signatureWeights.size() != pack.signatureCount ||
        pack.referenceContactMasks.size() != references ||
        pack.actionLower.size() != actions ||
        pack.actionUpper.size() != actions ||
        pack.maximumActionRate.size() != actions ||
        !std::isfinite(pack.observationClip) || pack.observationClip <= 0.0f ||
        !std::isfinite(pack.actionClip) || pack.actionClip <= 0.0f ||
        !std::isfinite(pack.maximumPhaseAdvancePerStep) ||
        pack.maximumPhaseAdvancePerStep < 0.0f ||
        !std::isfinite(pack.phaseAlignmentBlend) ||
        pack.phaseAlignmentBlend < 0.0f || pack.phaseAlignmentBlend > 1.0f ||
        !std::isfinite(pack.phaseAlignmentHuberDelta) ||
        pack.phaseAlignmentHuberDelta <= 0.0f ||
        !std::isfinite(pack.controlTimeStep) || pack.controlTimeStep <= 0.0f) {
        return fail(
            LearningPackStatus::invalidPack,
            "HyperPolicyPack identity, dimensions, contract, or cadence is invalid"
        );
    }
    const std::array<std::span<const float>, 18u> tables{
        pack.observationMean,
        pack.observationInverseStandardDeviation,
        pack.coefficientLimits,
        pack.actionBias,
        pack.actionScale,
        pack.actionLogStandardDeviation,
        pack.knotPhases,
        pack.coefficientKnots,
        pack.coefficientTangents,
        pack.authorityKnots,
        pack.authorityTangents,
        pack.phaseRateKnots,
        pack.phaseRateTangents,
        pack.referencePhases,
        pack.referenceActions,
        pack.referenceSignatures,
        pack.signatureWeights,
        pack.maximumActionRate,
    };
    if (!std::all_of(tables.begin(), tables.end(), finite) ||
        !finite(pack.actionLower) || !finite(pack.actionUpper)) {
        return fail(
            LearningPackStatus::invalidPack,
            "HyperPolicyPack contains non-finite numeric tables"
        );
    }
    if (!strictlyIncreasingUnitPhases(pack.knotPhases) ||
        !strictlyIncreasingUnitPhases(pack.referencePhases) ||
        std::any_of(
            pack.observationInverseStandardDeviation.begin(),
            pack.observationInverseStandardDeviation.end(),
            [](const float value) { return !(value > 0.0f); }
        ) ||
        std::any_of(
            pack.coefficientLimits.begin(), pack.coefficientLimits.end(),
            [](const float value) { return !(value > 0.0f); }
        ) ||
        std::any_of(
            pack.authorityKnots.begin(), pack.authorityKnots.end(),
            [](const float value) { return value < 0.0f || value > 1.0f; }
        ) ||
        std::any_of(
            pack.phaseRateKnots.begin(), pack.phaseRateKnots.end(),
            [](const float value) { return value < 0.0f; }
        ) ||
        std::any_of(
            pack.signatureWeights.begin(), pack.signatureWeights.end(),
            [](const float value) { return !(value > 0.0f); }
        ) ||
        std::any_of(
            pack.maximumActionRate.begin(), pack.maximumActionRate.end(),
            [](const float value) { return !(value > 0.0f); }
        ) ||
        (!pack.actionLogStandardDeviation.empty() && std::any_of(
            pack.actionLogStandardDeviation.begin(),
            pack.actionLogStandardDeviation.end(),
            [](const float value) { return value < -5.0f || value > 2.0f; }
        ))) {
        return fail(
            LearningPackStatus::invalidPack,
            "HyperPolicyPack normalized phases, bounds, or generated controls are invalid"
        );
    }
    for (std::size_t action = 0u; action < actions; ++action) {
        if (!(pack.actionLower[action] < pack.actionUpper[action]) ||
            std::abs(pack.actionScale[action]) <= 1.0e-12f) {
            return fail(
                LearningPackStatus::invalidPack,
                "HyperPolicyPack action transform or physical envelope is invalid"
            );
        }
    }
    const std::uint32_t contactMask = pack.contactTrackCount == 32u
        ? std::numeric_limits<std::uint32_t>::max()
        : ((1u << pack.contactTrackCount) - 1u);
    float previousEventPhase = -1.0f;
    for (const HyperPolicyEventGuard& event : pack.events) {
        if (!std::isfinite(event.phase) || event.phase < 0.0f ||
            event.phase > 1.0f || event.phase < previousEventPhase ||
            !std::isfinite(event.confidence) || event.confidence < 0.0f ||
            event.confidence > 1.0f || event.minimumDwellSteps == 0u ||
            ((event.requiredContactOnMask |
              event.requiredContactOffMask) & ~contactMask) != 0u ||
            (event.requiredContactOnMask &
             event.requiredContactOffMask) != 0u) {
            return fail(
                LearningPackStatus::invalidPack,
                "HyperPolicyPack event guard contract is invalid"
            );
        }
        previousEventPhase = event.phase;
    }
    return {};
}

std::vector<std::byte> serialize(const HyperPolicyPack& pack) {
    Writer writer;
    writer.string(pack.id);
    writer.pod(pack.revision);
    writer.pod(pack.contract.version);
    writer.pod(pack.contract.worldFingerprint);
    writer.pod(pack.contract.taskFingerprint);
    writer.pod(pack.contract.observationFingerprint);
    writer.pod(pack.contract.actionFingerprint);
    writer.pod(pack.sourceMotionFingerprint);
    writer.vector<float>(pack.observationMean);
    writer.vector<float>(pack.observationInverseStandardDeviation);
    writer.pod<std::uint64_t>(pack.layers.size());
    for (const HyperPolicyLayer& layer : pack.layers) {
        writer.pod(layer.inputCount);
        writer.pod(layer.outputCount);
        writer.pod(layer.rank);
        const auto activation = static_cast<std::uint32_t>(layer.activation);
        writer.pod(activation);
        writer.vector<float>(layer.weights);
        writer.vector<float>(layer.bias);
        writer.vector<float>(layer.adapterDown);
        writer.vector<float>(layer.adapterUp);
        writer.vector<float>(layer.adapterBiasBasis);
    }
    writer.vector<float>(pack.coefficientLimits);
    writer.vector<float>(pack.actionBias);
    writer.vector<float>(pack.actionScale);
    writer.vector<float>(pack.actionLogStandardDeviation);
    writer.pod(pack.observationClip);
    writer.pod(pack.actionClip);
    writer.vector<float>(pack.knotPhases);
    writer.vector<float>(pack.coefficientKnots);
    writer.vector<float>(pack.coefficientTangents);
    writer.vector<float>(pack.authorityKnots);
    writer.vector<float>(pack.authorityTangents);
    writer.vector<float>(pack.phaseRateKnots);
    writer.vector<float>(pack.phaseRateTangents);
    writer.vector<float>(pack.referencePhases);
    writer.vector<float>(pack.referenceActions);
    writer.pod(pack.signatureCount);
    writer.vector<float>(pack.referenceSignatures);
    writer.vector<float>(pack.signatureWeights);
    writer.pod(pack.contactTrackCount);
    writer.vector<std::uint32_t>(pack.referenceContactMasks);
    writer.vector<std::uint32_t>(pack.contactGroupIndices);
    writer.pod<std::uint64_t>(pack.events.size());
    for (const HyperPolicyEventGuard& event : pack.events) {
        writer.pod(event.phase);
        writer.pod(event.confidence);
        writer.pod(event.requiredContactOnMask);
        writer.pod(event.requiredContactOffMask);
        writer.pod(event.minimumDwellSteps);
        writer.pod(event.kind);
    }
    writer.vector<float>(pack.actionLower);
    writer.vector<float>(pack.actionUpper);
    writer.vector<float>(pack.maximumActionRate);
    writer.pod(pack.maximumPhaseAdvancePerStep);
    writer.pod(pack.phaseAlignmentBlend);
    writer.pod(pack.phaseAlignmentHuberDelta);
    writer.pod(pack.controlTimeStep);
    return std::move(writer).take();
}

bool deserialize(std::span<const std::byte> payload, HyperPolicyPack& pack) {
    Reader reader{payload};
    std::uint64_t layerCount = 0u;
    if (!reader.string(pack.id) || !reader.pod(pack.revision) ||
        !reader.pod(pack.contract.version) ||
        !reader.pod(pack.contract.worldFingerprint) ||
        !reader.pod(pack.contract.taskFingerprint) ||
        !reader.pod(pack.contract.observationFingerprint) ||
        !reader.pod(pack.contract.actionFingerprint) ||
        !reader.pod(pack.sourceMotionFingerprint) ||
        !reader.vector(pack.observationMean) ||
        !reader.vector(pack.observationInverseStandardDeviation) ||
        !reader.pod(layerCount) ||
        layerCount == 0u ||
        layerCount > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    pack.layers.resize(static_cast<std::size_t>(layerCount));
    for (HyperPolicyLayer& layer : pack.layers) {
        std::uint32_t activation = 0u;
        if (!reader.pod(layer.inputCount) ||
            !reader.pod(layer.outputCount) ||
            !reader.pod(layer.rank) || !reader.pod(activation) ||
            activation > static_cast<std::uint32_t>(PolicyActivation::silu) ||
            !reader.vector(layer.weights) || !reader.vector(layer.bias) ||
            !reader.vector(layer.adapterDown) ||
            !reader.vector(layer.adapterUp) ||
            !reader.vector(layer.adapterBiasBasis)) {
            return false;
        }
        layer.activation = static_cast<PolicyActivation>(activation);
    }
    std::uint64_t eventCount = 0u;
    if (!reader.vector(pack.coefficientLimits) ||
        !reader.vector(pack.actionBias) ||
        !reader.vector(pack.actionScale) ||
        !reader.vector(pack.actionLogStandardDeviation) ||
        !reader.pod(pack.observationClip) || !reader.pod(pack.actionClip) ||
        !reader.vector(pack.knotPhases) ||
        !reader.vector(pack.coefficientKnots) ||
        !reader.vector(pack.coefficientTangents) ||
        !reader.vector(pack.authorityKnots) ||
        !reader.vector(pack.authorityTangents) ||
        !reader.vector(pack.phaseRateKnots) ||
        !reader.vector(pack.phaseRateTangents) ||
        !reader.vector(pack.referencePhases) ||
        !reader.vector(pack.referenceActions) ||
        !reader.pod(pack.signatureCount) ||
        !reader.vector(pack.referenceSignatures) ||
        !reader.vector(pack.signatureWeights) ||
        !reader.pod(pack.contactTrackCount) ||
        !reader.vector(pack.referenceContactMasks) ||
        !reader.vector(pack.contactGroupIndices) ||
        !reader.pod(eventCount) ||
        eventCount > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    pack.events.resize(static_cast<std::size_t>(eventCount));
    for (HyperPolicyEventGuard& event : pack.events) {
        if (!reader.pod(event.phase) || !reader.pod(event.confidence) ||
            !reader.pod(event.requiredContactOnMask) ||
            !reader.pod(event.requiredContactOffMask) ||
            !reader.pod(event.minimumDwellSteps) ||
            !reader.pod(event.kind)) {
            return false;
        }
    }
    return reader.vector(pack.actionLower) &&
        reader.vector(pack.actionUpper) &&
        reader.vector(pack.maximumActionRate) &&
        reader.pod(pack.maximumPhaseAdvancePerStep) &&
        reader.pod(pack.phaseAlignmentBlend) &&
        reader.pod(pack.phaseAlignmentHuberDelta) &&
        reader.pod(pack.controlTimeStep) && reader.finished();
}

bool writeAll(int descriptor, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const std::byte*>(data);
    while (bytes != 0u) {
        const std::size_t chunk = std::min(bytes, kMaximumIOChunk);
        const ssize_t written = ::write(descriptor, cursor, chunk);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (written == 0) {
            return false;
        }
        cursor += written;
        bytes -= static_cast<std::size_t>(written);
    }
    return true;
}

LearningPackResult writeFile(
    const std::filesystem::path& path,
    std::span<const std::byte> payload
) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "HyperPolicyPack payload exceeds the 32-bit artifact boundary"
        );
    }
    Header header;
    header.payloadBytes = payload.size();
    header.contentHash = hashPayload(payload);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(::getpid());
    const int descriptor = ::open(
        temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644
    );
    if (descriptor < 0) {
        return fail(LearningPackStatus::ioFailure,
            "failed to create temporary HyperPolicyPack");
    }
    const bool complete = writeAll(descriptor, &header, sizeof(header)) &&
        writeAll(descriptor, payload.data(), payload.size()) &&
        ::fsync(descriptor) == 0;
    const int closeStatus = ::close(descriptor);
    if (!complete || closeStatus != 0) {
        std::filesystem::remove(temporary);
        return fail(LearningPackStatus::ioFailure,
            "failed to durably write HyperPolicyPack");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        return fail(LearningPackStatus::ioFailure,
            "failed to atomically publish HyperPolicyPack: " + error.message());
    }
    const std::filesystem::path parent = path.parent_path().empty()
        ? std::filesystem::path{"."}
        : path.parent_path();
    const int directoryDescriptor = ::open(parent.c_str(), O_RDONLY);
    if (directoryDescriptor < 0) {
        return fail(
            LearningPackStatus::ioFailure,
            "HyperPolicyPack was renamed but its parent directory could not be opened"
        );
    }
    const bool directorySynchronized = ::fsync(directoryDescriptor) == 0;
    const bool directoryClosed = ::close(directoryDescriptor) == 0;
    if (!directorySynchronized || !directoryClosed) {
        return fail(
            LearningPackStatus::ioFailure,
            "HyperPolicyPack was renamed but its directory could not be durably synchronized"
        );
    }
    return {
        .status = LearningPackStatus::success,
        .contentHash = header.contentHash,
        .message = {},
    };
}

} // namespace

LearningPackResult writeHyperPolicyPack(
    const HyperPolicyPack& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation = validate(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        const std::vector<std::byte> payload = serialize(pack);
        return writeFile(path, payload);
    } catch (const std::bad_alloc&) {
        return fail(LearningPackStatus::capacityOverflow,
            "HyperPolicyPack serialization allocation failed");
    } catch (const std::exception& error) {
        return fail(LearningPackStatus::internalFailure, error.what());
    }
}

LearningPackResult readHyperPolicyPack(
    const std::filesystem::path& path,
    HyperPolicyPack& output
) {
    try {
        const std::uintmax_t fileBytes = std::filesystem::file_size(path);
        if (fileBytes < sizeof(Header) ||
            fileBytes - sizeof(Header) >
                std::numeric_limits<std::uint32_t>::max()) {
            return fail(LearningPackStatus::corruptPayload,
                "HyperPolicyPack size is invalid");
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(fileBytes));
        const int descriptor = ::open(path.c_str(), O_RDONLY);
        if (descriptor < 0) {
            return fail(LearningPackStatus::ioFailure,
                "failed to open HyperPolicyPack");
        }
        std::size_t offset = 0u;
        while (offset != bytes.size()) {
            const ssize_t count = ::read(
                descriptor,
                bytes.data() + offset,
                std::min(bytes.size() - offset, kMaximumIOChunk)
            );
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                ::close(descriptor);
                return fail(LearningPackStatus::ioFailure,
                    "failed to read complete HyperPolicyPack");
            }
            offset += static_cast<std::size_t>(count);
        }
        if (::close(descriptor) != 0) {
            return fail(
                LearningPackStatus::ioFailure,
                "failed to close HyperPolicyPack after reading"
            );
        }
        Header header;
        std::memcpy(&header, bytes.data(), sizeof(header));
        const std::span<const std::byte> payload{
            bytes.data() + sizeof(header), bytes.size() - sizeof(header)
        };
        if (header.magic != kMagic || header.kind != kHyperPolicyKind ||
            header.formatVersion != kHyperPolicyPackFormatVersion ||
            header.payloadBytes != payload.size() ||
            header.contentHash != hashPayload(payload)) {
            return fail(LearningPackStatus::corruptPayload,
                "HyperPolicyPack header or content fingerprint is invalid");
        }
        HyperPolicyPack staged;
        if (!deserialize(payload, staged)) {
            return fail(LearningPackStatus::corruptPayload,
                "HyperPolicyPack payload is truncated or malformed");
        }
        const LearningPackResult validation = validate(staged);
        if (!validation.succeeded()) {
            return validation;
        }
        output = std::move(staged);
        return {
            .status = LearningPackStatus::success,
            .contentHash = header.contentHash,
            .message = {},
        };
    } catch (const std::filesystem::filesystem_error& error) {
        return fail(LearningPackStatus::ioFailure, error.what());
    } catch (const std::bad_alloc&) {
        return fail(LearningPackStatus::capacityOverflow,
            "HyperPolicyPack read allocation failed");
    } catch (const std::exception& error) {
        return fail(LearningPackStatus::internalFailure, error.what());
    }
}

} // namespace metalrobo
