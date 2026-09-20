#include "metalrobo/HumanBehaviorStateComponents.hpp"
#include "metalrobo/NumanXExactTransaction.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>

namespace metalrobo {
namespace {

constexpr std::string_view kSensorDomain =
    "numi.numanx.exact-sensor-state.v1";
constexpr std::string_view kPublicationDomain =
    "numi.numanx.exact-publication-state.v1";
constexpr std::uint32_t kCanonicalVersion = 1u;
constexpr std::uint32_t kUnpublishedState = 0u;
constexpr std::uint32_t kPublishedState = 1u;
constexpr std::array<std::uint32_t, 7u> kCanonicalModalities{
    MRNX_CANDIDATE_MODALITY_VISION_V1,
    MRNX_CANDIDATE_MODALITY_AUDITION_V1,
    MRNX_CANDIDATE_MODALITY_TOUCH_V1,
    MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1,
    MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1,
    MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1,
    MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1,
};

class SHA256Writer final {
public:
    SHA256Writer() noexcept : valid_(CC_SHA256_Init(&context_) == 1) {}

    [[nodiscard]] bool append(const void* bytes, std::size_t count) noexcept {
        if (!valid_ || (bytes == nullptr && count != 0u)) return false;
        const auto* cursor = static_cast<const std::uint8_t*>(bytes);
        while (count != 0u) {
            const auto chunk = static_cast<CC_LONG>(std::min<std::size_t>(
                count, std::numeric_limits<CC_LONG>::max()));
            if (CC_SHA256_Update(&context_, cursor, chunk) != 1) {
                valid_ = false;
                return false;
            }
            cursor += chunk;
            count -= chunk;
        }
        return true;
    }

    [[nodiscard]] bool appendU32(const std::uint32_t value) noexcept {
        std::array<std::uint8_t, 4u> bytes{};
        for (std::uint32_t index = 0u; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(
                (value >> (index * 8u)) & 0xffu);
        }
        return append(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool appendU64(const std::uint64_t value) noexcept {
        std::array<std::uint8_t, 8u> bytes{};
        for (std::uint32_t index = 0u; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(
                (value >> (index * 8u)) & 0xffu);
        }
        return append(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool finish(HumanBehaviorStateDigest& output) noexcept {
        if (!valid_ || CC_SHA256_Final(output.data(), &context_) != 1) {
            output.fill(0u);
            valid_ = false;
            return false;
        }
        valid_ = false;
        return true;
    }

private:
    CC_SHA256_CTX context_{};
    bool valid_ = false;
};

[[nodiscard]] bool begin(
    SHA256Writer& writer,
    const std::string_view domain,
    const std::uint32_t state
) noexcept {
    return writer.append(domain.data(), domain.size()) &&
        writer.appendU32(kCanonicalVersion) && writer.appendU32(state);
}

[[nodiscard]] bool product(
    const std::uint64_t first,
    const std::uint64_t second,
    std::uint64_t& output
) noexcept {
    if (first != 0u &&
        second > std::numeric_limits<std::uint64_t>::max() / first) {
        return false;
    }
    output = first * second;
    return true;
}

[[nodiscard]] bool appendRangeMetadata(
    SHA256Writer& writer,
    const mrnx_metal_range_v1& range
) noexcept {
    return writer.appendU32(range.element_type) &&
        writer.appendU32(range.element_byte_count) &&
        writer.appendU64(range.byte_count);
}

[[nodiscard]] bool appendPublication(
    SHA256Writer& writer,
    const mrnx_publication_v2& publication
) noexcept {
    return writer.appendU32(publication.abi_version) &&
        writer.appendU32(publication.struct_size) &&
        writer.appendU32(publication.clock_domain) &&
        writer.appendU32(publication.clock_quantum_nanoseconds) &&
        writer.appendU64(publication.transaction_fingerprint) &&
        writer.appendU64(publication.accepted_physics_token_fingerprint) &&
        writer.appendU64(publication.candidate_publication_fingerprint) &&
        writer.appendU64(publication.joint_commit_fingerprint) &&
        writer.appendU64(publication.brain_generation) &&
        writer.appendU64(publication.committed_timestamp_nanoseconds) &&
        writer.appendU64(publication.publication_fingerprint);
}

} // namespace

HumanBehaviorStateDigest
humanBehaviorUnpublishedSensorStateSHA256() noexcept {
    HumanBehaviorStateDigest output{};
    SHA256Writer writer;
    if (!begin(writer, kSensorDomain, kUnpublishedState) ||
        !writer.appendU32(0u) ||
        !writer.appendU32(0u) || !writer.appendU32(0u) ||
        !writer.appendU64(0u) || !writer.appendU64(0u) ||
        !writer.appendU64(0u) || !writer.appendU64(0u) ||
        !writer.appendU32(0u) || !writer.finish(output)) {
        output.fill(0u);
    }
    return output;
}

bool humanBehaviorExactSensorStateSHA256(
    const std::uint32_t sourcePacketABI,
    const mrnx_candidate_timing_v2& timing,
    const std::span<const HumanBehaviorSensorStateChannelV1> channels,
    HumanBehaviorStateDigest& output,
    std::string& error
) noexcept {
    output.fill(0u);
    try {
        if (sourcePacketABI != MRNX_EXACT_SENSOR_PACKET_ABI_V2 ||
            !metalNumanXExactCandidateTimingV2Valid(timing) ||
            channels.size() != kCanonicalModalities.size()) {
            error = "exact sensor state must contain seven canonical channels";
            return false;
        }
        SHA256Writer writer;
        if (!begin(writer, kSensorDomain, kPublishedState) ||
            !writer.appendU32(sourcePacketABI) ||
            !writer.appendU32(timing.clock_domain) ||
            !writer.appendU32(timing.clock_quantum_nanoseconds) ||
            !writer.appendU64(timing.capture_timestamp_nanoseconds) ||
            !writer.appendU64(timing.delivery_timestamp_nanoseconds) ||
            !writer.appendU64(timing.latency_nanoseconds) ||
            !writer.appendU64(timing.sample_interval_nanoseconds) ||
            !writer.appendU32(static_cast<std::uint32_t>(channels.size()))) {
            error = "sensor state SHA-256 initialization failed";
            return false;
        }
        for (std::size_t index = 0u; index < channels.size(); ++index) {
            const auto& payload = channels[index];
            if (payload.descriptor == nullptr) {
                error = "sensor state channel descriptor is null";
                return false;
            }
            const auto& channel = *payload.descriptor;
            std::uint64_t valueElements = 0u;
            std::uint64_t valueBytes = 0u;
            std::uint64_t validityBytes = 0u;
            if (!metalNumanXExactCandidateChannelV2Valid(channel, timing) ||
                channel.modality != kCanonicalModalities[index] ||
                channel.flags != MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1 ||
                channel.receptor_timestamp_nanoseconds == 0u ||
                channel.clock_domain !=
                    MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS ||
                channel.clock_quantum_nanoseconds !=
                    MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS ||
                channel.receptor_count == 0u ||
                channel.feature_dimension == 0u ||
                channel.values.abi_version != MRNX_BRIDGE_ABI_V1 ||
                channel.values.struct_size != sizeof(channel.values) ||
                channel.values.element_type != MRNX_ELEMENT_FLOAT32_V1 ||
                channel.values.element_byte_count != sizeof(float) ||
                channel.validity.abi_version != MRNX_BRIDGE_ABI_V1 ||
                channel.validity.struct_size != sizeof(channel.validity) ||
                channel.validity.element_type != MRNX_ELEMENT_UINT32_V1 ||
                channel.validity.element_byte_count != sizeof(std::uint32_t) ||
                !product(channel.receptor_count,
                    channel.feature_dimension, valueElements) ||
                !product(valueElements, sizeof(float), valueBytes) ||
                !product(channel.receptor_count, sizeof(std::uint32_t),
                    validityBytes) ||
                channel.values.byte_count != valueBytes ||
                channel.validity.byte_count != validityBytes ||
                payload.values.size() != valueBytes ||
                payload.validity.size() != validityBytes) {
                error = "sensor state channel metadata or byte extent is invalid";
                return false;
            }
            if (!writer.appendU32(channel.modality) ||
                !writer.appendU32(channel.flags) ||
                !writer.appendU64(channel.receptor_timestamp_nanoseconds) ||
                !writer.appendU32(channel.clock_domain) ||
                !writer.appendU32(channel.clock_quantum_nanoseconds) ||
                !writer.appendU32(channel.receptor_count) ||
                !writer.appendU32(channel.feature_dimension) ||
                !appendRangeMetadata(writer, channel.values) ||
                !writer.append(payload.values.data(), payload.values.size()) ||
                !appendRangeMetadata(writer, channel.validity) ||
                !writer.append(
                    payload.validity.data(), payload.validity.size())) {
                error = "sensor state SHA-256 update failed";
                return false;
            }
        }
        if (!writer.finish(output)) {
            error = "sensor state SHA-256 finalization failed";
            return false;
        }
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        output.fill(0u);
        return false;
    } catch (...) {
        error = "sensor state SHA-256 failed";
        output.fill(0u);
        return false;
    }
}

HumanBehaviorStateDigest
humanBehaviorUnpublishedPublicationStateSHA256() noexcept {
    HumanBehaviorStateDigest output{};
    SHA256Writer writer;
    const mrnx_publication_v2 unpublished{};
    if (!begin(writer, kPublicationDomain, kUnpublishedState) ||
        !writer.appendU64(0u) ||
        !appendPublication(writer, unpublished) ||
        !writer.finish(output)) {
        output.fill(0u);
    }
    return output;
}

bool humanBehaviorExactPublicationStateSHA256(
    const std::uint64_t publicationEpoch,
    const mrnx_publication_v2& publication,
    HumanBehaviorStateDigest& output,
    std::string& error
) noexcept {
    output.fill(0u);
    try {
        if (publicationEpoch == 0u ||
            publication.abi_version != MRNX_PUBLICATION_ABI_V2 ||
            publication.struct_size != sizeof(publication) ||
            publication.clock_domain !=
                MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS ||
            publication.clock_quantum_nanoseconds !=
                MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS ||
            publication.transaction_fingerprint == 0u ||
            publication.accepted_physics_token_fingerprint == 0u ||
            publication.candidate_publication_fingerprint == 0u ||
            publication.joint_commit_fingerprint == 0u ||
            publication.brain_generation == 0u ||
            publication.committed_timestamp_nanoseconds == 0u ||
            publication.publication_fingerprint == 0u ||
            publication.publication_fingerprint !=
                metalNumanXExactPublicationV2Fingerprint(publication)) {
            error = "exact publication state is invalid";
            return false;
        }
        SHA256Writer writer;
        if (!begin(writer, kPublicationDomain, kPublishedState) ||
            !writer.appendU64(publicationEpoch) ||
            !appendPublication(writer, publication) ||
            !writer.finish(output)) {
            error = "publication state SHA-256 failed";
            output.fill(0u);
            return false;
        }
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        output.fill(0u);
        return false;
    } catch (...) {
        error = "publication state SHA-256 failed";
        output.fill(0u);
        return false;
    }
}

bool humanBehaviorStateDigestPresent(
    const HumanBehaviorStateDigest& digest
) noexcept {
    return std::any_of(
        digest.begin(), digest.end(), [](const auto byte) { return byte != 0u; });
}

} // namespace metalrobo
