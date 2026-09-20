#include "metalrobo/NumiHumanInitialState.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace metalrobo {
namespace {
constexpr std::array<std::byte, 8u> magicV1{
    std::byte{'N'}, std::byte{'H'}, std::byte{'I'}, std::byte{'N'},
    std::byte{'I'}, std::byte{'T'}, std::byte{'1'}, std::byte{0}};
constexpr std::array<std::byte, 8u> magicV2{
    std::byte{'N'}, std::byte{'H'}, std::byte{'I'}, std::byte{'N'},
    std::byte{'I'}, std::byte{'T'}, std::byte{'2'}, std::byte{0}};
constexpr std::array<std::byte, 8u> magicV3{
    std::byte{'N'}, std::byte{'H'}, std::byte{'I'}, std::byte{'N'},
    std::byte{'I'}, std::byte{'T'}, std::byte{'3'}, std::byte{0}};
constexpr std::uint32_t v1HeaderBytes = 96u, v2HeaderBytes = 160u,
    v3HeaderBytes = 224u;
constexpr std::uint32_t supportHistoryRecordBytes = 16u;
constexpr std::uint32_t supportHistoryEncoding = 1u;
constexpr std::uint64_t maxTimestepNanoseconds = 1'000'000'000u;
bool fail(std::string& error, const char* text) { error = text; return false; }
void put(std::vector<std::byte>& bytes, std::uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) bytes.push_back(std::byte((value >> (8u*i)) & 255u));
}
std::uint64_t get(std::span<const std::byte> bytes, std::size_t at, unsigned count) {
    std::uint64_t result = 0;
    for (unsigned i = 0; i < count; ++i) result |= std::uint64_t(std::to_integer<unsigned>(bytes[at+i])) << (8u*i);
    return result;
}
bool version2(const NumiHumanInitialState& state) {
    return state.rootTranslation.has_value() || state.timestepNanoseconds != 0u;
}
bool validSupportIdentity(const NumiHumanSupportPayloadIdentity& identity) {
    if (std::all_of(identity.sha256.begin(), identity.sha256.end(),
            [](auto x){ return x == 0u; }) ||
        identity.byteCount < 84u ||
        (identity.payloadABI != 1u && identity.payloadABI != 2u) ||
        identity.sourceRecordCount == 0u || identity.expandedRowCount == 0u)
        return false;
    if (identity.payloadABI == 1u)
        return identity.sourceRecordCount == identity.expandedRowCount &&
            identity.byteCount == 84ull + 48ull * identity.sourceRecordCount;
    return identity.expandedRowCount >= identity.sourceRecordCount &&
        identity.expandedRowCount <= 2ull * identity.sourceRecordCount &&
        identity.byteCount == 84ull + 96ull * identity.sourceRecordCount;
}
bool valid(const NumiHumanInitialState& s, std::string& error) {
    if (!s.humanSourceFingerprint || !s.worldFingerprint ||
        !numiHumanInitialStateTimestepNanoseconds(s) ||
        std::all_of(s.sourceArchiveSHA256.begin(), s.sourceArchiveSHA256.end(), [](auto x){return x == 0;}))
        return fail(error, "initial-state source/world/clock identity missing or inconsistent");
    if (s.q.empty() || s.v.empty() || s.muscles.empty() ||
        ((version2(s) || s.preparedSupportHistory) && s.q.size() < 3u) ||
        s.q.size() > std::numeric_limits<std::uint32_t>::max() ||
        s.v.size() > std::numeric_limits<std::uint32_t>::max() ||
        s.muscles.size() > std::numeric_limits<std::uint32_t>::max())
        return fail(error, "initial-state dimensions invalid");
    if (s.preparedSupportHistory) {
        const auto& prepared = *s.preparedSupportHistory;
        if (!validSupportIdentity(prepared.support) ||
            prepared.rows.size() != prepared.support.expandedRowCount)
            return fail(error, "prepared support identity/count invalid");
        for (const auto& row : prepared.rows)
            if (!std::isfinite(row.tangentImpulseWorldX) ||
                !std::isfinite(row.tangentImpulseWorldY) ||
                !std::isfinite(row.tangentImpulseWorldZ) ||
                !std::isfinite(row.normalImpulse) || row.normalImpulse < 0.0f)
                return fail(error, "invalid prepared support impulse");
    }
    for (const auto* values : {&s.q, &s.v}) for (float x : *values)
        if (!std::isfinite(x)) return fail(error, "nonfinite initial generalized state");
    if (version2(s)) {
        const auto root = s.rootTranslation.value_or(
            mrCompensatedTranslationFromProjection({s.q[0], s.q[1], s.q[2], 0.0f}));
        if (!mrCompensatedTranslationValid(root))
            return fail(error, "invalid or noncanonical initial root translation");
        const auto projection = mrCompensatedTranslationProjection(root);
        const std::array<float, 3u> values{projection.x, projection.y, projection.z};
        for (std::size_t axis = 0u; axis < values.size(); ++axis)
            if (!std::isfinite(values[axis]) ||
                std::bit_cast<std::uint32_t>(values[axis]) != std::bit_cast<std::uint32_t>(s.q[axis]))
                return fail(error, "initial root translation projection differs from q");
    }
    for (const auto& muscle : s.muscles) {
        const auto& x = muscle.excitationAndActivation;
        if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z) || !std::isfinite(x.w) ||
            x.x < 0 || x.x > 1 || x.y < 0 || x.y > 1 || x.z <= 0)
            return fail(error, "invalid initial excitation/activation/fibre state");
    }
    return true;
}
} // namespace

bool makeNumiHumanInitialRootTranslation(
    const std::array<double, 3u>& position,
    MRCompensatedRootTranslationGPU& output, std::string& error) {
    std::array<float, 3u> reference{}, displacement{}, correction{};
    for (std::size_t axis = 0u; axis < position.size(); ++axis) {
        const double value = position[axis];
        if (!std::isfinite(value) ||
            std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
            return fail(error, "initial root placement is not finite or representable");
        reference[axis] = static_cast<float>(value);
        const double residual = value - static_cast<double>(reference[axis]);
        const float high = static_cast<float>(residual);
        const float low = static_cast<float>(residual - static_cast<double>(high));
        const auto normalized = mrCompensatedSum(high, low);
        displacement[axis] = normalized.high;
        correction[axis] = normalized.low;
        const double reconstructed = (static_cast<double>(reference[axis]) +
            static_cast<double>(displacement[axis])) + static_cast<double>(correction[axis]);
        if (reconstructed != value)
            return fail(error, "initial root placement loses precision in the FP32 expansion");
    }
    const MRCompensatedRootTranslationGPU candidate{
        {reference[0], reference[1], reference[2], 0.0f},
        {displacement[0], displacement[1], displacement[2], 0.0f},
        {correction[0], correction[1], correction[2], 0.0f}};
    if (!mrCompensatedTranslationValid(candidate))
        return fail(error, "initial root placement produced a noncanonical expansion");
    const auto projected = mrCompensatedTranslationProjection(candidate);
    const std::array<float, 3u> projection{projected.x, projected.y, projected.z};
    for (std::size_t axis = 0u; axis < projection.size(); ++axis)
        if (std::bit_cast<std::uint32_t>(projection[axis]) !=
            std::bit_cast<std::uint32_t>(reference[axis]))
            return fail(error, "initial root placement changed the canonical q projection");
    output = candidate;
    error.clear();
    return true;
}

std::uint64_t numiHumanInitialStateTimestepNanoseconds(const NumiHumanInitialState& s) noexcept {
    if (s.timestepMicroseconds > maxTimestepNanoseconds / 1000u ||
        s.timestepNanoseconds > maxTimestepNanoseconds ||
        (s.timestepMicroseconds && s.timestepNanoseconds &&
         s.timestepMicroseconds * 1000u != s.timestepNanoseconds)) return 0u;
    return s.timestepNanoseconds ? s.timestepNanoseconds : s.timestepMicroseconds * 1000u;
}

bool encodeNumiHumanInitialState(const NumiHumanInitialState& s,
    std::vector<std::byte>& output, std::string& error) {
    if (!valid(s, error)) return false;
    const bool v3 = s.preparedSupportHistory.has_value();
    const bool v2 = v3 || version2(s);
    const auto& magic = v3 ? magicV3 : (v2 ? magicV2 : magicV1);
    const std::uint32_t header = v3 ? v3HeaderBytes : (v2 ? v2HeaderBytes : v1HeaderBytes);
    std::vector<std::byte> bytes(magic.begin(), magic.end());
    for (std::uint32_t word : {v3 ? 3u : (v2 ? 2u : 1u), header,
         static_cast<std::uint32_t>(s.q.size()), static_cast<std::uint32_t>(s.v.size()),
         static_cast<std::uint32_t>(s.muscles.size()), 4u,
         (s.rootTranslation ? 1u : 0u) | (v3 ? 2u : 0u), 0u}) put(bytes, word, 4);
    put(bytes, s.humanSourceFingerprint, 8); put(bytes, s.worldFingerprint, 8); put(bytes, s.timestepMicroseconds, 8);
    for (auto byte : s.sourceArchiveSHA256) bytes.push_back(std::byte(byte));
    const auto scalar = [&](float x){ put(bytes, std::bit_cast<std::uint32_t>(x), 4); };
    if (v2) {
        put(bytes, numiHumanInitialStateTimestepNanoseconds(s), 8);
        const MRCompensatedRootTranslationGPU root = s.rootTranslation.value_or(MRCompensatedRootTranslationGPU{});
        for (const auto& vector : {root.reference, root.displacement, root.correction}) {
            scalar(vector.x); scalar(vector.y); scalar(vector.z); scalar(vector.w);
        }
        put(bytes, 0u, 8);
    }
    if (v3) {
        const auto& support = s.preparedSupportHistory->support;
        for (auto byte : support.sha256) bytes.push_back(std::byte(byte));
        put(bytes, support.byteCount, 8);
        put(bytes, support.payloadABI, 4);
        put(bytes, support.sourceRecordCount, 4);
        put(bytes, support.expandedRowCount, 4);
        put(bytes, supportHistoryRecordBytes, 4);
        put(bytes, supportHistoryEncoding, 4);
        put(bytes, 0u, 4);
    }
    for (float x : s.q) scalar(x);
    for (float x : s.v) scalar(x);
    for (const auto& muscle : s.muscles) {
        const auto& x = muscle.excitationAndActivation;
        scalar(x.x); scalar(x.y); scalar(x.z); scalar(x.w);
    }
    if (v3) for (const auto& row : s.preparedSupportHistory->rows) {
        scalar(row.tangentImpulseWorldX); scalar(row.tangentImpulseWorldY);
        scalar(row.tangentImpulseWorldZ); scalar(row.normalImpulse);
    }
    output = std::move(bytes); error.clear(); return true;
}

bool decodeImpl(std::span<const std::byte> bytes,
    std::uint32_t nq, std::uint32_t nv, std::uint32_t muscleCount,
    const std::array<std::uint8_t,32u>& sourceSHA,
    const NumiHumanSupportPayloadIdentity* expectedSupport,
    NumiHumanInitialState& output, std::string& error) {
    if (!nq || !nv || !muscleCount || bytes.size() < v1HeaderBytes)
        return fail(error, "initial-state byte count/dimensions mismatch");
    const bool v1 = std::equal(magicV1.begin(), magicV1.end(), bytes.begin());
    const bool v2 = std::equal(magicV2.begin(), magicV2.end(), bytes.begin());
    const bool v3 = std::equal(magicV3.begin(), magicV3.end(), bytes.begin());
    if ((v2 && bytes.size() < v2HeaderBytes) ||
        (v3 && bytes.size() < v3HeaderBytes))
        return fail(error, "truncated initial-state extension");
    if (v3 && expectedSupport == nullptr)
        return fail(error, "NHINIT3 requires an expected support identity");
    const std::uint32_t header = v3 ? v3HeaderBytes : (v2 ? v2HeaderBytes : v1HeaderBytes);
    const std::uint32_t version = v3 ? 3u : (v2 ? 2u : 1u);
    const std::uint32_t flags = static_cast<std::uint32_t>(get(bytes,32,4));
    if ((!v1 && !v2 && !v3) || get(bytes,8,4) != version || get(bytes,12,4) != header ||
        get(bytes,16,4) != nq || get(bytes,20,4) != nv || get(bytes,24,4) != muscleCount || get(bytes,28,4) != 4 ||
        (v1 ? flags != 0u : (v2 ? flags > 1u : (flags & 2u) == 0u || (flags & ~3u) != 0u)) ||
        get(bytes,36,4) != 0)
        return fail(error, "initial-state ABI mismatch");
    NumiHumanSupportPayloadIdentity embeddedSupport;
    std::uint64_t historyBytes = 0u;
    if (v3) {
        for (std::size_t i=0; i<32u; ++i)
            embeddedSupport.sha256[i] = std::to_integer<std::uint8_t>(bytes[160u+i]);
        embeddedSupport.byteCount = get(bytes,192,8);
        embeddedSupport.payloadABI = static_cast<std::uint32_t>(get(bytes,200,4));
        embeddedSupport.sourceRecordCount = static_cast<std::uint32_t>(get(bytes,204,4));
        embeddedSupport.expandedRowCount = static_cast<std::uint32_t>(get(bytes,208,4));
        if (!validSupportIdentity(embeddedSupport) ||
            get(bytes,212,4) != supportHistoryRecordBytes ||
            get(bytes,216,4) != supportHistoryEncoding || get(bytes,220,4) != 0u ||
            embeddedSupport != *expectedSupport)
            return fail(error, "initial-state support payload identity mismatch");
        historyBytes = 16ull * embeddedSupport.expandedRowCount;
    }
    const std::uint64_t expected = std::uint64_t(header) + 4ull*nq + 4ull*nv +
        16ull*muscleCount + historyBytes;
    if (expected > std::numeric_limits<std::size_t>::max() || bytes.size() != expected ||
        ((v2 || v3) && nq < 3u))
        return fail(error, "initial-state byte count/dimensions mismatch");
    NumiHumanInitialState candidate;
    candidate.humanSourceFingerprint = get(bytes,40,8);
    candidate.worldFingerprint = get(bytes,48,8);
    candidate.timestepMicroseconds = get(bytes,56,8);
    for (std::size_t i=0; i<32; ++i) candidate.sourceArchiveSHA256[i] = std::to_integer<std::uint8_t>(bytes[64+i]);
    if (candidate.sourceArchiveSHA256 != sourceSHA) return fail(error, "initial-state source archive mismatch");
    std::size_t offset = (v2 || v3) ? 104u : v1HeaderBytes;
    const auto scalar = [&](){ const auto x = std::bit_cast<float>(static_cast<std::uint32_t>(get(bytes,offset,4))); offset += 4; return x; };
    if (v2 || v3) {
        candidate.timestepNanoseconds = get(bytes,96,8);
        if (!candidate.timestepNanoseconds || get(bytes,152,8) != 0u)
            return fail(error, "invalid initial-state nanosecond clock or reserved bytes");
        if ((flags & 1u) != 0u) {
            MRCompensatedRootTranslationGPU root{};
            for (auto* vector : {&root.reference, &root.displacement, &root.correction}) {
                vector->x = scalar(); vector->y = scalar(); vector->z = scalar(); vector->w = scalar();
            }
            candidate.rootTranslation = root;
        } else if (std::any_of(bytes.begin()+104, bytes.begin()+152, [](auto x){return x != std::byte{0};})) {
            return fail(error, "absent initial root translation has nonzero payload");
        }
        offset = header;
    }
    candidate.q.resize(nq); candidate.v.resize(nv); candidate.muscles.resize(muscleCount);
    for (float& x : candidate.q) x = scalar();
    for (float& x : candidate.v) x = scalar();
    for (auto& muscle : candidate.muscles) {
        auto& x = muscle.excitationAndActivation;
        x.x = scalar(); x.y = scalar(); x.z = scalar(); x.w = scalar();
    }
    if (v3) {
        NumiHumanPreparedSupportHistory prepared;
        prepared.support = embeddedSupport;
        prepared.rows.resize(embeddedSupport.expandedRowCount);
        for (auto& row : prepared.rows) {
            row.tangentImpulseWorldX = scalar();
            row.tangentImpulseWorldY = scalar();
            row.tangentImpulseWorldZ = scalar();
            row.normalImpulse = scalar();
        }
        candidate.preparedSupportHistory = std::move(prepared);
    }
    if (!valid(candidate,error)) return false;
    output = std::move(candidate); error.clear(); return true;
}

bool decodeNumiHumanInitialState(std::span<const std::byte> bytes,
    std::uint32_t nq, std::uint32_t nv, std::uint32_t muscleCount,
    const std::array<std::uint8_t,32u>& sourceSHA, NumiHumanInitialState& output,
    std::string& error) {
    return decodeImpl(bytes, nq, nv, muscleCount, sourceSHA, nullptr, output, error);
}

bool decodeNumiHumanInitialState(std::span<const std::byte> bytes,
    std::uint32_t nq, std::uint32_t nv, std::uint32_t muscleCount,
    const std::array<std::uint8_t,32u>& sourceSHA,
    const NumiHumanSupportPayloadIdentity& expectedSupport,
    NumiHumanInitialState& output, std::string& error) {
    return decodeImpl(bytes, nq, nv, muscleCount, sourceSHA, &expectedSupport,
        output, error);
}
} // namespace metalrobo
