#include "metalrobo/NumiHumanInitialState.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace metalrobo {
namespace {
constexpr std::array<std::byte, 8u> magic{
    std::byte{'N'}, std::byte{'H'}, std::byte{'I'}, std::byte{'N'},
    std::byte{'I'}, std::byte{'T'}, std::byte{'1'}, std::byte{0}};
bool fail(std::string& error, const char* text) { error = text; return false; }
void put(std::vector<std::byte>& bytes, std::uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) bytes.push_back(std::byte((value >> (8u*i)) & 255u));
}
std::uint64_t get(std::span<const std::byte> bytes, std::size_t at, unsigned count) {
    std::uint64_t result = 0;
    for (unsigned i = 0; i < count; ++i) result |= std::uint64_t(std::to_integer<unsigned>(bytes[at+i])) << (8u*i);
    return result;
}
bool valid(const NumiHumanInitialState& s, std::string& error) {
    if (!s.humanSourceFingerprint || !s.worldFingerprint || !s.timestepMicroseconds ||
        s.timestepMicroseconds > 1'000'000u ||
        std::all_of(s.sourceArchiveSHA256.begin(), s.sourceArchiveSHA256.end(), [](auto x){return x == 0;}))
        return fail(error, "initial-state source/world/clock identity missing");
    if (s.q.empty() || s.v.empty() || s.muscles.empty() ||
        s.q.size() > std::numeric_limits<std::uint32_t>::max() ||
        s.v.size() > std::numeric_limits<std::uint32_t>::max() ||
        s.muscles.size() > std::numeric_limits<std::uint32_t>::max())
        return fail(error, "initial-state dimensions invalid");
    for (const auto* values : {&s.q, &s.v}) for (float x : *values)
        if (!std::isfinite(x)) return fail(error, "nonfinite initial generalized state");
    for (const auto& muscle : s.muscles) {
        const auto& x = muscle.excitationAndActivation;
        if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z) || !std::isfinite(x.w) ||
            x.x < 0 || x.x > 1 || x.y < 0 || x.y > 1 || x.z <= 0)
            return fail(error, "invalid initial excitation/activation/fibre state");
    }
    return true;
}
} // namespace

bool encodeNumiHumanInitialState(const NumiHumanInitialState& s,
    std::vector<std::byte>& output, std::string& error) {
    if (!valid(s, error)) return false;
    std::vector<std::byte> bytes(magic.begin(), magic.end());
    for (std::uint32_t word : {1u, 96u, static_cast<std::uint32_t>(s.q.size()),
         static_cast<std::uint32_t>(s.v.size()), static_cast<std::uint32_t>(s.muscles.size()), 4u, 0u, 0u}) put(bytes, word, 4);
    put(bytes, s.humanSourceFingerprint, 8); put(bytes, s.worldFingerprint, 8); put(bytes, s.timestepMicroseconds, 8);
    for (auto byte : s.sourceArchiveSHA256) bytes.push_back(std::byte(byte));
    const auto scalar = [&](float x){ put(bytes, std::bit_cast<std::uint32_t>(x), 4); };
    for (float x : s.q) scalar(x);
    for (float x : s.v) scalar(x);
    for (const auto& muscle : s.muscles) {
        const auto& x = muscle.excitationAndActivation;
        scalar(x.x); scalar(x.y); scalar(x.z); scalar(x.w);
    }
    output = std::move(bytes); error.clear(); return true;
}

bool decodeNumiHumanInitialState(std::span<const std::byte> bytes,
    std::uint32_t nq, std::uint32_t nv, std::uint32_t muscleCount,
    const std::array<std::uint8_t,32u>& sourceSHA, NumiHumanInitialState& output, std::string& error) {
    const std::uint64_t expected = 96ull + 4ull*nq + 4ull*nv + 16ull*muscleCount;
    if (bytes.size() != expected || !nq || !nv || !muscleCount || bytes.size() < 96u)
        return fail(error, "initial-state byte count/dimensions mismatch");
    if (!std::equal(magic.begin(), magic.end(), bytes.begin()) || get(bytes,8,4) != 1 || get(bytes,12,4) != 96 ||
        get(bytes,16,4) != nq || get(bytes,20,4) != nv || get(bytes,24,4) != muscleCount || get(bytes,28,4) != 4 ||
        get(bytes,32,4) != 0 || get(bytes,36,4) != 0)
        return fail(error, "initial-state ABI mismatch");
    NumiHumanInitialState candidate;
    candidate.humanSourceFingerprint = get(bytes,40,8);
    candidate.worldFingerprint = get(bytes,48,8);
    candidate.timestepMicroseconds = get(bytes,56,8);
    for (std::size_t i=0; i<32; ++i) candidate.sourceArchiveSHA256[i] = std::to_integer<std::uint8_t>(bytes[64+i]);
    if (candidate.sourceArchiveSHA256 != sourceSHA) return fail(error, "initial-state source archive mismatch");
    std::size_t offset = 96;
    const auto scalar = [&](){ const auto x = std::bit_cast<float>(static_cast<std::uint32_t>(get(bytes,offset,4))); offset += 4; return x; };
    candidate.q.resize(nq); candidate.v.resize(nv); candidate.muscles.resize(muscleCount);
    for (float& x : candidate.q) x = scalar();
    for (float& x : candidate.v) x = scalar();
    for (auto& muscle : candidate.muscles) {
        auto& x = muscle.excitationAndActivation;
        x.x = scalar(); x.y = scalar(); x.z = scalar(); x.w = scalar();
    }
    if (!valid(candidate,error)) return false;
    output = std::move(candidate); error.clear(); return true;
}
} // namespace metalrobo
