#include "metalrobo/NeuronCultureEmbodiment.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

void hashBytes(std::uint64_t& hash, const void* data, std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0u; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFNVPrime;
    }
}

} // namespace

bool encodeAcceptedSupportStimulus(
    const CompiledNeuronCulture& culture,
    std::span<const MRNumanXHumanSupportConsequenceGPU> consequences,
    float physicsTimestepSeconds,
    float currentPerNewton,
    NeuronCultureStimulus& output
) noexcept {
    if (!culture.valid() || consequences.size() != 10u ||
        !std::isfinite(physicsTimestepSeconds) || physicsTimestepSeconds <= 0.0f ||
        !std::isfinite(currentPerNewton) || currentPerNewton < 0.0f) return false;
    double weightedX = 0.0;
    double weightedY = 0.0;
    double totalNormalImpulse = 0.0;
    for (std::size_t i = 0u; i < consequences.size(); ++i) {
        const auto& consequence = consequences[i];
        const bool valid = consequence.identity.x == i &&
            consequence.identity.w == MR_NUMANX_HUMAN_SUPPORT_CONSEQUENCE_VERSION &&
            std::isfinite(consequence.pointAndSeparation.x) &&
            std::isfinite(consequence.pointAndSeparation.y) &&
            std::isfinite(consequence.pointAndSeparation.z) &&
            std::isfinite(consequence.pointAndSeparation.w) &&
            std::isfinite(consequence.impulseAndNormal.w) &&
            std::isfinite(consequence.tangentVelocityAndImpulse.w) &&
            consequence.impulseAndNormal.w >= 0.0f &&
            consequence.tangentVelocityAndImpulse.w >= 0.0f;
        if (!valid) return false;
        const double weight = consequence.impulseAndNormal.w;
        weightedX += weight * consequence.pointAndSeparation.x;
        weightedY += weight * consequence.pointAndSeparation.y;
        totalNormalImpulse += weight;
    }
    float x = 1.5f;
    float y = 1.5f;
    if (totalNormalImpulse > 0.0) {
        // NHCNT world metres are centered around the root. A bounded 1 m
        // support footprint is mapped into the authored 3 mm culture plane.
        x = 1.5f + 3.0f * std::clamp(
            static_cast<float>(weightedX / totalNormalImpulse), -0.5f, 0.5f);
        y = 1.5f + 3.0f * std::clamp(
            static_cast<float>(weightedY / totalNormalImpulse), -0.5f, 0.5f);
    }
    std::uint32_t nearest = 0u;
    float nearestDistance = std::numeric_limits<float>::max();
    for (std::uint32_t i = 0u; i < culture.electrodes().size(); ++i) {
        const auto& electrode = culture.electrodes()[i];
        const float dx = electrode.x - x;
        const float dy = electrode.y - y;
        const float distance = dx * dx + dy * dy;
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearest = i;
        }
    }
    const double normalForce = totalNormalImpulse / physicsTimestepSeconds;
    const double current = normalForce * currentPerNewton;
    if (!std::isfinite(current)) return false;
    NeuronCultureStimulus candidate;
    candidate.electrode = nearest;
    candidate.current = std::clamp(static_cast<float>(current), 0.0f, 5000.0f);
    std::uint64_t hash = kFNVOffset;
    hashBytes(hash, &culture.header().cultureFingerprint,
              sizeof(culture.header().cultureFingerprint));
    hashBytes(hash, consequences.data(), consequences.size_bytes());
    hashBytes(hash, &physicsTimestepSeconds, sizeof(physicsTimestepSeconds));
    hashBytes(hash, &currentPerNewton, sizeof(currentPerNewton));
    if (hash == 0u) hash = kFNVOffset;
    candidate.sourceFingerprint = hash;
    output = candidate;
    return true;
}

} // namespace metalrobo
