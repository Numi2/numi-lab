#include "metalrobo/MeasuredSurfaceRobot.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace metalrobo {
namespace {

void hashBytes(std::uint64_t& hash, const void* bytes, std::size_t count) {
    const auto* data = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t i = 0; i < count; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
}

void hashString(std::uint64_t& hash, std::string_view value) {
    hashBytes(hash, value.data(), value.size());
    const std::uint8_t terminator = 0u;
    hashBytes(hash, &terminator, 1u);
}

template <typename T>
void hashValue(std::uint64_t& hash, const T& value) {
    hashBytes(hash, &value, sizeof(T));
}

bool validSHA256(const std::string& value) {
    return value.size() == 64u && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

} // namespace

std::array<MeasuredSurfaceAction, kMeasuredSurfaceActionCount>
makeMeasuredSurfaceFlightActions() {
    const std::array<const char*, kMeasuredSurfaceActionCount> names {
        "rhythm_frequency", "rhythm_duty_factor", "rhythm_amplitude", "flap_glide_blend",
        "left_phase", "left_stroke", "left_deviation", "left_pitch",
        "left_twist", "left_sweep", "left_span", "left_camber",
        "right_phase", "right_stroke", "right_deviation", "right_pitch",
        "right_twist", "right_sweep", "right_span", "right_camber",
        "tail_elevation", "tail_lateral", "tail_tilt", "tail_spread"
    };
    std::array<MeasuredSurfaceAction, kMeasuredSurfaceActionCount> actions;
    for (std::uint32_t i = 0u; i < kMeasuredSurfaceActionCount; ++i) {
        actions[i].name = names[i];
        actions[i].naturalFrequencyHertz = i < 4u ? 8.0f : 14.0f;
        actions[i].dampingRatio = i < 4u ? 1.0f : 0.82f;
    }
    actions[0].lowerBound = -0.35f;
    actions[0].upperBound = 0.35f;
    actions[1].lowerBound = -0.25f;
    actions[1].upperBound = 0.25f;
    actions[2].lowerBound = -0.45f;
    actions[2].upperBound = 0.45f;
    actions[3].lowerBound = 0.0f;
    actions[3].upperBound = 1.0f;
    return actions;
}

CompiledMeasuredSurfaceRobot compileMeasuredSurfaceRobot(
    const MeasuredSurfaceRobotPack& pack) {
    if (pack.id.empty() || pack.datasetIdentifier.empty() ||
        !validSHA256(pack.manifestSHA256) || !validSHA256(pack.positionsSHA256) ||
        !validSHA256(pack.trianglesSHA256)) {
        throw std::invalid_argument("measured-surface provenance contract is incomplete");
    }
    if (pack.frameCount < 2u || pack.vertexCount == 0u || pack.triangleCount == 0u ||
        !std::isfinite(pack.sampleRateHertz) || pack.sampleRateHertz <= 0.0f ||
        pack.sourcePeriodic || !std::isfinite(pack.bodyMassKilograms) ||
        pack.bodyMassKilograms <= 0.0f ||
        !std::isfinite(pack.airDensityKilogramsPerCubicMeter) ||
        pack.airDensityKilogramsPerCubicMeter <= 0.0f) {
        throw std::invalid_argument("measured-surface physical contract is invalid");
    }
    for (const float inertia : pack.principalInertiaKilogramMetersSquared) {
        if (!std::isfinite(inertia) || inertia <= 0.0f) {
            throw std::invalid_argument("principal inertia must be finite and positive");
        }
    }
    if (pack.components.size() != 4u) {
        throw std::invalid_argument("body, bilateral wings, and tail are required");
    }
    CompiledMeasuredSurfaceRobot result;
    result.pack = pack;
    result.vertexComponents.assign(pack.vertexCount, 0u);
    result.triangleComponents.assign(pack.triangleCount, 0u);
    std::uint32_t nextVertex = 0u;
    std::uint32_t nextTriangle = 0u;
    for (std::uint32_t i = 0u; i < 4u; ++i) {
        const auto& component = pack.components[i];
        if (static_cast<std::uint32_t>(component.component) != i + 1u ||
            component.vertexOffset != nextVertex || component.triangleOffset != nextTriangle ||
            component.vertexCount == 0u || component.triangleCount == 0u ||
            component.vertexCount > pack.vertexCount - nextVertex ||
            component.triangleCount > pack.triangleCount - nextTriangle) {
            throw std::invalid_argument("component ranges must be ordered, positive, and contiguous");
        }
        std::fill_n(result.vertexComponents.begin() + nextVertex,
                    component.vertexCount, static_cast<std::uint8_t>(i + 1u));
        std::fill_n(result.triangleComponents.begin() + nextTriangle,
                    component.triangleCount, static_cast<std::uint8_t>(i + 1u));
        nextVertex += component.vertexCount;
        nextTriangle += component.triangleCount;
    }
    if (nextVertex != pack.vertexCount || nextTriangle != pack.triangleCount) {
        throw std::invalid_argument("component ranges must cover the complete measured surface");
    }
    std::uint64_t hash = 1469598103934665603ull;
    hashString(hash, pack.id);
    hashString(hash, pack.datasetIdentifier);
    hashString(hash, pack.manifestSHA256);
    hashString(hash, pack.positionsSHA256);
    hashString(hash, pack.trianglesSHA256);
    hashValue(hash, pack.frameCount);
    hashValue(hash, pack.vertexCount);
    hashValue(hash, pack.triangleCount);
    hashValue(hash, pack.sampleRateHertz);
    hashValue(hash, pack.sourcePeriodic);
    hashValue(hash, pack.bodyMassKilograms);
    hashValue(hash, pack.principalInertiaKilogramMetersSquared);
    hashValue(hash, pack.airDensityKilogramsPerCubicMeter);
    for (const auto& component : pack.components) hashValue(hash, component);
    for (const auto& action : pack.actions) {
        if (action.name.empty() || !std::isfinite(action.lowerBound) ||
            !std::isfinite(action.upperBound) || action.lowerBound >= action.upperBound ||
            !std::isfinite(action.naturalFrequencyHertz) ||
            action.naturalFrequencyHertz <= 0.0f ||
            !std::isfinite(action.dampingRatio) || action.dampingRatio < 0.0f) {
            throw std::invalid_argument("action contract is invalid");
        }
        hashString(hash, action.name);
        hashValue(hash, action.lowerBound);
        hashValue(hash, action.upperBound);
        hashValue(hash, action.naturalFrequencyHertz);
        hashValue(hash, action.dampingRatio);
    }
    result.fingerprint = hash == 0u ? 1u : hash;
    return result;
}

void stepMeasuredSurfaceActuators(
    const CompiledMeasuredSurfaceRobot& robot,
    std::span<const float, kMeasuredSurfaceActionCount> targets,
    float timeStepSeconds,
    MeasuredSurfaceActuatorState& state) {
    if (!std::isfinite(timeStepSeconds) || timeStepSeconds <= 0.0f ||
        timeStepSeconds > 0.02f) {
        throw std::invalid_argument("actuator time step is outside (0, 0.02] seconds");
    }
    auto candidate = state;
    for (std::uint32_t i = 0u; i < kMeasuredSurfaceActionCount; ++i) {
        const auto& action = robot.pack.actions[i];
        if (!std::isfinite(targets[i]) || !std::isfinite(state.position[i]) ||
            !std::isfinite(state.velocity[i])) {
            throw std::invalid_argument("non-finite measured-surface actuator state");
        }
        const float target = std::clamp(targets[i], action.lowerBound, action.upperBound);
        const float omega = 2.0f * 3.14159265358979323846f * action.naturalFrequencyHertz;
        const float acceleration = omega * omega * (target - state.position[i]) -
            2.0f * action.dampingRatio * omega * state.velocity[i];
        candidate.velocity[i] += timeStepSeconds * acceleration;
        candidate.position[i] += timeStepSeconds * candidate.velocity[i];
        if (!std::isfinite(candidate.position[i]) || !std::isfinite(candidate.velocity[i])) {
            throw std::runtime_error("measured-surface actuator step rejected");
        }
    }
    state = candidate;
}

} // namespace metalrobo
