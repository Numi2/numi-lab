#include "metalrobo/MeasuredSurfaceRobot.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <ranges>
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
    // The reflected 143 ms measurement is a 3.50 Hz wingbeat. The robotic
    // rhythm lane spans 2.27--7.00 Hz while preserving that exact measured
    // cadence at zero command.
    actions[0].lowerBound = -0.35f;
    actions[0].upperBound = 1.0f;
    actions[1].lowerBound = -0.25f;
    actions[1].upperBound = 0.25f;
    // Zero is the unmodified measurement. Positive command reaches 2.25x the
    // measured excursion for the robot's bounded recovery envelope.
    actions[2].lowerBound = -0.50f;
    actions[2].upperBound = 1.25f;
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
        pack.vertexCount >
            static_cast<std::uint32_t>(
                std::numeric_limits<std::uint16_t>::max()) + 1u ||
        !std::isfinite(pack.bodyMassKilograms) ||
        pack.bodyMassKilograms <= 0.0f ||
        !std::isfinite(pack.airDensityKilogramsPerCubicMeter) ||
        pack.airDensityKilogramsPerCubicMeter <= 0.0f) {
        throw std::invalid_argument("measured-surface physical contract is invalid");
    }
    const std::size_t positionCount = static_cast<std::size_t>(pack.frameCount) *
        pack.vertexCount * 3u;
    const std::size_t indexCount = static_cast<std::size_t>(pack.triangleCount) * 3u;
    if (pack.frameMajorPositions.size() != positionCount ||
        pack.triangleIndices.size() != indexCount ||
        pack.frameTimesSeconds.size() != pack.frameCount) {
        throw std::invalid_argument("measured-surface payload dimensions are incomplete");
    }
    if (!std::ranges::all_of(pack.frameMajorPositions, [](const float value) {
            return std::isfinite(value);
        }) ||
        !std::ranges::all_of(pack.triangleIndices, [&](const std::uint16_t value) {
            return value < pack.vertexCount;
        }) ||
        !std::ranges::all_of(pack.frameTimesSeconds, [](const float value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("measured-surface payload contains invalid values");
    }
    for (std::size_t frame = 1u; frame < pack.frameTimesSeconds.size(); ++frame) {
        if (!(pack.frameTimesSeconds[frame] > pack.frameTimesSeconds[frame - 1u])) {
            throw std::invalid_argument("measured-surface frame times must increase strictly");
        }
        const float authoredStep = pack.frameTimesSeconds[frame] -
            pack.frameTimesSeconds[frame - 1u];
        const float expectedStep = 1.0f / pack.sampleRateHertz;
        if (std::abs(authoredStep - expectedStep) >
            std::max(1.0e-7f, expectedStep * 1.0e-4f)) {
            throw std::invalid_argument(
                "measured-surface GPU interpolation currently requires uniform frame times");
        }
    }
    if (pack.phaseBoundary == MeasuredSurfacePhaseBoundary::wrap &&
        !pack.sourcePeriodic) {
        throw std::invalid_argument("nonperiodic measured surfaces cannot use wrap phase semantics");
    }
    if (!std::isfinite(pack.normalDragCoefficient) ||
        pack.normalDragCoefficient < 0.0f ||
        !std::isfinite(pack.tangentialDragCoefficient) ||
        pack.tangentialDragCoefficient < 0.0f) {
        throw std::invalid_argument("measured-surface aerodynamic coefficients are invalid");
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
        result.gpuComponents.push_back({
            static_cast<std::uint32_t>(component.component),
            component.vertexOffset,
            component.vertexCount,
            component.triangleOffset,
            component.triangleCount,
            0u, 0u, 0u,
        });
    }
    if (nextVertex != pack.vertexCount || nextTriangle != pack.triangleCount) {
        throw std::invalid_argument("component ranges must cover the complete measured surface");
    }
    for (const MeasuredSurfaceComponentRange& component : pack.components) {
        const std::uint32_t vertexEnd =
            component.vertexOffset + component.vertexCount;
        const std::uint32_t triangleEnd =
            component.triangleOffset + component.triangleCount;
        for (std::uint32_t triangle = component.triangleOffset;
             triangle < triangleEnd; ++triangle) {
            const std::size_t first = static_cast<std::size_t>(triangle) * 3u;
            const std::uint32_t a = pack.triangleIndices[first];
            const std::uint32_t b = pack.triangleIndices[first + 1u];
            const std::uint32_t c = pack.triangleIndices[first + 2u];
            if (a < component.vertexOffset || a >= vertexEnd ||
                b < component.vertexOffset || b >= vertexEnd ||
                c < component.vertexOffset || c >= vertexEnd ||
                a == b || b == c || a == c) {
                throw std::invalid_argument(
                    "component triangle topology crosses a component boundary or is degenerate");
            }
        }
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
    hashValue(hash, pack.phaseBoundary);
    hashValue(hash, pack.bodyMassKilograms);
    hashValue(hash, pack.principalInertiaKilogramMetersSquared);
    hashValue(hash, pack.airDensityKilogramsPerCubicMeter);
    hashValue(hash, pack.normalDragCoefficient);
    hashValue(hash, pack.tangentialDragCoefficient);
    hashBytes(hash, pack.frameMajorPositions.data(),
              pack.frameMajorPositions.size() * sizeof(float));
    hashBytes(hash, pack.triangleIndices.data(),
              pack.triangleIndices.size() * sizeof(std::uint16_t));
    hashBytes(hash, pack.frameTimesSeconds.data(),
              pack.frameTimesSeconds.size() * sizeof(float));
    for (const auto& component : pack.components) hashValue(hash, component);
    for (std::size_t i = 0u; i < pack.actions.size(); ++i) {
        const auto& action = pack.actions[i];
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
        result.gpuActions[i].boundsFrequencyDamping = {
            action.lowerBound,
            action.upperBound,
            action.naturalFrequencyHertz,
            action.dampingRatio,
        };
    }
    mr_float4 minimum{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 0.0f,
    };
    mr_float4 maximum{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 0.0f,
    };
    for (std::size_t offset = 0u;
         offset < pack.frameMajorPositions.size(); offset += 3u) {
        minimum.x = std::min(minimum.x, pack.frameMajorPositions[offset]);
        minimum.y = std::min(minimum.y, pack.frameMajorPositions[offset + 1u]);
        minimum.z = std::min(minimum.z, pack.frameMajorPositions[offset + 2u]);
        maximum.x = std::max(maximum.x, pack.frameMajorPositions[offset]);
        maximum.y = std::max(maximum.y, pack.frameMajorPositions[offset + 1u]);
        maximum.z = std::max(maximum.z, pack.frameMajorPositions[offset + 2u]);
    }
    const mr_float4 center{
        0.5f * (minimum.x + maximum.x),
        0.5f * (minimum.y + maximum.y),
        0.5f * (minimum.z + maximum.z), 0.0f,
    };
    const float dx = maximum.x - minimum.x;
    const float dy = maximum.y - minimum.y;
    const float dz = maximum.z - minimum.z;
    result.gpuModel = {
        MR_MEASURED_SURFACE_ABI_VERSION,
        pack.frameCount,
        pack.vertexCount,
        pack.triangleCount,
        static_cast<std::uint32_t>(pack.components.size()),
        kMeasuredSurfaceActionCount,
        static_cast<std::uint32_t>(pack.phaseBoundary),
        pack.sourcePeriodic ? 1u : 0u,
        {pack.sampleRateHertz,
         pack.airDensityKilogramsPerCubicMeter,
         pack.normalDragCoefficient,
         pack.tangentialDragCoefficient},
        {center.x, center.y, center.z,
         0.5f * std::sqrt(dx * dx + dy * dy + dz * dz)},
        minimum,
        maximum,
    };
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
