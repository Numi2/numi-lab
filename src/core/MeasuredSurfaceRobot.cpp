#include "metalrobo/MeasuredSurfaceRobot.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <fstream>
#include <ranges>
#include <sstream>
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

template <typename T>
std::vector<T> readBinary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid binary size for " + path.string());
    }
    std::vector<T> values(static_cast<std::size_t>(bytes) / sizeof(T));
    stream.seekg(0);
    if (!values.empty() &&
        !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return values;
}

std::string sha256File(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    CC_SHA256_CTX context{};
    CC_SHA256_Init(&context);
    std::array<char, 1024 * 1024> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            CC_SHA256_Update(&context, buffer.data(),
                static_cast<CC_LONG>(count));
        }
    }
    if (!stream.eof()) {
        throw std::runtime_error("cannot hash " + path.string());
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) {
        encoded << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return encoded.str();
}

} // namespace

MeasuredSurfaceRobotPack loadDeetjenMeasuredDoveRobotPack(
    const std::filesystem::path& manifestPath
) {
    const std::filesystem::path directory = manifestPath.parent_path();
    MeasuredSurfaceRobotPack pack;
    pack.id = "deetjen-f03-surface-robot-v1";
    pack.datasetIdentifier =
        "deetjen-ob-2018-12-11-f03-complete-surface-v1";
    pack.manifestSHA256 =
        "ad42148aa9ee72d994d668ba16f8b6572cb8b192b77539fe66d97586ed9e1a13";
    pack.positionsSHA256 =
        "690b6dd2a24d593a512d799b7fe5f3f756ca7ae3ce1cd1cdc4bb12b2531567a6";
    pack.trianglesSHA256 =
        "9d832ff22ecedc15e47c454378146a1006ae7f6974512ce222994e2f12f43d61";
    const std::filesystem::path positionsPath = directory / "positions.f32le";
    const std::filesystem::path trianglesPath = directory / "triangles.u16le";
    if (sha256File(manifestPath) != pack.manifestSHA256 ||
        sha256File(positionsPath) != pack.positionsSHA256 ||
        sha256File(trianglesPath) != pack.trianglesSHA256) {
        throw std::runtime_error(
            "measured Deetjen dove artifact failed SHA-256 qualification");
    }
    pack.frameCount = 144u;
    pack.vertexCount = 2157u;
    pack.triangleCount = 3968u;
    pack.sampleRateHertz = 1000.0f;
    pack.sourcePeriodic = false;
    pack.phaseBoundary = MeasuredSurfacePhaseBoundary::reflect;
    pack.actions = makeMeasuredSurfaceFlightActions();
    pack.components = {
        {MeasuredSurfaceComponent::body, 0u, 1443u, 0u, 2736u},
        {MeasuredSurfaceComponent::leftWing, 1443u, 297u, 2736u, 512u},
        {MeasuredSurfaceComponent::rightWing, 1740u, 297u, 3248u, 512u},
        {MeasuredSurfaceComponent::tail, 2037u, 120u, 3760u, 208u},
    };
    pack.frameMajorPositions = readBinary<float>(positionsPath);
    pack.triangleIndices = readBinary<std::uint16_t>(trianglesPath);
    pack.frameTimesSeconds.resize(pack.frameCount);
    for (std::uint32_t frame = 0u; frame < pack.frameCount; ++frame) {
        pack.frameTimesSeconds[frame] =
            static_cast<float>(frame) / pack.sampleRateHertz;
    }
    return pack;
}

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

std::array<float, kMeasuredSurfaceActionCount>
measuredSurfaceRecoveryTrimActions() {
    return {
        1.0f, 0.149022f, 0.971734f, 0.0f,
        -0.244330f, -0.958796f, -0.164683f, 0.737151f,
        0.484145f, 0.0105697f, -0.111965f, -0.164570f,
        -0.756487f, -0.860807f, -0.752742f, 0.469568f,
        0.785600f, -0.247472f, 0.123905f, -0.303837f,
        0.194382f, 0.0452973f, -0.366264f, -0.506249f,
    };
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
    if (!std::ranges::all_of(pack.aerodynamicCorrections,
            [](const float value) { return std::isfinite(value); }) ||
        pack.aerodynamicCorrections[0u] <= 0.0f ||
        pack.aerodynamicCorrections[0u] >= 1.0f ||
        pack.aerodynamicCorrections[1u] < 0.0f ||
        pack.aerodynamicCorrections[1u] > 1.0f ||
        pack.aerodynamicCorrections[2u] < 0.0f ||
        pack.aerodynamicCorrections[2u] > 1.0f ||
        pack.aerodynamicCorrections[3u] <= 0.0f) {
        throw std::invalid_argument(
            "measured-surface aerodynamic corrections are invalid");
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
    hashValue(hash, pack.aerodynamicCorrections);
    hashValue(hash, pack.normalizedActionBias);
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
        if (!std::isfinite(pack.normalizedActionBias[i]) ||
            pack.normalizedActionBias[i] < -1.0f ||
            pack.normalizedActionBias[i] > 1.0f) {
            throw std::invalid_argument("normalized action bias is outside [-1, 1]");
        }
        result.gpuActions[i].boundsFrequencyDamping = {
            action.lowerBound,
            action.upperBound,
            action.naturalFrequencyHertz,
            action.dampingRatio,
        };
        result.gpuActions[i].normalizedBiasReserved = {
            pack.normalizedActionBias[i], 0.0f, 0.0f, 0.0f};
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
        {pack.aerodynamicCorrections[0u],
         pack.aerodynamicCorrections[1u],
         pack.aerodynamicCorrections[2u],
         pack.aerodynamicCorrections[3u]},
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
