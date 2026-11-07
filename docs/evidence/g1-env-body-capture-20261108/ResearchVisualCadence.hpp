#pragma once
// Isolated single-step experiment. Not wired into production rendering.
#include <cmath>
#include <cstdint>
#include <limits>

namespace research_cadence {
constexpr std::uint32_t none = std::numeric_limits<std::uint32_t>::max();
// Four exact uint32 words, shared with research_cadence.metal.
struct Record {
    std::uint32_t acceptedSteps = 0;
    std::uint32_t lastSensor = none;
    std::uint32_t captureStep = none;
    std::uint32_t captureTimeBits = 0;
};
static_assert(sizeof(Record) == 16);
struct Candidate {
    Record accepted;
    std::uint32_t sensor;
    bool sample;
};
// Preserve the owner's double calculation from its float configuration.
// A caller supplies capture time in float bits only for a freshly sampled
// frame. The owner must establish that timestamp's semantics separately.
inline bool propose(const Record& old, float stepSeconds, float rateHz,
                    std::uint32_t captureTimeBits, bool reset,
                    Candidate& out) {
    if (reset || old.acceptedSteps >= none - 1u ||
        !(stepSeconds > 0) || !(rateHz > 0) ||
        !std::isfinite(stepSeconds) || !std::isfinite(rateHz)) return false;
    const double time = double(old.acceptedSteps) * double(stepSeconds) * double(rateHz);
    const double sensor = std::floor(time + 1e-6);
    // Reserve UINT32_MAX as the no-frame sentinel; never narrow unsafely.
    if (!std::isfinite(sensor) || sensor < 0 || sensor >= double(none)) return false;
    out.sensor = static_cast<std::uint32_t>(sensor);
    out.sample = out.sensor != old.lastSensor;
    out.accepted = old;
    ++out.accepted.acceptedSteps;
    if (out.sample) {
        out.accepted.lastSensor = out.sensor;
        out.accepted.captureStep = old.acceptedSteps;
        out.accepted.captureTimeBits = captureTimeBits;
    }
    return true;
}
} // namespace research_cadence
