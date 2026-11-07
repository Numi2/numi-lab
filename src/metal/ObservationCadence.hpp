#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

namespace metalrobo::detail {
// Host encoding cadence only. Physical acceptance and GPU history remain
// separate transactions that this helper does not certify.
struct ObservationCadence {
    struct Candidate {
        std::uint64_t observation;
        std::uint64_t sensor;
        bool sample;
    };
    std::uint64_t observation = 0u;
    std::uint64_t lastSensor = std::numeric_limits<std::uint64_t>::max();

    Candidate candidate(float timestep, float rate) const {
        const double time = double(observation) * double(timestep) * double(rate);
        const auto sensor = static_cast<std::uint64_t>(std::floor(time + 1.0e-6));
        return {observation, sensor, sensor != lastSensor};
    }
    void commit(const Candidate& value) {
        ++observation;
        if (value.sample) lastSensor = value.sensor;
    }
    void reset() {
        observation = 0u;
        lastSensor = std::numeric_limits<std::uint64_t>::max();
    }
};
}
