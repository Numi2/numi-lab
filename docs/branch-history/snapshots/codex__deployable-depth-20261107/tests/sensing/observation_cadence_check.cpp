#include "../../src/metal/ObservationCadence.hpp"
#include <cassert>
int main() {
    metalrobo::detail::ObservationCadence cadence;
    const auto first = cadence.candidate(0.02f, 25.0f);
    assert(first.observation == 0u && first.sensor == 0u && first.sample);
    // A rejected encode does not commit its candidate.
    const auto retried = cadence.candidate(0.02f, 25.0f);
    assert(retried.observation == first.observation &&
           retried.sensor == first.sensor && retried.sample);
    cadence.commit(retried);
    const auto held = cadence.candidate(0.02f, 25.0f);
    assert(held.observation == 1u && held.sensor == 0u && !held.sample);
    cadence.commit(held);
    const auto next = cadence.candidate(0.02f, 25.0f);
    assert(next.observation == 2u && next.sensor == 1u && next.sample);
    cadence.reset();
    const auto reset = cadence.candidate(0.02f, 25.0f);
    assert(reset.observation == 0u && reset.sensor == 0u && reset.sample);
}
