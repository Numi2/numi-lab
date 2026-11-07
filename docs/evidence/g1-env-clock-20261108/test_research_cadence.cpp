#include "research_cadence.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
using namespace research_cadence;
int main() {
    Record e[2]{}; Candidate c[2]{};
    for (unsigned step = 0; step < 5; ++step) {
        for (int i = 0; i < 2; ++i) {
            assert(propose(e[i], .02f, 10.f, 100 + step, false, c[i]));
            assert(c[i].sensor == 0 && c[i].sample == (step == 0));
            e[i] = c[i].accepted;
        }
    }
    assert(propose(e[0], .02f, 10.f, 105, false, c[0]));
    assert(propose(e[1], .02f, 10.f, 105, false, c[1]));
    assert(c[0].sensor == 1 && c[0].sample);
    const Record rejected = e[0];
    e[1] = c[1].accepted; // env 0 rejection preserves every field.
    assert(e[0].acceptedSteps == rejected.acceptedSteps && e[0].lastSensor == rejected.lastSensor);
    assert(e[0].captureStep == rejected.captureStep && e[0].captureTimeBits == rejected.captureTimeBits);
    assert(propose(e[0], .02f, 10.f, 205, false, c[0]));
    assert(propose(e[1], .02f, 10.f, 206, false, c[1]));
    assert(c[0].sample && !c[1].sample);
    assert(c[0].accepted.captureStep == 5 && c[0].accepted.captureTimeBits == 205);
    assert(c[1].accepted.captureStep == 5 && c[1].accepted.captureTimeBits == 105);
    assert(!propose(e[0], .02f, 10.f, 0, true, c[0]));
    Record overflow{}; overflow.acceptedSteps = none;
    assert(!propose(overflow, .02f, 10.f, 0, false, c[0]));
    assert(!propose(e[0], 0, 10.f, 0, false, c[0]));
    Record fresh{};
    for (unsigned step = 0; step < 10; ++step) {
        assert(propose(fresh, .02f, 50.f, step, false, c[0]));
        assert(c[0].sample && c[0].sensor == step);
        fresh = c[0].accepted;
    }
    std::cout << "research cadence CPU cases passed\n";
}
