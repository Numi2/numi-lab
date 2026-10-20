#include "metalrobo/MetalArticulatedOperator.hpp"
#include "numi/matter/detail/gravity_scale.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

int main() {
    metalrobo::MetalNumiHumanGravityScaleRamp disabled{};
    assert(!disabled.configured());
    assert(disabled.scaleAtAcceptedStep(1u) == 1.0f);

    metalrobo::MetalNumiHumanGravityScaleRamp ramp{};
    ramp.firstAcceptedStep = 1u;
    ramp.fullGravityAcceptedStep = 3u;
    ramp.initialScale = 0.0f;
    ramp.fullGravityScale = 1.0f;
    assert(ramp.scaleAtAcceptedStep(1u) == 0.0f);
    assert(ramp.scaleAtAcceptedStep(2u) == 0.5f);
    assert(ramp.scaleAtAcceptedStep(3u) == 1.0f);

    const std::array<float, 3> base{0.0f, 0.0f, -9.81f};
    std::array<float, 3> output{7.0f, 8.0f, 9.0f};
    assert(numi::matter::detail::scaleGravityFromBase(base, 1.0f, output));
    assert(output == base);
    assert(numi::matter::detail::scaleGravityFromBase(base, 0.0f, output));
    assert(output == (std::array<float, 3>{0.0f, 0.0f, 0.0f}));
    const auto pendingStep = 2u;
    const float retryScale = ramp.scaleAtAcceptedStep(pendingStep);
    assert(numi::matter::detail::scaleGravityFromBase(base, retryScale, output));
    const auto midpoint = output;
    for (unsigned rejection = 0u; rejection < 3u; ++rejection) {
        assert(ramp.scaleAtAcceptedStep(pendingStep) == retryScale);
        assert(numi::matter::detail::scaleGravityFromBase(base, retryScale, output));
        assert(output == midpoint);
    }
    assert(numi::matter::detail::scaleGravityFromBase(base, retryScale, output));
    assert(output == midpoint);

    const std::array<float, 3> sentinel{1.0f, 2.0f, 3.0f};
    for (float invalid : {-0.01f, 1.01f,
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
        output = sentinel;
        assert(!numi::matter::detail::scaleGravityFromBase(base, invalid, output));
        assert(output == sentinel);
    }
    return 0;
}
