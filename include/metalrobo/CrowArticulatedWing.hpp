#pragma once

#include "metalrobo/RunProgram.hpp"

namespace metalrobo {

// EXPERIMENTAL MECHANICS FIXTURE: not a flight-capable RobotPack.
// The input must be the source-identified v15 hip-roll/toe research body.
// The result intentionally has no flapping-wing aerodynamic program until
// segment-attached aerodynamic forces are available and independently tested.
[[nodiscard]] RobotPack makeCrowArticulatedWingMechanicsFixture(RobotPack v15);

} // namespace metalrobo
