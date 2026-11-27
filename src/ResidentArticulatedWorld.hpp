#pragma once

#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/resident_articulated_world_types.h"

namespace metalrobo::detail {

// Shared validation and lowering for the live runtime and recorded clips.
MRResidentProgramGPU compileResidentArticulation(
    const CompiledWorld& world, std::span<const MRBodyStateGPU> sceneBodies
);
void bindResidentArticulationActions(
    const CompiledTaskProgram& bindings, float controlSeconds, MRResidentProgramGPU& program
);

} // namespace metalrobo::detail
