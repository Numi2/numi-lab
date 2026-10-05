#pragma once

#include "numi/matter/matter.hpp"
#include "metalrobo/MetalWorld.hpp"

namespace numi::matter {

// Adapts a persistent Matter Runtime to MetalWorld's per-substep borrowed
// command-buffer boundary. The returned program does not own Runtime; the
// Runtime must outlive every MetalWorld submission that references it.
[[nodiscard]] metalrobo::MetalWorldDevicePhysicsProgram
makeMetalWorldDevicePhysicsProgram(Runtime& runtime) noexcept;

// The same adapter with a caller-owned accepted-step extension. This is for
// scenes that compose additional device state with the existing rigid/Matter
// transaction; it introduces no second integration or contact authority.
// The enclosing program fingerprint must bind the extension's code/config.
[[nodiscard]] bool encodeMetalWorldDevicePhysics(
    Runtime& runtime,
    const metalrobo::MetalWorldDevicePhysicsPass& pass,
    void* extensionContext,
    EncodeAcceptedStepExtension extension
);

} // namespace numi::matter
