#include "numi/matter/metal_world.hpp"

#include <cmath>

namespace numi::matter {
namespace {

std::uint64_t programFingerprint(
    const std::uint64_t runtimeFingerprint,
    const std::uint32_t flags
) noexcept {
    std::uint64_t fingerprint = runtimeFingerprint;
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
        fingerprint ^= static_cast<std::uint8_t>(flags >> shift);
        fingerprint *= 1099511628211ull;
    }
    return fingerprint == 0u ? 1u : fingerprint;
}

bool encodeMetalWorldMatter(
    void* context,
    const metalrobo::MetalWorldDevicePhysicsPass& pass
) {
    if (context == nullptr || pass.commandBuffer == nullptr ||
        pass.physicsSubsteps == 0u ||
        pass.physicsSubstep >= pass.physicsSubsteps) {
        return false;
    }
    auto& runtime = *static_cast<Runtime*>(context);
    EncodeRequest request{};
    request.commandBuffer = pass.commandBuffer;
    request.rigid.currentBodies = pass.currentBodies;
    request.rigid.bodyWrenches = pass.bodyWrenches;
    request.rigid.sceneBodies = pass.sceneBodies;
    request.rigid.currentBodyCount = pass.bodyCount;
    request.rigid.currentBodyStride = pass.bodyStateStride;
    request.rigid.bodyWrenchCount = pass.bodyCount;
    request.rigid.sceneBodyCount = pass.sceneBodyCount;
    request.rigid.bodyWrenchStride = pass.bodyWrenchStride;
    request.rigid.sceneStride = pass.sceneBodyStride;
    request.environmentStatuses = pass.environmentStatuses;
    request.rigidContactConstraints = pass.contactConstraints;
    request.rigidContactStatuses = pass.contactStatuses;
    request.rigidContactConstraintStride = pass.contactConstraintStride;
    request.phase = pass.phase ==
            metalrobo::MetalWorldDevicePhysicsPhase::postCommit
        ? EncodePhase::postCommit
        : EncodePhase::preDynamics;
    const bool firstPrePass =
        request.phase == EncodePhase::preDynamics &&
        pass.physicsSubstep == 0u;
    request.resetMasks = firstPrePass ? pass.resetMasks : nullptr;
    request.resetMaskStepStride = firstPrePass
        ? pass.resetMaskStepStride
        : 0u;
    const float cookedTimestep = runtime.timestepSeconds();
    if (!std::isfinite(pass.timestepSeconds) ||
        !(pass.timestepSeconds > 0.0f) ||
        pass.timestepSeconds != cookedTimestep) {
        return false;
    }
    request.controlStep = pass.controlStep;
    request.physicsSubstep = pass.physicsSubstep;
    request.physicsSubsteps = pass.physicsSubsteps;
    request.seed = pass.seed;
    request.timestepSeconds = pass.timestepSeconds;
    // A MetalWorld submission encodes its full rollout horizon into one
    // command buffer. Identification therefore updates/samples once at the
    // beginning of that horizon; later control steps consume the same
    // candidate material overlay and cannot race a CPU-authored loss buffer.
    request.runIdentification =
        runtime.automaticIdentificationEnabled() &&
        request.phase == EncodePhase::preDynamics &&
        pass.controlStep == 0u &&
        pass.physicsSubstep == 0u;
    request.runAdaptiveTransfer =
        runtime.adaptiveTransferEnabled() &&
        request.phase == EncodePhase::postCommit &&
        pass.physicsSubstep + 1u == pass.physicsSubsteps;
    return runtime.encode(request).encoded;
}

void abortMetalWorldMatter(
    void* context,
    void* commandBuffer
) {
    if (context != nullptr && commandBuffer != nullptr) {
        static_cast<Runtime*>(context)->cancel(commandBuffer);
    }
}

} // namespace

metalrobo::MetalWorldDevicePhysicsProgram
makeMetalWorldDevicePhysicsProgram(Runtime& runtime) noexcept {
    metalrobo::MetalWorldDevicePhysicsProgram program{};
    if (!runtime.valid() || runtime.deviceProgramFingerprint() == 0u) {
        return program;
    }
    program.context = &runtime;
    program.encode = &encodeMetalWorldMatter;
    program.abort = &abortMetalWorldMatter;
    program.flags =
        (runtime.requiresBodyWrenches()
             ? metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches
             : 0u) |
        (runtime.requiresRigidContactEvidence()
             ? metalrobo::
                   MetalWorldDevicePhysicsRequiresRigidContactEvidence
             : 0u);
    program.fingerprint = programFingerprint(
        runtime.deviceProgramFingerprint(),
        program.flags
    );
    return program;
}

} // namespace numi::matter
