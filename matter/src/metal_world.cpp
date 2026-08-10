#include "numi/matter/metal_world.hpp"

#include <cmath>

static_assert(
    NM_MATTER_MAX_ARTICULATED_DOFS == MR_ARTICULATED_ABA_MAX_DOFS,
    "Matter and MetalWorld coupled velocity capacities must match"
);
static_assert(
    NM_MATTER_MAX_ARTICULATED_Q == MR_ARTICULATED_ABA_MAX_Q,
    "Matter and MetalWorld coupled position capacities must match"
);

namespace numi::matter {
namespace {

struct CoupledCandidateBridge {
    const metalrobo::MetalWorldDevicePhysicsPass* pass = nullptr;
};

bool encodeCoupledCandidateBridge(
    void* context,
    const CoupledCandidateQuery& query
) {
    const auto* bridge = static_cast<CoupledCandidateBridge*>(context);
    if (bridge == nullptr || bridge->pass == nullptr ||
        bridge->pass->encodeCoupledCandidate == nullptr ||
        bridge->pass->coupledCandidateContext == nullptr) {
        return false;
    }
    const metalrobo::MetalWorldCoupledCandidateQuery request{
        .input = query.input,
        .output = query.output,
        .candidateQ = query.candidateQ,
        .candidateBodies = query.candidateBodies,
        .statuses = query.statuses,
        .pointQueries = query.pointQueries,
        .pointJacobians = query.pointJacobians,
        .operation = static_cast<
            metalrobo::MetalWorldCoupledCandidateOperation>(query.operation),
        .generalizedVectorStride = query.generalizedVectorStride,
        .candidateQStride = query.candidateQStride,
        .candidateBodyStride = query.candidateBodyStride,
        .statusStride = query.statusStride,
        .pointCount = query.pointCount,
        .pointStride = query.pointStride,
        .pointJacobianStride = query.pointJacobianStride,
    };
    return bridge->pass->encodeCoupledCandidate(
        bridge->pass->coupledCandidateContext,
        *bridge->pass,
        request
    );
}

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
    request.rigid.q = pass.q;
    request.rigid.v = pass.v;
    request.rigid.currentBodies = pass.currentBodies;
    request.rigid.bodyWrenches = pass.bodyWrenches;
    request.rigid.sceneBodies = pass.sceneBodies;
    request.rigid.currentBodyCount = pass.bodyCount;
    request.rigid.currentBodyStride = pass.bodyStateStride;
    request.rigid.bodyWrenchCount = pass.bodyCount;
    request.rigid.sceneBodyCount = pass.sceneBodyCount;
    request.rigid.bodyWrenchStride = pass.bodyWrenchStride;
    request.rigid.sceneStride = pass.sceneBodyStride;
    request.rigid.qStride = pass.qStride;
    request.rigid.vStride = pass.nv;
    request.environmentStatuses = pass.environmentStatuses;
    request.rigidContactConstraints = pass.contactConstraints;
    request.rigidContactStatuses = pass.contactStatuses;
    request.rigidContactConstraintStride = pass.contactConstraintStride;
    CoupledCandidateBridge coupledBridge{&pass};
    if (pass.encodeCoupledCandidate != nullptr &&
        pass.coupledCandidateContext != nullptr) {
        request.coupledCandidateContext = &coupledBridge;
        request.encodeCoupledCandidate = &encodeCoupledCandidateBridge;
    }
    request.articulationRootBody = pass.articulationRootBody;
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
             : 0u) |
        (runtime.requiresCoupledCandidate()
             ? metalrobo::MetalWorldDevicePhysicsOwnsCoupledCandidate
             : 0u);
    program.fingerprint = programFingerprint(
        runtime.deviceProgramFingerprint(),
        program.flags
    );
    return program;
}

} // namespace numi::matter
