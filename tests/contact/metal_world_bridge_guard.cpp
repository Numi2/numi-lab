// Exercise the actual adapter without GPU submission. The owner callback must
// never see a request whose precision/ownership fields it cannot represent.
#include "../../matter/src/metal_world.cpp"
#include <cstdint>
#include <iostream>

namespace {
unsigned calls = 0;
metalrobo::MetalWorldCoupledCandidateQuery captured{};
bool owner(void*, const metalrobo::MetalWorldDevicePhysicsPass&,
           const metalrobo::MetalWorldCoupledCandidateQuery& q) {
    ++calls; captured = q; return true;
}
bool encodeProgram(void*, const metalrobo::MetalWorldDevicePhysicsPass&) { return true; }
void abortProgram(void*, void*) {}
void require(bool value, const char* message) {
    if (!value) { std::cerr << message << '\n'; std::exit(1); }
}
}
int main() {
    using namespace numi::matter;
    metalrobo::MetalWorldDevicePhysicsPass pass{};
    pass.coupledCandidateContext = &calls;
    pass.encodeCoupledCandidate = &owner;
    CoupledCandidateBridge bridge{&pass};
    CoupledCandidateQuery q{};
    q.generalizedVectorStride = 9;
    q.indirectDispatchArgumentsGPUAddress = 19;
    require(encodeCoupledCandidateBridge(&bridge, q), "legacy geometry was rejected");
    require(calls == 1 && captured.generalizedVectorStride == 9 &&
            captured.indirectDispatchArgumentsGPUAddress == 19, "legacy fields were not forwarded");
    const auto reject = [&](const CoupledCandidateQuery& candidate) {
        const unsigned before = calls;
        require(!encodeCoupledCandidateBridge(&bridge, candidate) && calls == before,
                "unsupported request reached owner or reported success");
    };
    auto bad = q; bad.legacyHighOnly = false; reject(bad);
    bad.operation = CoupledCandidateOperation::candidateHalfStepKinematics; reject(bad);
    // Non-geometry operations may carry the global precision flag with no
    // geometry outputs; they remain valid and must still reach the owner.
    bad.operation = CoupledCandidateOperation::massAction;
    require(encodeCoupledCandidateBridge(&bridge, bad), "mass action rejected global precision flag");
    auto pointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(1));
#define REJECT_VALUE(field, value) do { bad = q; bad.field = value; reject(bad); } while (false)
    REJECT_VALUE(candidateRootTranslation, pointer);
    REJECT_VALUE(candidateBodyPositionLow, pointer);
    REJECT_VALUE(pointPositionLow, pointer);
    REJECT_VALUE(candidateRootTranslationGPUAddress, 1u);
    REJECT_VALUE(candidateBodyPositionLowGPUAddress, 1u);
    REJECT_VALUE(pointPositionLowGPUAddress, 1u);
    REJECT_VALUE(candidateRootTranslationElementCount, 1u);
    REJECT_VALUE(candidateBodyPositionLowElementCount, 1u);
    REJECT_VALUE(pointPositionLowElementCount, 1u);
    REJECT_VALUE(borrowedComputeEncoder, pointer);
    REJECT_VALUE(borrowedCommandBuffer, pointer);
    REJECT_VALUE(recordSourceMassOperationICB, true);
    REJECT_VALUE(sourceMassOperationICBRecordOutput, pointer);
#undef REJECT_VALUE
    metalrobo::MetalWorldDevicePhysicsProgram program{};
    program.context=&calls; program.encode=&encodeProgram; program.abort=&abortProgram;
    program.fingerprint=1u; program.coupledCandidatePointCapacity=1u;
    program.flags=metalrobo::MetalWorldDevicePhysicsOwnsCoupledCandidate |
        metalrobo::MetalWorldDevicePhysicsOwnsSliderMidpoint |
        metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches;
    require(program.valid(),"restricted slider moment capability rejected");
    program.flags|=metalrobo::MetalWorldDevicePhysicsRequiresRigidContactEvidence;
    require(!program.valid(),"slider admitted unrelated rigid contact owner");
    program.flags=metalrobo::MetalWorldDevicePhysicsOwnsCoupledCandidate |
        metalrobo::MetalWorldDevicePhysicsOwnsSliderMidpoint |
        metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches |
        metalrobo::MetalWorldDevicePhysicsCouplesRodNodes;
    require(!program.valid(),"slider admitted rod owner");
    std::cout << "unsupported precision and ownership requests rejected; slider moment capability admitted\n";
}
