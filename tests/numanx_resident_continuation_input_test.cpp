#include "../src/metal/NumanXResidentContinuationInput.hpp"

#include <array>
#include <cassert>

using namespace metalrobo;

int main() {
    std::array<float, 2> q{1.0f, 2.0f}, v{3.0f, 4.0f};
    std::array<MRCompensatedRootTranslationGPU, 1> root{};
    std::array<MRMujocoMuscleStateGPU, 1> state{};
    const auto input = [&] {
        MetalArticulatedOperatorInput x{};
        x.q = q; x.v = v; x.rootTranslations = root;
        x.stand.v = v; x.mujoco.states = state;
        return x;
    };
    const auto hasImage = [](const MetalArticulatedOperatorInput& x) {
        return !x.q.empty() && !x.v.empty() && !x.rootTranslations.empty() &&
            !x.stand.v.empty() && !x.mujoco.states.empty();
    };
    const auto hasNoImage = [](const MetalArticulatedOperatorInput& x) {
        return x.q.empty() && x.v.empty() && x.rootTranslations.empty() &&
            x.stand.v.empty() && x.mujoco.states.empty();
    };
    auto initial = input();
    selectNumanXResidentContinuationInput(initial);
    assert(hasImage(initial));
    assert(initial.q.data() == q.data() && initial.v.data() == v.data());
    auto continuation = input();
    continuation.residentContinuation.previousTransactionFingerprint = 11;
    continuation.residentContinuation.previousPhysicsGeneration = 7;
    selectNumanXResidentContinuationInput(continuation);
    assert(hasNoImage(continuation));
    assert(continuation.residentContinuation.validLegacy());
    assert(continuation.residentContinuation.previousTransactionFingerprint == 11);
    assert(continuation.residentContinuation.previousPhysicsGeneration == 7);
    auto exact = input();
    exact.residentContinuation = {11, 7, 13, 17};
    selectNumanXResidentContinuationInput(exact);
    assert(hasNoImage(exact) && exact.residentContinuation.validExact());
    assert(exact.residentContinuation.previousAcceptedTokenFingerprint == 13);
    assert(exact.residentContinuation.previousHumanIOProgramFingerprint == 17);
    auto partial = input();
    partial.residentContinuation.previousTransactionFingerprint = 11;
    selectNumanXResidentContinuationInput(partial);
    assert(hasNoImage(partial) && !partial.residentContinuation.valid());
    auto badExact = input();
    badExact.residentContinuation = {11, 7, 13, 0};
    selectNumanXResidentContinuationInput(badExact);
    assert(hasNoImage(badExact) && !badExact.residentContinuation.valid());
}
