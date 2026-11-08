#include "../apps/SutureRodWitnessAudit.hpp"
#include <cassert>
#include <limits>
#include <vector>
int main() {
    std::vector<MRRodColliderGPU> colliders(1);
    colliders[0].topologyGeneration = 7u;
    std::vector<MRRodToolPairGPU> pairs(1);
    pairs[0].rodCollider = 0u; pairs[0].rigidCollider = 4u;
    std::vector<MRRodToolWitnessGPU> witnesses(MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR);
    auto& w = witnesses[0];
    w.featuresAndFlags.w = MR_ROD_TOOL_WITNESS_VALID | MR_ROD_TOOL_WITNESS_WARM_STARTED;
    w.identity = {0u, 0u, 0u, 4u};
    w.materialAndGeneration.z = 7u;
    w.impulses.x = 0.25f;
    auto a = auditSutureRodWitnesses(witnesses, pairs, colliders);
    assert(a.capacity && a.identity && a.finiteImpulses && a.valid == 1u &&
        a.warmStarted == 1u && a.nonzeroImpulse == 1u);
    witnesses.pop_back();
    assert(!auditSutureRodWitnesses(witnesses, pairs, colliders).capacity);
    witnesses.resize(MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR);
    witnesses[0].materialAndGeneration.z = 8u;
    assert(!auditSutureRodWitnesses(witnesses, pairs, colliders).identity);
    witnesses[0].materialAndGeneration.z = 7u;
    witnesses[0].identity.w = 3u;
    assert(!auditSutureRodWitnesses(witnesses, pairs, colliders).identity);
    witnesses[0].identity.w = 4u;
    witnesses[0].identity.x = 1u;
    assert(!auditSutureRodWitnesses(witnesses, pairs, colliders).identity);
    witnesses[0].identity.x = 0u;
    pairs[0].rodCollider = 1u;
    assert(!auditSutureRodWitnesses(witnesses, pairs, colliders).identity);
    pairs[0].rodCollider = 0u;
    witnesses[0].impulses.x = std::numeric_limits<float>::quiet_NaN();
    assert(!auditSutureRodWitnesses(witnesses, pairs, colliders).finiteImpulses);
}
