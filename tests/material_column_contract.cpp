#include "NumiHumanMaterialColumnSelfContact.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace numi::matter;
using namespace numi::matter::human;

static ObjectSource validSource(const std::uint32_t n) {
    ObjectSource source;
    source.representation = Representation::fem;
    source.deformableContact = true;
    source.deformableSelfContact = false;
    source.mixedFEM = false;
    source.femNodes.resize(3u * n);
    source.tetrahedra.emplace_back();
    return source;
}

static std::vector<NMFEMTopologyNodeGPU> validNodes(
    const std::uint32_t n,
    const std::uint32_t owner
) {
    std::vector<NMFEMTopologyNodeGPU> nodes(3u * n);
    for (std::uint32_t i = 0u; i < nodes.size(); ++i)
        nodes[i].identity = {i, owner, 1u, NM_TOPOLOGY_ACTIVE};
    return nodes;
}

static bool throwsInvalid(const auto& operation) {
    try { operation(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

int main() {
    constexpr std::uint32_t owner = 7u;
    constexpr std::uint32_t n = 2u;
    auto source = validSource(n);
    validateMaterialColumnSource(source, n);
    auto nodes = validNodes(n, owner);
    tagMaterialColumnNodes(nodes, owner, n);
    for (std::uint32_t i = 0u; i < nodes.size(); ++i) {
        assert((nodes[i].identity.w & kTopologyLowFlagMask) == NM_TOPOLOGY_ACTIVE);
        assert(materialColumnTag(nodes[i].identity.w) == i % n + 1u);
    }
    assert(sameMaterialColumnAdjacency(owner, nodes[0].identity.w,
                                      owner, nodes[2].identity.w));
    assert(sameMaterialColumnAdjacency(owner, nodes[1].identity.w,
                                      owner, nodes[5].identity.w));
    assert(!sameMaterialColumnAdjacency(owner, nodes[0].identity.w,
                                       owner, nodes[1].identity.w));
    // Cross-object pairs keep their preexisting eligibility even when the
    // upper metadata happen to match.
    assert(!sameMaterialColumnAdjacency(owner, nodes[0].identity.w,
                                        owner + 1u, nodes[2].identity.w));
    assert(!sameMaterialColumnAdjacency(owner, NM_TOPOLOGY_ACTIVE,
                                        owner, NM_TOPOLOGY_ACTIVE));

    auto badIdentity = validNodes(n, owner);
    badIdentity.back().identity.x = 0u;
    assert(throwsInvalid([&] { tagMaterialColumnNodes(badIdentity, owner, n); }));
    assert(badIdentity.front().identity.w == NM_TOPOLOGY_ACTIVE);
    assert(badIdentity.back().identity.w == NM_TOPOLOGY_ACTIVE);
    assert(throwsInvalid([&] { tagMaterialColumnNodes(nodes, owner, n + 1u); }));
    assert(throwsInvalid([&] { validateMaterialColumnSource(source, 0u); }));
    assert(throwsInvalid([&] { validateMaterialColumnSource(source, kMaterialColumnIdMask + 1u); }));

    auto tooFewNodes = validSource(n);
    tooFewNodes.femNodes.pop_back();
    assert(throwsInvalid([&] { validateMaterialColumnSource(tooFewNodes, n); }));
    auto mutableSource = validSource(n);
    mutableSource.mutationPolicy.enabled = true;
    assert(throwsInvalid([&] { validateMaterialColumnSource(mutableSource, n); }));
    auto reservedMutation = validSource(n);
    reservedMutation.femCapacity.mutationCommands = 1u;
    assert(throwsInvalid([&] { validateMaterialColumnSource(reservedMutation, n); }));
    auto cohesiveSource = validSource(n);
    cohesiveSource.femCohesiveInterfaces.emplace_back();
    assert(throwsInvalid([&] { validateMaterialColumnSource(cohesiveSource, n); }));

    const auto fingerprint = materialColumnPhysicsFingerprint(1234u, owner, n);
    assert(fingerprint != 0u);
    assert(fingerprint != materialColumnPhysicsFingerprint(1234u, owner + 1u, n));
    assert(fingerprint != materialColumnPhysicsFingerprint(1234u, owner, n + 1u));
    return 0;
}
