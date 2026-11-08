#include "../apps/SutureRigidCacheIdentity.hpp"
#include <cassert>
#include <vector>
int main() {
    std::vector<MRShapeGPU> shapes(2);
    shapes[0].slotGeneration = 7; shapes[1].slotGeneration = 9;
    std::vector<MRManifoldHeaderGPU> headers(1);
    auto& h = headers[0];
    h.pairAndCount[0] = 0; h.pairAndCount[1] = 0;
    h.pairAndCount[2] = 1; h.pairAndCount[3] = 1;
    h.generationsAndFlags[0] = 7; h.generationsAndFlags[1] = 9;
    assert(sutureRigidCacheTopologyMatches(headers, shapes));
    h.generationsAndFlags[1] = 8;
    assert(!sutureRigidCacheTopologyMatches(headers, shapes));
    h.generationsAndFlags[1] = 9; h.pairAndCount[2] = 2;
    assert(!sutureRigidCacheTopologyMatches(headers, shapes));
    h.pairAndCount[2] = 1; h.pairAndCount[0] = 1;
    assert(!sutureRigidCacheTopologyMatches(headers, shapes));
    h.pairAndCount[0] = 0; h.pairAndCount[3] = 0;
    assert(!sutureRigidCacheTopologyMatches(headers, shapes));
    h.pairAndCount[3] = MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY + 1;
    assert(!sutureRigidCacheTopologyMatches(headers, shapes));
}
