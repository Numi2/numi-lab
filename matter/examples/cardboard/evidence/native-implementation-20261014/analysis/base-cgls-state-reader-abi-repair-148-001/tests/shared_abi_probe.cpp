#include <cstddef>
#include <cstdio>
#include <numi/matter/shared.h>

static_assert(sizeof(NMFEMNodeStateGPU) == 80);
static_assert(offsetof(NMFEMNodeStateGPU, positionAndMass) == 0);
static_assert(offsetof(NMFEMNodeStateGPU, velocityAndInverseMass) == 16);
static_assert(offsetof(NMFEMNodeStateGPU, restAndFixed) == 32);
static_assert(offsetof(NMFEMNodeStateGPU, deltaVelocity) == 48);
static_assert(offsetof(NMFEMNodeStateGPU, referenceDisplacementAndMode) == 64);
static_assert(sizeof(NMTetrahedronGPU) == 96);
static_assert(offsetof(NMTetrahedronGPU, nodes) == 0);
static_assert(offsetof(NMTetrahedronGPU, inverseRestRow0) == 16);
static_assert(offsetof(NMTetrahedronGPU, inverseRestRow1) == 32);
static_assert(offsetof(NMTetrahedronGPU, inverseRestRow2) == 48);
static_assert(offsetof(NMTetrahedronGPU, identity) == 64);
static_assert(offsetof(NMTetrahedronGPU, materialFrameRotation) == 80);

int main() {
    std::printf("record:NMFEMNodeStateGPU %zu\n", sizeof(NMFEMNodeStateGPU));
    std::printf("record:NMTetrahedronGPU %zu\n", sizeof(NMTetrahedronGPU));
    std::printf("NMFEMNodeStateGPU.positionAndMass %zu\n", offsetof(NMFEMNodeStateGPU, positionAndMass));
    std::printf("NMFEMNodeStateGPU.velocityAndInverseMass %zu\n", offsetof(NMFEMNodeStateGPU, velocityAndInverseMass));
    std::printf("NMFEMNodeStateGPU.restAndFixed %zu\n", offsetof(NMFEMNodeStateGPU, restAndFixed));
    std::printf("NMFEMNodeStateGPU.deltaVelocity %zu\n", offsetof(NMFEMNodeStateGPU, deltaVelocity));
    std::printf("NMFEMNodeStateGPU.referenceDisplacementAndMode %zu\n", offsetof(NMFEMNodeStateGPU, referenceDisplacementAndMode));
    std::printf("NMTetrahedronGPU.nodes %zu\n", offsetof(NMTetrahedronGPU, nodes));
    std::printf("NMTetrahedronGPU.inverseRestRow0 %zu\n", offsetof(NMTetrahedronGPU, inverseRestRow0));
    std::printf("NMTetrahedronGPU.inverseRestRow1 %zu\n", offsetof(NMTetrahedronGPU, inverseRestRow1));
    std::printf("NMTetrahedronGPU.inverseRestRow2 %zu\n", offsetof(NMTetrahedronGPU, inverseRestRow2));
    std::printf("NMTetrahedronGPU.identity %zu\n", offsetof(NMTetrahedronGPU, identity));
    std::printf("NMTetrahedronGPU.materialFrameRotation %zu\n", offsetof(NMTetrahedronGPU, materialFrameRotation));
}
