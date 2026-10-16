#include "numi/matter/shared.h"
#include <cstdint>
#include <cstdio>
int main() {
    std::printf("NMFEMNodeStateGPU=%zu\n", sizeof(NMFEMNodeStateGPU));
    std::printf("NMTetrahedronGPU=%zu\n", sizeof(NMTetrahedronGPU));
    std::printf("NMCohesiveFaceGPU=%zu\n", sizeof(NMCohesiveFaceGPU));
    std::printf("NMCohesiveFaceVectorGPU=%zu\n", sizeof(NMCohesiveFaceVectorGPU));
    std::printf("NMContactPairGPU=%zu\n", sizeof(NMContactPairGPU));
    std::printf("NMContactSampleGPU=%zu\n", sizeof(NMContactSampleGPU));
    std::printf("NMContinuumSurfacePrimitiveGPU=%zu\n", sizeof(NMContinuumSurfacePrimitiveGPU));
    std::printf("NMDeformableContactCandidateGPU=%zu\n", sizeof(NMDeformableContactCandidateGPU));
    std::printf("NMDeformableContactGPU=%zu\n", sizeof(NMDeformableContactGPU));
    std::printf("NMIncidenceRangeGPU=%zu\n", sizeof(NMIncidenceRangeGPU));
    std::printf("NMIndirectDispatchGPU=%zu\n", sizeof(NMIndirectDispatchGPU));
    return 0;
}
