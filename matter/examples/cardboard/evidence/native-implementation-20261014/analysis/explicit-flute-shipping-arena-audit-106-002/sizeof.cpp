#include <cstddef>
#include <cstdio>
#include "numi/matter/shared.h"
int main() {
#define P(T) std::printf("%s %zu\n", #T, sizeof(T))
  P(nm_float4); P(NMFEMNodeStateGPU); P(NMFEMFieldStateGPU);
  P(NMFEMTopologyNodeGPU); P(NMTetrahedronGPU); P(NMFEMElementVectorGPU);
  P(NMCohesiveFaceGPU); P(NMCohesiveFaceVectorGPU);
  P(NMDeformableContactCandidateGPU); P(NMDeformableContactGPU);
  P(NMDeformableContactHistoryGPU); P(NMContinuumSurfacePrimitiveGPU);
  P(NMIncidenceRangeGPU); P(NMContactPairGPU); P(NMContactSampleGPU);
  P(NMRigidStateGPU); P(NMRigidReactionGPU); P(NMRigidProxyGPU);
  P(NMMatterStatusGPU); P(NMSolverCertificateGPU);
#undef P
}
