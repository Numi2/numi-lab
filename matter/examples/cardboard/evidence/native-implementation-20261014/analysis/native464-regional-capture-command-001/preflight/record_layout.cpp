#include <numi/matter/shared.h>
#include <cstddef>
#include <iostream>
#define P(T) std::cout << #T << "=" << sizeof(T) << "\n"
int main(){
P(NMMatterDispatchGPU); P(NMFEMNodeStateGPU); P(NMTetrahedronGPU);
P(NMContinuumObjectGPU); P(NMMaterialGPU); P(NMScalarProgramGPU);
P(NMExpressionInstructionGPU); P(NMSchedulerStateGPU); P(NMAdaptiveStateGPU);
P(NMFGMRESStateGPU); P(NMMixedSolverGPU); P(NMMatterStatusGPU); P(NMIncidenceRangeGPU);
std::cout << "restAndFixed.w=" << offsetof(NMFEMNodeStateGPU, restAndFixed)+offsetof(nm_float4,w) << "\n";
}
