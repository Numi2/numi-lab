#include "cardboard_shipping_memory.hpp"
#include <iostream>
using namespace numi::cardboard;
void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
void checkSelectedRequiredRigidArenaIsOneAllocation() {
    ShippingMemoryBudgetInput input;
    input.nodeCount = 24u;
    input.tetrahedronCount = 6u;
    input.cohesiveFaceCount = 1u;
    input.contactPairCount = 48u;
    input.activeContactCapacity = 48u;
    input.deformableContactCapacity = 64u;
    input.rigidProxyCount = 2u;
    input.fullContactArmijoRequired = true;
    input.deformableSelfContactEnabled = true;
    const auto required = estimateShippingMemoryBudget(input);
    input.rigidContactArmijoEnabled = true;
    const auto selected = estimateShippingMemoryBudget(input);
    require(required.requiredResidualDescentArenaAdmitted &&
                selected.requiredResidualDescentArenaAdmitted &&
                selected.rigidContactArmijoBytes ==
                    required.requiredRigidContactArmijoBytes &&
                selected.estimatedMetalPeakBytes ==
                    required.estimatedMetalPeakBytes &&
                selected.estimatedPeakDomainBytes ==
                    required.estimatedPeakDomainBytes,
            "selecting an already-required rigid trial arena counted a second allocation");
}

int main(){try {checkSelectedRequiredRigidArenaIsOneAllocation();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
