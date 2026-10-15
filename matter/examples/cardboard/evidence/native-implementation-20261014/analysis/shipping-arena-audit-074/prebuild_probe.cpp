#include "cardboard_shipping_memory.hpp"
#include "cardboard_box_assembly_controller.hpp"
#include <cstdint>
#include <iostream>

int main() {
    using namespace numi::cardboard;
    constexpr std::uint32_t restart = 16u;
    const auto screen = screenShippingPrebuildMemory(1.025, 0.0079, 1u,
                                                     restart, 48u, 1u);
    ShippingPrebuildArenaOptions options;
    options.cohesiveFaceUpperBound = screen.nominalTetrahedronCount / 5u;
    options.rigidProxyCount = kShippingRigidProxyCount;
    options.rigidBodyCount = kShippingRigidBodyCount;
    options.fgmresRestart = restart;
    options.requireFullContactResidualDescent = true;
    const auto budget = estimateShippingPrebuildMemoryBudget(screen, options);
    std::cout << "{\"nominal_nodes\":" << screen.nominalNodeCount
              << ",\"nominal_tetrahedra\":" << screen.nominalTetrahedronCount
              << ",\"estimated_nodes\":" << screen.estimatedNodeCount
              << ",\"estimated_tetrahedra\":" << screen.estimatedTetrahedronCount
              << ",\"rigid_proxy_count\":" << kShippingRigidProxyCount
              << ",\"prebuild_pair_upper_bound\":"
              << screen.nominalNodeCount * kShippingRigidProxyCount
              << ",\"dense_pair_limit\":65536"
              << ",\"default_deformable_capacity\":"
              << screen.nominalTetrahedronCount * 8ull
              << ",\"private_rigid_admitted\":"
              << (budget.privateRigidContactAdmitted ? "true" : "false")
              << ",\"private_rigid_failure\":\""
              << budget.privateRigidContactFailure << "\""
              << ",\"private_descent_admitted\":"
              << (budget.requiredResidualDescentArenaAdmitted ? "true" : "false")
              << ",\"private_descent_failure\":\""
              << budget.requiredResidualDescentArenaFailure << "\""
              << ",\"required_rigid_bytes\":"
              << budget.requiredRigidContactArmijoBytes
              << ",\"required_full_contact_bytes\":"
              << budget.requiredFullContactArmijoBytes
              << ",\"required_cohesive_bytes\":"
              << budget.requiredCohesiveArmijoBytes
              << ",\"modeled_peak_bytes\":" << budget.estimatedPeakDomainBytes
              << ",\"within_16gib_model\":"
              << (budget.withinLimit ? "true" : "false")
              << ",\"qualification\":false}\n";
}
