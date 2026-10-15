#include "cardboard_shipping_memory.hpp"
#include "cardboard_box_assembly_controller.hpp"

#include <cstdint>
#include <iostream>

int main() {
    using namespace numi::cardboard;
    constexpr std::uint32_t restart = 16u;
    const auto screen = screenShippingPrebuildMemory(
        1.025, 0.0079, 1u, restart, 48u, 1u);

    ShippingPrebuildArenaOptions denseOptions;
    denseOptions.cohesiveFaceUpperBound =
        shippingPrebuildTetrahedronUpperBound(screen) / 5u;
    denseOptions.rigidProxyCount = kShippingRigidProxyCount;
    denseOptions.rigidBodyCount = kShippingRigidBodyCount;
    denseOptions.rigidGeneralizedCapacity = 6u;
    denseOptions.fgmresRestart = restart;
    denseOptions.compactImmutableSurfaceMap = true;
    denseOptions.requireFullContactResidualDescent = true;
    denseOptions.deformableSelfContactEnabled = true;
    const auto dense = estimateShippingPrebuildMemoryBudget(screen, denseOptions);

    ShippingPrebuildArenaOptions sparseOptions = denseOptions;
    const bool boundedCandidate = shippingPrebuildBoundedFreeBoxScopeCandidate(
        true, true, kShippingRigidProxyCount, kShippingRigidBodyCount, 6u,
        kShippingRigidProxyCount, kShippingRigidBodyCount);
    configureShippingPrebuildSparsePrivateContact(
        sparseOptions, "65536", nullptr, true, boundedCandidate);
    const auto sparse = estimateShippingPrebuildMemoryBudget(screen, sparseOptions);

    std::cout
        << "{\"schema\":\"numi.cardboard.prebuild-sparse-wiring-check.v1\""
        << ",\"nominal_nodes\":" << screen.nominalNodeCount
        << ",\"nominal_tetrahedra\":" << screen.nominalTetrahedronCount
        << ",\"padded_nodes\":" << screen.estimatedNodeCount
        << ",\"padded_tetrahedra\":" << screen.estimatedTetrahedronCount
        << ",\"prepared_node_upper_bound\":"
        << shippingPrebuildRigidNodeUpperBound(screen)
        << ",\"tetrahedron_upper_bound\":"
        << shippingPrebuildTetrahedronUpperBound(screen)
        << ",\"rigid_proxy_count\":" << kShippingRigidProxyCount
        << ",\"actual_contact_eligible_pairs\":" << 527964u * 12u
        << ",\"conservative_pair_domain_upper_bound\":"
        << sparse.rigidContactPairDomainUpperBound
        << ",\"default_dense\":{\"private_rigid_admitted\":"
        << (dense.privateRigidContactAdmitted ? "true" : "false")
        << ",\"private_descent_admitted\":"
        << (dense.requiredResidualDescentArenaAdmitted ? "true" : "false")
        << ",\"modeled_peak_bytes\":" << dense.estimatedPeakDomainBytes
        << ",\"failure\":\"" << dense.privateRigidContactFailure << "\"}"
        << ",\"explicit_sparse\":{\"requested\":"
        << (sparse.sparsePrivateContactRequested ? "true" : "false")
        << ",\"capacity\":" << sparse.sparsePrivateContactCapacity
        << ",\"scope_candidate\":"
        << (sparse.sparsePrivateContactBoundedFreeBoxScope ? "true" : "false")
        << ",\"private_rigid_admitted\":"
        << (sparse.privateRigidContactAdmitted ? "true" : "false")
        << ",\"sparse_private_bytes\":" << sparse.sparsePrivateContactBytes
        << ",\"full_contact_arena_admitted\":"
        << (sparse.requiredResidualDescentArenaAdmitted ? "true" : "false")
        << ",\"failure\":\""
        << sparse.requiredResidualDescentArenaFailure << "\"}"
        << ",\"qualification\":false}\n";
}
