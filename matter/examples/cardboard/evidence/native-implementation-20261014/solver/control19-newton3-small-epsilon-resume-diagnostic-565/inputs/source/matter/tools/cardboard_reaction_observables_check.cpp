#include "cardboard_reaction_observables.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int checks = 0;

void check(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void near(const double actual, const double expected,
          const double absoluteTolerance, const std::string& message) {
    check(std::isfinite(actual) &&
              std::abs(actual - expected) <= absoluteTolerance,
          message + " (actual=" + std::to_string(actual) +
              ", expected=" + std::to_string(expected) + ")");
}

NMFEMNodeStateGPU node(const float x, const float y, const float z,
                       const float mass, const float tag) {
    NMFEMNodeStateGPU value{};
    value.positionAndMass = {x, y, z, mass};
    value.restAndFixed = {0.0f, 0.0f, 0.0f, tag};
    return value;
}

void impulseConventions() {
    const auto rigid = numi::cardboard::forceOnFemFromRigidImpulse(
        {4.0, -2.0, 1.0}, 0.02);
    near(rigid.x, -200.0, 1.0e-12,
         "rigid proxy impulse must be negated for the FEM-side force");
    near(rigid.y, 100.0, 1.0e-12,
         "rigid proxy impulse y sign must be action-reaction consistent");
    near(rigid.z, -50.0, 1.0e-12,
         "rigid proxy impulse z sign must be action-reaction consistent");
    near(rigid.x * 0.02 + 4.0, 0.0, 1.0e-12,
         "FEM and proxy impulses must sum to zero");

    const auto deformable =
        numi::cardboard::forceOnFemFromDeformableResidualImpulse(
            {3.0, -6.0, 9.0}, 0.4, 0.02);
    near(deformable.x, 60.0, 1.0e-12,
         "deformable source/contact time scaling x");
    near(deformable.y, -120.0, 1.0e-12,
         "deformable source/contact time scaling y");
    near(deformable.z, 180.0, 1.0e-12,
         "deformable source/contact time scaling z");
}

void acceptedConstraintReduction() {
    std::vector<NMFEMNodeStateGPU> states(8u);
    // Environment 0 is intentionally unrelated. The selected environment 1
    // starts at node four and contains a grip, a Human attachment, a free node,
    // and a second prescribed grip.
    states[0] = node(-100.0f, 0.0f, 0.0f, 0.0f, 3.0f);
    states[4] = node(1.0f, 2.0f, 3.0f, 0.0f, 3.0f);
    states[5] = node(10.0f, 0.0f, 0.0f, 0.0f, 2.0f);
    states[6] = node(20.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    states[7] = node(3.0f, 2.0f, 3.0f, 0.0f, 1.0f);

    std::vector<nm_float4> bulk(8u, nm_float4{});
    std::vector<nm_float4> contact(8u, nm_float4{});
    bulk[4] = {2.0f, 0.0f, 0.0f, 0.0f};
    contact[4] = {-0.5f, 0.0f, 0.0f, 0.0f};
    bulk[5] = {0.0f, 100.0f, 0.0f, 0.0f};
    contact[5] = {0.0f, 200.0f, 0.0f, 0.0f};
    bulk[6] = {0.0f, 300.0f, 0.0f, 0.0f};
    contact[6] = {0.0f, 400.0f, 0.0f, 0.0f};
    bulk[7] = {2.0f, 0.0f, 0.0f, 0.0f};
    contact[7] = {0.0f, 0.0f, 1.0f, 0.0f};
    const std::array<std::uint32_t, 4> selected{0u, 1u, 2u, 3u};
    const auto summary = numi::cardboard::summarizeConstraintReactions(
        states, bulk, contact, 4u, 4u, selected);
    check(summary.nodeCount == 2u,
          "Human/free rows must not enter the static grip reaction channel");
    near(summary.pivot.x, 2.0, 1.0e-12,
         "default pivot must use accepted constrained-node positions");
    near(summary.pivot.y, 2.0, 1.0e-12,
         "accepted position y pivot");
    near(summary.pivot.z, 3.0, 1.0e-12,
         "accepted position z pivot");
    near(summary.mechanicalConstraintForce.x, 4.0, 1.0e-12,
         "continuum/cohesive/inertial constraint reaction reduction");
    near(summary.contactForce.x, -0.5, 1.0e-12,
         "separate contact reaction reduction");
    near(summary.contactForce.z, 1.0, 1.0e-12,
         "contact reaction is additive on the FEM constraint support");
    near(summary.totalForce.x, 3.5, 1.0e-12,
         "combined clamp force must contain both production channels");
    near(summary.totalForce.z, 1.0, 1.0e-12,
         "combined contact force z");
    near(summary.contactMoment.y, -1.0, 1.0e-12,
         "contact moment must use accepted node position about accepted grip centroid");
    near(summary.totalMoment.y, -1.0, 1.0e-12,
         "total support moment includes the contact moment exactly once");

    bool threw = false;
    try {
        const std::array<std::uint32_t, 1> outside{4u};
        (void)numi::cardboard::summarizeConstraintReactions(
            states, bulk, contact, 4u, 4u, outside);
    } catch (const std::out_of_range&) { threw = true; }
    check(threw, "a grip index from the next environment must be rejected");
}

void invalidInputsFailClosed() {
    bool badDt = false;
    try {
        (void)numi::cardboard::forceOnFemFromRigidImpulse({1.0, 0.0, 0.0}, 0.0);
    } catch (const std::invalid_argument&) { badDt = true; }
    check(badDt, "zero timestep must reject rigid reaction conversion");

    bool badScale = false;
    try {
        (void)numi::cardboard::forceOnFemFromDeformableResidualImpulse(
            {1.0, 0.0, 0.0}, -1.0, 0.01);
    } catch (const std::invalid_argument&) { badScale = true; }
    check(badScale, "negative deformable time scale must be rejected");

    bool badEnvironment = false;
    std::vector<NMFEMNodeStateGPU> states(5u);
    std::vector<nm_float4> bulk(5u), contact(5u);
    const std::array<std::uint32_t, 0> empty{};
    try {
        (void)numi::cardboard::summarizeConstraintReactions(
            states, bulk, contact, 4u, 2u, empty);
    } catch (const std::invalid_argument&) { badEnvironment = true; }
    check(badEnvironment,
          "environment slice extending past the state arena must be rejected");
}

} // namespace

int main() {
    try {
        impulseConventions();
        acceptedConstraintReduction();
        invalidInputsFailClosed();
        std::cout << "cardboard reaction observables CPU check passed "
                  << checks << " checks\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "cardboard reaction observables CPU check failed after "
                  << checks << " checks: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
