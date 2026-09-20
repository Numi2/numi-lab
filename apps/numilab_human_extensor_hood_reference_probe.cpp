#include "metalrobo/NumiHumanTensionNetwork.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using metalrobo::NumiHumanTensionNetworkElement;
using metalrobo::NumiHumanTensionNetworkLoad;
using metalrobo::NumiHumanTensionNetworkNode;
using metalrobo::NumiHumanTensionNetworkResult;

double length(const std::array<double, 3>& a,
              const std::array<double, 3>& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

std::array<double, 3> normalizedFromTo(
    const std::array<double, 3>& from,
    const std::array<double, 3>& to,
    const double magnitude
) {
    std::array<double, 3> value{
        to[0] - from[0], to[1] - from[1], to[2] - from[2]};
    const double norm = std::hypot(value[0], value[1], value[2]);
    if (!(norm > 0.0)) throw std::runtime_error("degenerate muscle direction");
    for (double& component : value) component *= magnitude / norm;
    return value;
}

struct Fixture {
    std::vector<NumiHumanTensionNetworkNode> nodes;
    std::vector<NumiHumanTensionNetworkElement> elements;
    std::vector<NumiHumanTensionNetworkLoad> loads;
};

Fixture makeFixture(const bool intercrossing) {
    // Literature-scale canonical middle-finger dorsal topology. Coordinates
    // are a transparent preflight layout; live Human integration replaces
    // them with posed MyoSim/BodyParts3D attachment coordinates.
    Fixture fixture;
    fixture.nodes = {
        {{ 0.000, 0.045,  0.000}, true},  // middle-phalanx attachment
        {{ 0.000, 0.086,  0.000}, true},  // distal-phalanx attachment
        {{ 0.000, 0.000, -0.004}, true},  // sagittal-band/capsule anchor
        {{ 0.000,-0.018,  0.005}, false}, // EDC input
        {{-0.007, 0.002,  0.002}, false}, // radial interosseous input
        {{ 0.007, 0.002,  0.002}, false}, // ulnar interosseous input
        {{-0.004, 0.002, -0.002}, false}, // lumbrical input
        {{ 0.000, 0.025,  0.005}, false}, // medial-band junction
        {{-0.004, 0.041,  0.003}, false}, // radial lateral band
        {{ 0.004, 0.041,  0.003}, false}, // ulnar lateral band
        {{ 0.000, 0.066,  0.003}, false}, // terminal-tendon junction
    };
    const auto add = [&](const std::uint32_t a, const std::uint32_t b,
                         const double modulusMPa, const double areaMM2,
                         const double restScale = 0.99) {
        fixture.elements.push_back({
            a, b,
            restScale * length(
                fixture.nodes[a].position, fixture.nodes[b].position),
            modulusMPa * 1.0e6, areaMM2 * 1.0e-6});
    };
    add(3, 7, 110.0, 1.0); // EDC medial input
    add(7, 0, 110.0, 1.0); // medial tendon attachment
    add(4, 8, 100.0, 0.6); // radial interosseous lateral band
    add(6, 8, 100.0, 0.6); // lumbrical into radial band
    add(5, 9, 100.0, 0.6); // ulnar interosseous lateral band
    add(8,10, 100.0, 0.6);
    add(9,10, 100.0, 0.6);
    add(10,1, 100.0, 0.8); // terminal tendon attachment
    add(2, 7,  90.0, 0.3); // extensor hood/sagittal anchor
    if (intercrossing) {
        // Published material family: 65--157 MPa bands with 0.01 mm^2
        // individual intercrossing fibres.
        add(7, 8,  90.0, 0.01, 0.95);
        add(7, 9,  90.0, 0.01, 0.95);
        add(8, 9,  90.0, 0.01, 0.95);
        add(7,10,  90.0, 0.01, 0.95);
    }
    constexpr double muscleForce = 2.9;
    fixture.loads = {
        {3, normalizedFromTo(fixture.nodes[7].position,
                             fixture.nodes[3].position, muscleForce)},
        {4, normalizedFromTo(fixture.nodes[8].position,
                             fixture.nodes[4].position, muscleForce)},
        {5, normalizedFromTo(fixture.nodes[9].position,
                             fixture.nodes[5].position, muscleForce)},
        {6, normalizedFromTo(fixture.nodes[8].position,
                             fixture.nodes[6].position, muscleForce)},
    };
    return fixture;
}

Fixture mirroredFixture(Fixture fixture) {
    for (auto& node : fixture.nodes) node.position[0] = -node.position[0];
    for (auto& load : fixture.loads) load.force[0] = -load.force[0];
    return fixture;
}

NumiHumanTensionNetworkResult solve(const Fixture& fixture) {
    NumiHumanTensionNetworkResult result;
    metalrobo::NumiHumanTensionNetworkConfig config;
    config.maximumIterations = 256u;
    config.forceTolerance = 2.0e-8;
    const auto diagnostics = metalrobo::solveNumiHumanTensionNetwork(
        fixture.nodes, fixture.elements, fixture.loads, result, config);
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(std::string("network solve failed: ") +
            metalrobo::numiHumanTensionNetworkStatusName(diagnostics.status));
    }
    return result;
}

double norm(const std::array<double, 3>& value) {
    return std::hypot(value[0], value[1], value[2]);
}

double reactionMagnitude(const NumiHumanTensionNetworkResult& result,
                         const std::size_t node) {
    return norm(result.fixedReactionForce[node]);
}

bool bitwiseEqual(const NumiHumanTensionNetworkResult& a,
                  const NumiHumanTensionNetworkResult& b) {
    return a.position.size() == b.position.size() &&
        a.elementTension.size() == b.elementTension.size() &&
        std::memcmp(a.position.data(), b.position.data(),
                    a.position.size() * sizeof(a.position[0])) == 0 &&
        std::memcmp(a.elementTension.data(), b.elementTension.data(),
                    a.elementTension.size() * sizeof(double)) == 0 &&
        std::memcmp(&a.strainEnergy, &b.strainEnergy,
                    sizeof(double)) == 0;
}

double maximumTensionDifference(const NumiHumanTensionNetworkResult& a,
                                const NumiHumanTensionNetworkResult& b) {
    if (a.elementTension.size() != b.elementTension.size()) return INFINITY;
    double maximum = 0.0;
    for (std::size_t index = 0u; index < a.elementTension.size(); ++index) {
        maximum = std::max(maximum, std::abs(
            a.elementTension[index] - b.elementTension[index]));
    }
    return maximum;
}

} // namespace

int main() {
    try {
        const Fixture fullFixture = makeFixture(true);
        const Fixture trivialFixture = makeFixture(false);
        const auto full = solve(fullFixture);
        const auto replay = solve(fullFixture);
        const auto mirrored = solve(mirroredFixture(fullFixture));
        const auto trivial = solve(trivialFixture);
        const double fullMiddle = reactionMagnitude(full, 0u);
        const double fullDistal = reactionMagnitude(full, 1u);
        const double trivialMiddle = reactionMagnitude(trivial, 0u);
        const double trivialDistal = reactionMagnitude(trivial, 1u);
        const double transferDelta = std::hypot(
            fullMiddle - trivialMiddle, fullDistal - trivialDistal);
        const double bilateralTensionError =
            maximumTensionDifference(full, mirrored);
        auto invalidElements = fullFixture.elements;
        invalidElements.front().nodeB = invalidElements.front().nodeA;
        auto rejectedPublication = full;
        const auto rejectedDiagnostics =
            metalrobo::solveNumiHumanTensionNetwork(
                fullFixture.nodes, invalidElements, fullFixture.loads,
                rejectedPublication);
        if (!bitwiseEqual(full, replay) ||
            rejectedDiagnostics.succeeded() ||
            !bitwiseEqual(full, rejectedPublication) ||
            bilateralTensionError > 1.0e-12 ||
            std::abs(fullMiddle - reactionMagnitude(mirrored, 0u)) > 1.0e-12 ||
            std::abs(fullDistal - reactionMagnitude(mirrored, 1u)) > 1.0e-12 ||
            full.maximumFreeNodeResidual > 2.0e-8 ||
            norm(full.forceClosureResidual) > 1.0e-8 ||
            norm(full.momentClosureResidual) > 1.0e-8 ||
            full.activeElementCount < 9u ||
            !(full.strainEnergy > 0.0) || !(transferDelta > 1.0e-5)) {
            std::cerr << std::setprecision(12)
                      << "replay=" << bitwiseEqual(full, replay)
                      << " rollback=" << bitwiseEqual(full, rejectedPublication)
                      << " bilateral=" << bilateralTensionError
                      << " free=" << full.maximumFreeNodeResidual
                      << " force=" << norm(full.forceClosureResidual)
                      << " moment=" << norm(full.momentClosureResidual)
                      << " active=" << full.activeElementCount
                      << " energy=" << full.strainEnergy
                      << " transfer=" << transferDelta << '\n';
            throw std::runtime_error("extensor hood qualification gate failed");
        }
        std::cout << std::setprecision(12)
                  << "numi_human_extensor_hood_reference=passed"
                  << " topology=medial_lateral_hood_intercrossing_fibres"
                  << " muscle_inputs=4"
                  << " full_nodes=" << fullFixture.nodes.size()
                  << " full_elements=" << fullFixture.elements.size()
                  << " active_elements=" << full.activeElementCount
                  << " young_modulus_min_mpa=90"
                  << " young_modulus_max_mpa=110"
                  << " band_area_min_mm2=0.3"
                  << " band_area_max_mm2=1"
                  << " intercrossing_fibre_area_mm2=0.01"
                  << " intercrossing_rest_scale=0.95"
                  << " muscle_force_each_n=2.9"
                  << " strain_energy_j=" << full.strainEnergy
                  << " max_free_residual_n="
                  << full.maximumFreeNodeResidual
                  << " force_closure_residual_n="
                  << norm(full.forceClosureResidual)
                  << " moment_closure_residual_nm="
                  << norm(full.momentClosureResidual)
                  << " full_middle_attachment_reaction_n=" << fullMiddle
                  << " full_distal_attachment_reaction_n=" << fullDistal
                  << " trivial_middle_attachment_reaction_n="
                  << trivialMiddle
                  << " trivial_distal_attachment_reaction_n="
                  << trivialDistal
                  << " force_transfer_delta_n=" << transferDelta
                  << " bilateral_max_tension_error_n="
                  << bilateralTensionError
                  << " replay=bitwise"
                  << " rollback=verified"
                  << " boundary=cpu_fp64_literature_scale_reference_not_yet_posed_source_hand_or_live_metal_transaction\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numi_human_extensor_hood_reference FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
