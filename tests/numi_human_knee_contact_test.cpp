#include "metalrobo/NumiHumanKneeContact.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double length(const std::array<double, 3u>& value) {
    return std::sqrt(value[0u] * value[0u] + value[1u] * value[1u] +
                     value[2u] * value[2u]);
}

} // namespace

int main() {
    try {
        metalrobo::NumiHumanKneePayload payload;
        payload.regions = {
            {.name = "master",
             .kind = metalrobo::NumiHumanKneeRegionKind::cartilage,
             .firstNode = 0u, .nodeCount = 3u,
             .firstSurface = 0u, .surfaceCount = 1u},
            {.name = "slave",
             .kind = metalrobo::NumiHumanKneeRegionKind::cartilage,
             .firstNode = 3u, .nodeCount = 3u,
             .firstSurface = 1u, .surfaceCount = 1u},
        };
        payload.surfaces = {
            {.name = "master_contact", .regionIndex = 0u,
             .firstFace = 0u, .faceCount = 1u},
            {.name = "slave_contact", .regionIndex = 1u,
             .firstFace = 1u, .faceCount = 1u},
        };
        payload.faces = {{{0u, 1u, 2u}}, {{3u, 4u, 5u}}};
        for (std::uint32_t index = 0u; index < 7u; ++index) {
            payload.surfacePairs.push_back({
                .name = "pair_" + std::to_string(index),
                .masterSurface = 0u,
                .slaveSurface = 1u,
            });
        }
        payload.nodes.resize(6u);
        const std::vector<std::array<double, 3u>> reference{
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
            {0.0, 0.0, 0.001}, {1.0, 0.0, 0.001},
            {0.0, 1.0, 0.001},
        };
        const std::vector<metalrobo::NumiHumanKneeContactRegionMaterial>
            materials{
                {.regionIndex = 0u,
                 .material = {.elasticModulusPascals = 12.0e6,
                              .poissonRatio = 0.45,
                              .thicknessMeters = 0.003}},
                {.regionIndex = 1u,
                 .material = {.elasticModulusPascals = 12.0e6,
                              .poissonRatio = 0.45,
                              .thicknessMeters = 0.003}},
            };
        metalrobo::NumiHumanKneeContactModel model;
        auto diagnostics =
            metalrobo::buildNumiHumanKneeArticularContactModel(
                payload, reference, materials, model);
        require(diagnostics.succeeded() && model.pairs.size() == 7u &&
                    model.samples.size() == 21u,
                "synthetic contact model did not build");

        metalrobo::NumiHumanKneeContactResult unloaded;
        diagnostics = metalrobo::evaluateNumiHumanKneeContact(
            model, reference, 0.0, unloaded);
        require(diagnostics.succeeded() && unloaded.forceL1Newtons == 0.0 &&
                    unloaded.storedEnergyJoules == 0.0,
                "reference contact state is not preload free");

        constexpr double closure = 1.0e-4;
        metalrobo::NumiHumanKneeContactResult prescribed;
        diagnostics = metalrobo::evaluateNumiHumanKneeContact(
            model, reference, closure, prescribed);
        require(diagnostics.succeeded() && prescribed.forceL1Newtons > 0.0 &&
                    prescribed.storedEnergyJoules > 0.0 &&
                    length(prescribed.forceResidualNewtons) < 1.0e-8 &&
                    length(prescribed.momentResidualNewtonMeters) < 1.0e-8,
                "prescribed contact does not conserve force and moment");
        const double normalForce = 0.5 * prescribed.forceL1Newtons;
        require(std::abs(prescribed.storedEnergyJoules -
                         0.5 * normalForce * closure) <=
                    1.0e-12 * prescribed.storedEnergyJoules,
                "prescribed contact violates elastic work identity");

        auto compressed = reference;
        for (std::uint32_t node = 3u; node < 6u; ++node)
            compressed[node][2u] -= 0.5 * closure;
        metalrobo::NumiHumanKneeContactResult geometric;
        diagnostics = metalrobo::evaluateNumiHumanKneeContact(
            model, compressed, 0.0, geometric);
        require(diagnostics.succeeded() && geometric.forceL1Newtons > 0.0 &&
                    std::abs(geometric.forceL1Newtons /
                             prescribed.forceL1Newtons - 0.5) < 1.0e-10,
                "geometric closure did not drive the foundation law");

        auto separated = reference;
        for (std::uint32_t node = 3u; node < 6u; ++node)
            separated[node][2u] += closure;
        metalrobo::NumiHumanKneeContactResult tension;
        diagnostics = metalrobo::evaluateNumiHumanKneeContact(
            model, separated, 0.0, tension);
        require(diagnostics.succeeded() && tension.forceL1Newtons == 0.0,
                "frictionless articular law admitted tensile traction");

        payload.surfacePairs.pop_back();
        diagnostics = metalrobo::buildNumiHumanKneeArticularContactModel(
            payload, reference, materials, model);
        require(!diagnostics.succeeded() &&
                    diagnostics.status ==
                        metalrobo::NumiHumanKneeContactStatus::
                            incompleteAnatomy,
                "incomplete articular pair coverage did not fail closed");
        std::cout << "numi_human_knee_contact_test=passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numi_human_knee_contact_test=failed error=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
