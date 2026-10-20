#include "../../apps/NumiHumanImplicitSpringRows.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
void require(const bool condition, const std::string_view message) {
    if (condition) return;
    std::cerr << "human_implicit_spring_inertial_guard_check=failed: "
              << message << '\n';
    std::exit(1);
}

struct Sample {
    std::uint32_t flags = 0u;
    std::uint32_t slaveNode = 0u;
    struct { float x=0.0f, y=0.0f, z=0.0f, w=0.0f; }
        stiffnessAndNormalStrainPerPressure;
    struct { float x=0.0f, y=0.0f, z=0.0f, w=0.0f; }
        bodyLocalPointAndArea;
};

struct Node {
    struct { float x=0.0f, y=0.0f, z=0.0f, w=0.0f; } restAndFixed;
    struct { float x=0.0f, y=0.0f, z=0.0f, w=1.0f; } positionAndMass;
};

Sample spring(const std::uint32_t flags, const std::uint32_t node,
              const float stiffnessArea) {
    Sample value{};
    value.flags = flags;
    value.slaveNode = node;
    value.stiffnessAndNormalStrainPerPressure.x = stiffnessArea;
    value.bodyLocalPointAndArea.w = 1.0f;
    return value;
}

void testParser() {
    std::uint32_t count = 99u;
    require(numi::human::implicit_spring_rows::parseRequestedCount(
                nullptr, 20u, count) && count == 0u,
            "unset option must preserve zero implicit springs");
    require(numi::human::implicit_spring_rows::parseRequestedCount(
                "1", 20u, count) && count == 1u,
            "option 1 must parse");
    require(numi::human::implicit_spring_rows::parseRequestedCount(
                "20", 20u, count) && count == 20u,
            "option 20 must parse");
    require(!numi::human::implicit_spring_rows::parseRequestedCount(
                "0", 20u, count) &&
            !numi::human::implicit_spring_rows::parseRequestedCount(
                "21", 20u, count) &&
            !numi::human::implicit_spring_rows::parseRequestedCount(
                "2x", 20u, count) &&
            !numi::human::implicit_spring_rows::parseRequestedCount(
                "", 20u, count),
            "invalid option spellings must fail closed");
}

void testAdapterPrefixAndComplementBound() {
    constexpr std::uint32_t active = 1u;
    constexpr std::uint32_t elastic = 2u;
    const std::array<Sample, 6> samples{{
        spring(active, 0u, 9.0f),             // fascia glide: not a spring
        spring(active, 1u, 9.0f),             // another glide-only row
        spring(active | elastic, 2u, 0.8f),   // selected spring 1
        spring(active | elastic, 3u, 0.7f),   // selected spring 2
        spring(active | elastic, 4u, 0.2f),   // explicit complement
        spring(active | elastic, 5u, 0.1f),   // another explicit row
    }};
    std::array<Node, 6> nodes{};
    std::vector<std::uint8_t> selected;
    require(numi::human::implicit_spring_rows::selectExactFlagPrefix(
                samples, 2u, active, elastic, selected),
            "two exact elastic rows should be selectable around glide rows");
    require(selected == std::vector<std::uint8_t>({0u, 0u, 1u, 1u, 0u, 0u}),
            "selection must match adapter's first-N exact-flag prefix");

    numi::human::implicit_spring_rows::InertialBoundResult result{};
    const auto ok = numi::human::implicit_spring_rows::checkExplicitSkinRows(
        samples, 2u, selected, nodes, 1.0, 0.25, active, elastic, result);
    require(ok == numi::human::implicit_spring_rows::InertialBoundStatus::valid &&
                result.implicitSkinRows == 2u &&
                result.explicitSkinRows == 2u &&
                std::abs(result.maximumExplicitRatio - 0.2) < 1.0e-6,
            "selected stiff rows must leave the explicit guard while preserving the explicit complement");

    auto withExplicitComplement = samples;
    withExplicitComplement[4] = spring(active | elastic, 4u, 0.3f);
    require(numi::human::implicit_spring_rows::selectExactFlagPrefix(
                withExplicitComplement, 1u, active, elastic, selected),
            "one exact elastic row should be selectable");
    const auto rejected =
        numi::human::implicit_spring_rows::checkExplicitSkinRows(
            withExplicitComplement, 2u, selected, nodes, 1.0, 0.25,
            active, elastic, result);
    require(rejected ==
                numi::human::implicit_spring_rows::InertialBoundStatus::
                    explicitRowExceedsBound &&
                result.implicitSkinRows == 1u &&
                result.explicitSkinRows == 1u,
            "an unselected spring above the bound must still reject");

    require(!numi::human::implicit_spring_rows::selectExactFlagPrefix(
                samples, 5u, active, elastic, selected),
            "request beyond available exact spring rows must reject");
}

void testSelectedRowsStillRequireValidMass() {
    constexpr std::uint32_t active = 1u;
    constexpr std::uint32_t elastic = 2u;
    const std::array<Sample, 1> samples{{
        spring(active | elastic, 0u, 0.8f),
    }};
    std::array<Node, 1> nodes{};
    nodes[0].positionAndMass.w = 0.0f;
    std::vector<std::uint8_t> selected;
    require(numi::human::implicit_spring_rows::selectExactFlagPrefix(
                samples, 1u, active, elastic, selected),
            "selected row setup failed");
    numi::human::implicit_spring_rows::InertialBoundResult result{};
    require(numi::human::implicit_spring_rows::checkExplicitSkinRows(
                samples, 0u, selected, nodes, 1.0, 0.25, active, elastic,
                result) ==
                numi::human::implicit_spring_rows::InertialBoundStatus::badNode,
            "implicit selection must not conceal a massless/fixed source node");
}
} // namespace

int main() {
    testParser();
    testAdapterPrefixAndComplementBound();
    testSelectedRowsStillRequireValidMass();
    std::cout << "human_implicit_spring_inertial_guard_check=passed\n";
    return 0;
}
