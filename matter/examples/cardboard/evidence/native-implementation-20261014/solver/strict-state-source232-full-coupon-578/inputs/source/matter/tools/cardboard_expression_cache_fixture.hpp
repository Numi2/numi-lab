#pragma once

#include "numi/matter/matter.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace numi_cardboard_expression_cache {
using namespace numi::matter;

inline std::uint32_t node(
    ExpressionGraph& graph,
    const ExprKind kind,
    const std::uint32_t a = NM_INVALID_INDEX,
    const std::uint32_t b = NM_INVALID_INDEX,
    const std::uint32_t c = NM_INVALID_INDEX,
    const Dimension dimension = {},
    const std::uint32_t index = 0u
) {
    Expr expr;
    expr.kind = kind;
    expr.arguments = {a, b, c, NM_INVALID_INDEX, NM_INVALID_INDEX};
    expr.dimension = dimension;
    expr.index = index;
    return graph.append(expr);
}

inline MaterialProgram makeMaterial(const bool cacheExpressions,
                                    const bool invalidInactiveBranch = false) {
    MaterialProgram material;
    material.name = cacheExpressions
        ? "synthetic_cache_select_shared_dag_cached"
        : "synthetic_cache_select_shared_dag_uncached";
    material.cacheExpressions = cacheExpressions;
    material.supportedRepresentations = {Representation::fem};
    material.parameters = {
        {"density", kDensity, 1000.0, 1.0, 10000.0, false, false},
        {"k_positive", kPressure, 4.0e6, 1.0, 1.0e9, false, false},
        {"k_negative", kPressure, 7.0e6, 1.0, 1.0e9, false, false},
        {"shared_scale", kPressure, 2.5e5, 1.0, 1.0e9, false, false},
    };
    auto& graph = material.expressions;
    const auto constant = [&](const double value, const Dimension d = {}) {
        return graph.constant(value, d);
    };
    const auto parameter = [&](const std::uint32_t i, const Dimension d) {
        return node(graph, ExprKind::parameter, NM_INVALID_INDEX,
            NM_INVALID_INDEX, NM_INVALID_INDEX, d, i);
    };
    const auto deformation = [&](const std::uint32_t i) {
        return node(graph, ExprKind::deformation, NM_INVALID_INDEX,
            NM_INVALID_INDEX, NM_INVALID_INDEX, {}, i);
    };
    const auto binary = [&](const ExprKind kind, const std::uint32_t a,
                            const std::uint32_t b, const Dimension d) {
        return node(graph, kind, a, b, NM_INVALID_INDEX, d);
    };

    const auto one = constant(1.0);
    const auto half = constant(0.5);
    const auto x = binary(ExprKind::subtract, deformation(0), one, {});
    const auto x2 = binary(ExprKind::multiply, x, x, {});
    const auto positiveEnergy = binary(ExprKind::multiply,
        binary(ExprKind::multiply, half, parameter(1, kPressure), kPressure),
        x2, kPressure);
    // The opt-in invalid arm is a deliberate evaluator fixture: for x > 0,
    // the selected positive energy is finite while sqrt(-x) is not. It does
    // not represent a constitutive model.
    const auto negativeEnergy = [&]() {
        if (!invalidInactiveBranch) {
            return binary(ExprKind::multiply,
                binary(ExprKind::multiply, half, parameter(2, kPressure), kPressure),
                x2, kPressure);
        }
        const auto negativeX = node(graph, ExprKind::negate, x,
            NM_INVALID_INDEX, NM_INVALID_INDEX, {});
        const auto sqrtNegativeX = node(graph, ExprKind::squareRoot, negativeX,
            NM_INVALID_INDEX, NM_INVALID_INDEX, {});
        return binary(ExprKind::multiply, parameter(2, kPressure),
            sqrtNegativeX, kPressure);
    }();
    const auto selected = node(graph, ExprKind::selectNonnegative,
        x, positiveEnergy, negativeEnergy, kPressure);

    // Two shared DAG branches intentionally have disjoint live ranges. The
    // compiler must reuse the freed cache slot without changing the response.
    const auto shifted01 = binary(ExprKind::add, deformation(1), constant(0.2), {});
    const auto shared01 = binary(ExprKind::multiply, shifted01, shifted01, {});
    const auto scaled01 = binary(ExprKind::multiply,
        parameter(3, kPressure), shared01, kPressure);
    const auto shifted12 = binary(ExprKind::subtract, deformation(5), constant(0.3), {});
    const auto shared12 = binary(ExprKind::multiply, shifted12, shifted12, {});
    const auto scaled12 = binary(ExprKind::multiply,
        parameter(3, kPressure), shared12, kPressure);
    const auto sharedSum = binary(ExprKind::add,
        binary(ExprKind::add, scaled01, scaled01, kPressure),
        binary(ExprKind::add, scaled12, scaled12, kPressure), kPressure);
    material.energyRoot = binary(ExprKind::add, selected, sharedSum, kPressure);
    material.validityRoot = constant(1.0);
    material.minimumDeterminant = 0.05;
    material.maximumDeterminant = 20.0;
    material.maximumStress = 1.0e12;
    material.maximumEnergyDensity = 1.0e12;
    return material;
}

inline WorldSource makeWorld(const MaterialProgram& material) {
    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0e-4;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials = {material};
    ObjectSource object;
    object.name = "synthetic_cache_select_single_tet";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.characteristicLength = 0.01;
    object.femNodes = {{{0,0,0}}, {{0.01,0,0}}, {{0,0.01,0}}, {{0,0,0.01}}};
    object.tetrahedra = {{{0,1,2,3}}};
    source.objects = {object};
    return source;
}

inline MaterialProgram makeState32(const InternalState::Transfer highTransfer) {
    MaterialProgram material = makeMaterial(false);
    material.name = highTransfer == InternalState::Transfer::sum
        ? "synthetic_abi41_state31_sum"
        : "synthetic_abi41_state31_maximum";
    material.internalState.reserve(NM_MAX_MATERIAL_STATE);
    for (std::uint32_t i = 0; i < NM_MAX_MATERIAL_STATE; ++i) {
        InternalState state;
        state.name = "state_" + std::to_string(i);
        state.dimension = {};
        state.initialValue = 0.001 * static_cast<double>(i);
        state.transfer = i == NM_MAX_MATERIAL_STATE - 1u
            ? highTransfer : InternalState::Transfer::average;
        material.internalState.push_back(std::move(state));
    }
    return material;
}
} // namespace numi_cardboard_expression_cache
