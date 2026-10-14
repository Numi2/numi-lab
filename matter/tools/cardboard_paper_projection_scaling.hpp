#pragma once

#include "numi/matter/matter.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace numi::cardboard {

// Convert the six strain residuals and six normalized stress residuals into
// component yield-stress units. This left-scales equations only: it changes
// neither their zero set nor energy, density, strengths, accepted history, or
// the global nonlinear tolerances. It is opt-in pending native qualification.
inline std::array<double,13> paperProjectionRowScales(
    const numi::matter::MaterialProgram& material
) {
    const auto parameter=[&](const std::string& name) {
        for(const auto& p:material.parameters)if(p.name==name)return p.defaultValue;
        throw std::invalid_argument("paper residual scaling missing parameter "+name);
    };
    const std::array<const char*,13> names{"ep11","ep22","ep33","ep23","ep13","ep12",
        "sbar11","sbar22","sbar33","sbar23","sbar13","sbar12","accumulated_multiplier"};
    if(material.internalState.size()!=13 || material.stateImplicitRoots.size()!=13)
        throw std::invalid_argument("paper residual scaling requires the exact 13-state Hill system");
    for(unsigned i=0;i<13;++i)if(material.internalState[i].name!=names[i] ||
        material.stateImplicitRoots[i]==NM_INVALID_INDEX)
        throw std::invalid_argument("paper residual scaling state order mismatch");
    const double sigma=parameter("sigma_ref");
    const std::array<double,6> yieldWeights{
        std::sqrt(parameter("hill_G")+parameter("hill_H")),
        std::sqrt(parameter("hill_F")+parameter("hill_H")),
        std::sqrt(parameter("hill_F")+parameter("hill_G")),
        std::sqrt(2*parameter("hill_L")),std::sqrt(2*parameter("hill_M")),
        std::sqrt(2*parameter("hill_N"))};
    const std::array<double,6> stiffness{parameter("c11"),parameter("c22"),parameter("c33"),
        2*parameter("g23"),2*parameter("g13"),2*parameter("g12")};
    std::array<double,13> scales{};
    for(unsigned i=0;i<6;++i) {
        scales[i]=yieldWeights[i]*stiffness[i]/sigma;
        scales[6+i]=yieldWeights[i];
    }
    scales[12]=1;
    for(double x:scales)if(!(x>0)||!std::isfinite(x))
        throw std::invalid_argument("paper residual scaling requires positive finite component scales");
    return scales;
}

inline numi::matter::MaterialProgram scalePaperProjectionResiduals(
    numi::matter::MaterialProgram material
) {
    const auto scales=paperProjectionRowScales(material);
    for(unsigned i=0;i<13;++i) {
        const auto root=material.stateImplicitRoots[i];
        const auto dimension=material.expressions.nodes.at(root).dimension;
        const auto coefficient=material.expressions.constant(scales[i]);
        numi::matter::Expr expression;
        expression.kind=numi::matter::ExprKind::multiply;
        expression.dimension=dimension;
        expression.arguments[0]=coefficient;expression.arguments[1]=root;
        material.stateImplicitRoots[i]=material.expressions.append(expression);
    }
    return material;
}
}
