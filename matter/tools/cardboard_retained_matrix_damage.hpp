#pragma once

// Explicit damage/compaction overlay for the retained Haj-Ali liner/NSSC
// matrix programs. The source matrix program and density are cloned unchanged;
// this file adds an energy-derived exponential crack-band softening branch.
// Its pure-mode energy formula has a closed-form uniaxial check, but its
// mixed-mode and rate-dependent envelopes remain uncalibrated analogues.

#include "cardboard_damage_builder.hpp"
#include "cardboard_fracture_regularization.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace numi::cardboard::retained_damage {

using numi::cardboard::FractureVec3;
using numi::cardboard::TetCrackPlane;
using numi::cardboard::TetCrackPlaneGeometry;
using numi::cardboard::computeTetCrackPlaneGeometry;
using numi::matter::Dimension;
using numi::matter::Expr;
using numi::matter::ExprKind;
using numi::matter::InternalState;
using numi::matter::MaterialProgram;
using numi::matter::Parameter;
using numi::matter::kDimensionless;
using numi::matter::kLength;
using numi::matter::kPressure;
using numi::matter::kTime;

using ExplicitSetBinding = numi::cardboard::damage::ExplicitSetBinding;
using SourceBinding = numi::cardboard::damage::SourceBinding;

inline constexpr const char* kHardeningCsv =
    "haj_ali_2009_bilinear_tension_fit_nodes.csv";
inline constexpr const char* kOverlayIdentity =
    "HajAli2009_retained_Hill_matrix_with_energy_derived_crack_band_softening_and_compaction_candidate";

struct ModeParameters {
    double strengthPa = 0.0;
    double fractureEnergyJPerM2 = 0.0;
    double rateTimeConstantSeconds = 0.0;
};

struct ExplicitDamageSet {
    std::string setId;
    std::string sha256;
    std::string evidenceClass;
    // Softening is added to the source Hill flow as component-wise accumulated
    // plastic-strain hardening. MD/CD values may be bound to the digitized
    // Haj-Ali bilinear tension-fit nodes; ZD and shear require an explicit
    // analogue/feasible-set member.
    std::array<double, 6> plasticHardeningPa{}; // 11,22,33,23,13,12
    SourceBinding hardeningBinding;
    // Four material-frame planes; normal-opening and in-plane shear channels.
    std::array<ModeParameters, 8> fractureModes{};
    ExplicitSetBinding fractureBinding;
    ExplicitSetBinding rateBinding;
    // Axis compression densification is an engineering extension. The
    // original Hill compression strengths remain in the base program.
    std::array<double, 3> densificationModulusPa{};
    std::array<double, 3> densificationOnsetStrain{};
    std::array<double, 3> densificationRateTimeConstantSeconds{};
    ExplicitSetBinding densificationBinding;
    double damageCap = 0.0;
    double smoothStrainWidth = 0.0;
};

struct SourceMaterialBinding {
    std::string materialId;
    std::string sha256;
    std::string evidenceClass;
};

struct ElementRequest {
    std::uint32_t elementIndex = 0;
    std::uint32_t sourceMaterialIndex = 0;
    std::string regionId;
    MaterialProgram sourceMatrix;
    double sourceDensityKgPerM3 = 0.0;
    SourceMaterialBinding sourceBinding;
    std::array<FractureVec3, 4> referenceVerticesM{};
    std::array<TetCrackPlane, 4> selectedMaterialPlanes{};
    ExplicitDamageSet explicitSet;
};

struct ElementMapping {
    std::uint32_t elementIndex = 0;
    std::uint32_t sourceMaterialIndex = 0;
    std::uint32_t overlayPoolIndex = 0;
    std::string regionId;
    std::string sourceMaterialId;
    std::string sourceMaterialSha256;
    std::string explicitSetId;
    std::string explicitSetSha256;
    std::string evidenceClass;
    SourceBinding hardeningBinding;
    ExplicitSetBinding fractureBinding;
    ExplicitSetBinding rateBinding;
    ExplicitSetBinding densificationBinding;
    double sourceDensityKgPerM3 = 0.0;
    double referenceVolumeM3 = 0.0;
    std::array<double, 6> plasticHardeningPa{};
    std::array<ModeParameters, 8> fractureModes{};
    std::array<double, 3> densificationModulusPa{};
    std::array<double, 3> densificationOnsetStrain{};
    std::array<double, 3> densificationRateTimeConstantSeconds{};
    double damageCap = 0.0;
    double smoothStrainWidth = 0.0;
    std::array<TetCrackPlaneGeometry, 4> planeGeometry{};
    std::array<double, 4> modeLengthM{};
    std::array<double, 8> modeOnsetStrain{};
    std::array<double, 8> modeSofteningStrain{};
};

struct ElementProgram {
    MaterialProgram program;
    double densityKgPerM3 = 0.0;
    ElementMapping mapping;
};

struct MeshProgramPool {
    std::vector<MaterialProgram> overlayPrograms;
    std::vector<double> overlayDensitiesKgPerM3;
    // Pool-local indices in original element order. Integrators append the
    // programs after source matrices and add the returned pool index.
    std::vector<std::uint32_t> overlayMaterialIndices;
    std::vector<ElementMapping> elementMappings;
};

inline bool validSha256(const std::string& value) {
    return numi::cardboard::damage::isSha256(value);
}

inline const Parameter& parameter(const MaterialProgram& material,
                                  const std::string& name) {
    const auto found = std::find_if(material.parameters.begin(), material.parameters.end(),
        [&](const Parameter& value) { return value.name == name; });
    if (found == material.parameters.end())
        throw std::invalid_argument("source matrix lacks parameter " + name);
    return *found;
}

inline void validateSourceMatrix(const ElementRequest& request) {
    const auto& material = request.sourceMatrix;
    const bool liner = material.name == "hajali2009_liner_hill_ideal";
    const bool medium = material.name == "hajali2009_medium_hill_ideal";
    if (!(liner || medium))
        throw std::invalid_argument(
            "retained-matrix overlay only accepts the loaded Haj-Ali liner or NSSC Hill matrix program");
    const std::string expectedId = material.name;
    if (request.sourceBinding.materialId != expectedId ||
        !validSha256(request.sourceBinding.sha256) ||
        request.sourceBinding.evidenceClass.empty())
        throw std::invalid_argument("source matrix provenance binding is incomplete or mismatched");
    const double authoredDensity = parameter(material, "density").defaultValue;
    if (!(std::isfinite(request.sourceDensityKgPerM3) &&
          request.sourceDensityKgPerM3 > 0.0 &&
          request.sourceDensityKgPerM3 == authoredDensity))
        throw std::invalid_argument(
            "overlay must preserve the source material's authored density exactly");
    if (material.internalState.size() != 13u ||
        material.stateImplicitRoots.size() != 13u ||
        material.stateUpdateRoots.size() != 13u ||
        material.name.find("robertsson") != std::string::npos)
        throw std::invalid_argument("source matrix is not the retained 13-state Haj-Ali Hill program");
    for (const char* name : {"c11", "c22", "c33", "c12", "c13", "c23",
                             "g12", "g13", "g23", "sigma_ref", "hill_F", "hill_G",
                             "hill_H", "hill_L", "hill_M", "hill_N"})
        (void)parameter(material, name);
}

inline void validateExplicitSet(const ExplicitDamageSet& set) {
    if (set.setId.empty() || !validSha256(set.sha256) || set.evidenceClass.empty())
        throw std::invalid_argument("explicit damage-set identity/hash/evidence class is required");
    const auto validBinding = [](const ExplicitSetBinding& b) {
        return !b.setId.empty() && validSha256(b.sha256) && !b.evidenceClass.empty();
    };
    if (set.hardeningBinding.sourceId.empty() ||
        set.hardeningBinding.datasetId.empty() ||
        !validSha256(set.hardeningBinding.datasetSha256) ||
        set.hardeningBinding.mapping.empty() ||
        !validBinding(set.fractureBinding) || !validBinding(set.rateBinding) ||
        !validBinding(set.densificationBinding))
        throw std::invalid_argument(
            "hardening, fracture, rate, and densification bindings are all required");
    if (!(std::isfinite(set.damageCap) && (set.damageCap == 0.0 || set.damageCap == 1.0) &&
          std::isfinite(set.smoothStrainWidth) && set.smoothStrainWidth > 0.0))
        throw std::invalid_argument(
            "crack-band damage cap must select an undamaged control (0) or full softening (1), with explicit smoothing width");
    for (double h : set.plasticHardeningPa)
        if (!(std::isfinite(h) && h >= 0.0))
            throw std::invalid_argument("all six component hardening values must be explicit finite nonnegative values");
    for (const auto& mode : set.fractureModes)
        if (!(std::isfinite(mode.strengthPa) && mode.strengthPa > 0.0 &&
              std::isfinite(mode.fractureEnergyJPerM2) && mode.fractureEnergyJPerM2 > 0.0 &&
              std::isfinite(mode.rateTimeConstantSeconds) && mode.rateTimeConstantSeconds > 0.0))
            throw std::invalid_argument("all eight normal/shear modes require positive strength, Gc, and rate time");
    for (std::size_t axis = 0; axis < 3u; ++axis)
        if (!(std::isfinite(set.densificationModulusPa[axis]) &&
              set.densificationModulusPa[axis] > 0.0 &&
              std::isfinite(set.densificationOnsetStrain[axis]) &&
              set.densificationOnsetStrain[axis] >= 0.0 &&
              std::isfinite(set.densificationRateTimeConstantSeconds[axis]) &&
              set.densificationRateTimeConstantSeconds[axis] > 0.0))
            throw std::invalid_argument("all three compression axes require explicit densification parameters");
}

namespace detail {

class GraphBuilder {
public:
    explicit GraphBuilder(MaterialProgram& material) : material_(material) {}

    std::uint32_t constant(double value, Dimension dimension = kDimensionless) {
        return material_.expressions.constant(value, dimension);
    }

    std::uint32_t parameter(const std::string& name, double value, Dimension dimension) {
        if (std::any_of(material_.parameters.begin(), material_.parameters.end(),
                        [&](const Parameter& p) { return p.name == name; }))
            throw std::invalid_argument("overlay parameter name collides with source material: " + name);
        const auto index = static_cast<std::uint32_t>(material_.parameters.size());
        material_.parameters.push_back({name, dimension, value, value, value, false, false});
        Expr expression;
        expression.kind = ExprKind::parameter;
        expression.dimension = dimension;
        expression.index = index;
        return material_.expressions.append(expression);
    }

    std::uint32_t variable(ExprKind kind, std::uint32_t index,
                           Dimension dimension = kDimensionless) {
        Expr expression;
        expression.kind = kind;
        expression.dimension = dimension;
        expression.index = index;
        return material_.expressions.append(expression);
    }

    std::uint32_t unary(ExprKind kind, std::uint32_t x, Dimension dimension) {
        Expr expression;
        expression.kind = kind;
        expression.dimension = dimension;
        expression.arguments[0] = x;
        return material_.expressions.append(expression);
    }

    std::uint32_t binary(ExprKind kind, std::uint32_t a, std::uint32_t b,
                         Dimension dimension) {
        Expr expression;
        expression.kind = kind;
        expression.dimension = dimension;
        expression.arguments[0] = a;
        expression.arguments[1] = b;
        return material_.expressions.append(expression);
    }

    std::uint32_t add(std::uint32_t a, std::uint32_t b, Dimension d = kDimensionless) {
        return binary(ExprKind::add, a, b, d);
    }
    std::uint32_t sub(std::uint32_t a, std::uint32_t b, Dimension d = kDimensionless) {
        return binary(ExprKind::subtract, a, b, d);
    }
    std::uint32_t mul(std::uint32_t a, std::uint32_t b, Dimension d = kDimensionless) {
        return binary(ExprKind::multiply, a, b, d);
    }
    std::uint32_t div(std::uint32_t a, std::uint32_t b, Dimension d = kDimensionless) {
        return binary(ExprKind::divide, a, b, d);
    }
    std::uint32_t scale(std::uint32_t a, double b, Dimension d = kDimensionless) {
        return mul(a, constant(b), d);
    }
    std::uint32_t sum(const std::vector<std::uint32_t>& values,
                      Dimension d = kDimensionless) {
        if (values.empty()) return constant(0.0, d);
        auto result = values.front();
        for (std::size_t i = 1; i < values.size(); ++i) result = add(result, values[i], d);
        return result;
    }
    std::uint32_t square(std::uint32_t a) { return mul(a, a); }
    std::uint32_t sqrt(std::uint32_t a) {
        return unary(ExprKind::squareRoot, a, kDimensionless);
    }
    std::uint32_t exp(std::uint32_t a) {
        return unary(ExprKind::exponential, a, kDimensionless);
    }

    std::uint32_t smoothPositive(std::uint32_t x, std::uint32_t width) {
        const auto square = this->square(x);
        const auto widthSquare = this->square(width);
        const auto radius = sqrt(add(square, widthSquare));
        return scale(add(x, radius), 0.5);
    }

    std::uint32_t addState(const std::string& name,
                           InternalState::Transfer transfer = InternalState::Transfer::maximum) {
        const auto index = static_cast<std::uint32_t>(material_.internalState.size());
        material_.internalState.push_back({name, kDimensionless, 0.0, transfer});
        material_.stateUpdateRoots.resize(material_.internalState.size(), NM_INVALID_INDEX);
        material_.stateImplicitRoots.resize(material_.internalState.size(), NM_INVALID_INDEX);
        return index;
    }

    MaterialProgram& material() { return material_; }

private:
    MaterialProgram& material_;
};

inline std::uint32_t findState(const MaterialProgram& material, const std::string& name) {
    const auto found = std::find_if(material.internalState.begin(), material.internalState.end(),
        [&](const InternalState& state) { return state.name == name; });
    if (found == material.internalState.end())
        throw std::invalid_argument("source Hill state not found: " + name);
    return static_cast<std::uint32_t>(found - material.internalState.begin());
}

inline std::uint32_t findParameterIndex(const MaterialProgram& material,
                                       const std::string& name) {
    const auto found = std::find_if(material.parameters.begin(), material.parameters.end(),
        [&](const Parameter& value) { return value.name == name; });
    if (found == material.parameters.end())
        throw std::invalid_argument("source Hill parameter not found: " + name);
    return static_cast<std::uint32_t>(found - material.parameters.begin());
}

struct Matrix6 {
    std::array<double, 36> value{};
};

inline Matrix6 stiffness(const MaterialProgram& material) {
    return Matrix6{{
        parameter(material,"c11").defaultValue, parameter(material,"c12").defaultValue,
        parameter(material,"c13").defaultValue, 0,0,0,
        parameter(material,"c12").defaultValue, parameter(material,"c22").defaultValue,
        parameter(material,"c23").defaultValue, 0,0,0,
        parameter(material,"c13").defaultValue, parameter(material,"c23").defaultValue,
        parameter(material,"c33").defaultValue, 0,0,0,
        0,0,0,parameter(material,"g23").defaultValue,0,0,
        0,0,0,0,parameter(material,"g13").defaultValue,0,
        0,0,0,0,0,parameter(material,"g12").defaultValue}};
}

inline double stiffnessQuadratic(const Matrix6& c,const std::array<double,6>& q) {
    double result=0.0;
    for (unsigned i=0;i<6u;++i)
        for (unsigned j=0;j<6u;++j)
            result+=q[i]*c.value[i*6u+j]*q[j];
    if (!(std::isfinite(result)&&result>0.0))
        throw std::invalid_argument("selected source stiffness direction is not positive");
    return result;
}

// Normal crack-band modulus for a unit Green-strain increment n tensor n,
// holding the other strain components fixed in the authored material frame.
inline double directionalNormalModulus(const MaterialProgram& material, const FractureVec3 n) {
    const auto c=stiffness(material);
    const std::array<double,6> q{
        n.x*n.x,n.y*n.y,n.z*n.z,2.0*n.y*n.z,2.0*n.x*n.z,2.0*n.x*n.y};
    return stiffnessQuadratic(c,q);
}

// Physical shear traction modulus for the crack-band shear coordinate e_nt.
// The corresponding displacement jump is 2*l*e_nt; hence the factor 1/4
// applied to the engineering-strain directional stiffness q^T C q.
inline double directionalShearModulus(const MaterialProgram& material,
                                      const FractureVec3 n) {
    FractureVec3 seed = std::abs(n.x) < 0.75 ? FractureVec3{1,0,0} : FractureVec3{0,1,0};
    FractureVec3 t{n.y*seed.z-n.z*seed.y,
                   n.z*seed.x-n.x*seed.z,
                   n.x*seed.y-n.y*seed.x};
    const double norm = std::sqrt(t.x*t.x+t.y*t.y+t.z*t.z);
    if (!(norm > 1.0e-12)) throw std::invalid_argument("plane normal has no stable tangent basis");
    t = {t.x/norm,t.y/norm,t.z/norm};
    const auto c=stiffness(material);
    const std::array<double,6> q{
        2.0*n.x*t.x,2.0*n.y*t.y,2.0*n.z*t.z,
        2.0*(n.y*t.z+n.z*t.y),2.0*(n.x*t.z+n.z*t.x),
        2.0*(n.x*t.y+n.y*t.x)};
    return stiffnessQuadratic(c,q)*0.25;
}

inline void appendParameter( GraphBuilder& g, const std::string& name, double value, Dimension dim,
                             std::unordered_map<std::string,std::uint32_t>& ids) {
    ids.emplace(name,g.parameter(name,value,dim));
}

} // namespace detail

inline ElementProgram buildElementProgram(const ElementRequest& request) {
    validateSourceMatrix(request);
    validateExplicitSet(request.explicitSet);
    if (request.regionId.empty() || request.sourceBinding.materialId.empty())
        throw std::invalid_argument("element region and retained source material identity are required");

    ElementProgram result;
    result.program = request.sourceMatrix;
    result.densityKgPerM3 = request.sourceDensityKgPerM3;
    result.mapping.elementIndex = request.elementIndex;
    result.mapping.sourceMaterialIndex = request.sourceMaterialIndex;
    result.mapping.regionId = request.regionId;
    result.mapping.sourceMaterialId = request.sourceBinding.materialId;
    result.mapping.sourceMaterialSha256 = request.sourceBinding.sha256;
    result.mapping.explicitSetId = request.explicitSet.setId;
    result.mapping.explicitSetSha256 = request.explicitSet.sha256;
    result.mapping.evidenceClass = request.explicitSet.evidenceClass;
    result.mapping.hardeningBinding = request.explicitSet.hardeningBinding;
    result.mapping.fractureBinding = request.explicitSet.fractureBinding;
    result.mapping.rateBinding = request.explicitSet.rateBinding;
    result.mapping.densificationBinding = request.explicitSet.densificationBinding;
    result.mapping.sourceDensityKgPerM3 = request.sourceDensityKgPerM3;
    result.mapping.plasticHardeningPa = request.explicitSet.plasticHardeningPa;
    result.mapping.fractureModes = request.explicitSet.fractureModes;
    result.mapping.densificationModulusPa = request.explicitSet.densificationModulusPa;
    result.mapping.densificationOnsetStrain = request.explicitSet.densificationOnsetStrain;
    result.mapping.densificationRateTimeConstantSeconds =
        request.explicitSet.densificationRateTimeConstantSeconds;
    result.mapping.damageCap = request.explicitSet.damageCap;
    result.mapping.smoothStrainWidth = request.explicitSet.smoothStrainWidth;
    for (std::size_t i=0;i<4u;++i) {
        result.mapping.planeGeometry[i] = computeTetCrackPlaneGeometry(
            request.referenceVerticesM,request.selectedMaterialPlanes[i]);
        if (!result.mapping.planeGeometry[i].hasPositiveAreaCut())
            throw std::invalid_argument("each damage mode needs a positive-area selected plane cut");
        result.mapping.modeLengthM[i] =
            result.mapping.planeGeometry[i].characteristicLengthM.value();
        if (i==0u) result.mapping.referenceVolumeM3 =
            result.mapping.planeGeometry[i].referenceVolumeM3;
        else if (result.mapping.referenceVolumeM3!=
                 result.mapping.planeGeometry[i].referenceVolumeM3)
            throw std::invalid_argument("selected plane calculations disagree on tetrahedron volume");
    }

    auto& material = result.program;
    const std::uint32_t sourceEnergy = material.energyRoot;
    if (sourceEnergy == NM_INVALID_INDEX)
        throw std::invalid_argument("retained matrix program must provide a source free-energy root");
    detail::GraphBuilder g(material);
    std::unordered_map<std::string,std::uint32_t> p;
    const auto& set=request.explicitSet;
    const Dimension fractureEnergyDimension{0,1,-2,0}; // J/m^2 == N/m
    appendParameter(g,"overlay_damage_cap",set.damageCap,kDimensionless,p);
    appendParameter(g,"overlay_smooth_strain_width",set.smoothStrainWidth,kDimensionless,p);
    for (unsigned component=0;component<6u;++component)
        appendParameter(g,"overlay_plastic_hardening_"+std::to_string(component),
                        set.plasticHardeningPa[component],kPressure,p);
    for (unsigned mode=0;mode<8u;++mode) {
        const std::string suffix="_mode"+std::to_string(mode);
        appendParameter(g,"overlay_strength_pa"+suffix,set.fractureModes[mode].strengthPa,kPressure,p);
        appendParameter(g,"overlay_gc_j_m2"+suffix,set.fractureModes[mode].fractureEnergyJPerM2,
                        fractureEnergyDimension,p);
        appendParameter(g,"overlay_rate_tau_s"+suffix,set.fractureModes[mode].rateTimeConstantSeconds,
                        kTime,p);
    }
    for (unsigned axis=0;axis<3u;++axis) {
        const std::string suffix="_axis"+std::to_string(axis);
        appendParameter(g,"overlay_densification_modulus_pa"+suffix,
                        set.densificationModulusPa[axis],kPressure,p);
        appendParameter(g,"overlay_densification_onset"+suffix,
                        set.densificationOnsetStrain[axis],kDimensionless,p);
        appendParameter(g,"overlay_densification_tau_s"+suffix,
                        set.densificationRateTimeConstantSeconds[axis],kTime,p);
    }

    const auto width=p.at("overlay_smooth_strain_width");
    const auto dt=g.variable(ExprKind::timeStep,0u,kTime);
    const auto scalar=[&](ExprKind kind,std::uint32_t index) {
        return g.variable(kind,index,kDimensionless);
    };
    const auto f=[&](unsigned row,unsigned col) {
        return g.variable(ExprKind::deformation,3u*row+col,kDimensionless);
    };
    std::array<std::uint32_t,6> ep{};
    std::array<std::uint32_t,6> epOld{};
    constexpr const char* epNames[]{"ep11","ep22","ep33","ep23","ep13","ep12"};
    for (unsigned i=0;i<6u;++i) {
        const auto index=detail::findState(material,epNames[i]);
        ep[i]=scalar(ExprKind::candidateState,index);
        epOld[i]=scalar(ExprKind::internalState,index);
    }

    // Add component-wise accumulated plastic strain and replace only the
    // source multiplier complementarity threshold with its explicit hardening
    // extension. The source stiffness and Hill flow expressions are retained.
    const auto widthSquare=g.square(width);
    std::array<std::uint32_t,6> hardeningState{};
    std::array<std::uint32_t,3> densificationState{};
    std::vector<std::uint32_t> hardeningTerms;
    for (unsigned i=0;i<6u;++i) {
        const auto state=g.addState("overlay_plastic_history_"+std::to_string(i),
                                    InternalState::Transfer::maximum);
        hardeningState[i]=state;
        const auto next=scalar(ExprKind::candidateState,state);
        const auto old=scalar(ExprKind::internalState,state);
        const auto increment=g.sub(next,old);
        const auto dep=g.sub(ep[i],epOld[i]);
        const auto smoothAbs=g.sub(g.sqrt(g.add(g.square(dep),widthSquare)),width);
        const auto residual=g.sub(increment,smoothAbs);
        material.stateUpdateRoots[state]=old;
        material.stateImplicitRoots[state]=residual;
        const auto hValue=scalar(ExprKind::candidateState,state);
        const auto hParam=p.at("overlay_plastic_hardening_"+std::to_string(i));
        const auto term=g.mul(hParam,hValue,kPressure);
        hardeningTerms.push_back(term);
    }
    for (unsigned axis=0;axis<3u;++axis) {
        const auto state=g.addState("overlay_densification_"+std::to_string(axis),
                                    InternalState::Transfer::maximum);
        densificationState[axis]=state;
        const auto compaction=g.variable(ExprKind::candidateState,state);
        const auto modulus=p.at("overlay_densification_modulus_pa_axis"+
                                std::to_string(axis));
        hardeningTerms.push_back(g.mul(modulus,compaction,kPressure));
    }
    const auto sigmaRef=scalar(ExprKind::parameter,
        detail::findParameterIndex(material,"sigma_ref"));
    const auto hardeningStress=g.sum(hardeningTerms,kPressure);
    const auto sourceUnit=g.constant(1.0);
    const auto yieldRatio=g.add(sourceUnit,g.div(hardeningStress,sigmaRef));
    const auto sIndices=std::array<std::uint32_t,6>{
        detail::findState(material,"sbar11"),detail::findState(material,"sbar22"),
        detail::findState(material,"sbar33"),detail::findState(material,"sbar23"),
        detail::findState(material,"sbar13"),detail::findState(material,"sbar12")};
    std::array<std::uint32_t,6> s{};
    for (unsigned i=0;i<6u;++i) s[i]=scalar(ExprKind::candidateState,sIndices[i]);
    const auto hillParam=[&](const char* name) {
        return scalar(ExprKind::parameter,detail::findParameterIndex(material,name));
    };
    const auto d23=g.sub(s[1],s[2]), d31=g.sub(s[2],s[0]), d12=g.sub(s[0],s[1]);
    const auto q2=g.sum({g.mul(hillParam("hill_F"),g.square(d23)),
        g.mul(hillParam("hill_G"),g.square(d31)),g.mul(hillParam("hill_H"),g.square(d12)),
        g.scale(g.mul(hillParam("hill_L"),g.square(s[3])),2.0),
        g.scale(g.mul(hillParam("hill_M"),g.square(s[4])),2.0),
        g.scale(g.mul(hillParam("hill_N"),g.square(s[5])),2.0),
        g.square(scalar(ExprKind::parameter,detail::findParameterIndex(material,"q_floor"))) });
    const auto q=g.sqrt(q2);
    const auto lambdaIndex=detail::findState(material,"accumulated_multiplier");
    const auto dl=g.sub(scalar(ExprKind::candidateState,lambdaIndex),
                        scalar(ExprKind::internalState,lambdaIndex));
    const auto yieldGap=g.sub(yieldRatio,q);
    const auto fbEpsilon=scalar(ExprKind::parameter,
        detail::findParameterIndex(material,"fb_epsilon"));
    const auto fb=g.sub(g.sub(g.sqrt(g.add(g.add(g.square(dl),g.square(yieldGap)),
                                           g.square(fbEpsilon))),dl),yieldGap);
    const auto residualScale=scalar(ExprKind::parameter,
        detail::findParameterIndex(material,"residual_scale"));
    material.stateImplicitRoots[lambdaIndex]=g.mul(residualScale,fb);

    // Green strain in the retained MD/CD/ZD material frame.
    const auto e11=g.scale(g.sub(g.sum({g.square(f(0,0)),g.square(f(1,0)),g.square(f(2,0))}),
                                g.constant(1.0)),0.5);
    const auto e22=g.scale(g.sub(g.sum({g.square(f(0,1)),g.square(f(1,1)),g.square(f(2,1))}),
                                g.constant(1.0)),0.5);
    const auto e33=g.scale(g.sub(g.sum({g.square(f(0,2)),g.square(f(1,2)),g.square(f(2,2))}),
                                g.constant(1.0)),0.5);
    const auto e23=g.scale(g.sum({g.mul(f(0,1),f(0,2)),g.mul(f(1,1),f(1,2)),
                                  g.mul(f(2,1),f(2,2))}),0.5);
    const auto e13=g.scale(g.sum({g.mul(f(0,0),f(0,2)),g.mul(f(1,0),f(1,2)),
                                  g.mul(f(2,0),f(2,2))}),0.5);
    const auto e12=g.scale(g.sum({g.mul(f(0,0),f(0,1)),g.mul(f(1,0),f(1,1)),
                                  g.mul(f(2,0),f(2,1))}),0.5);
    std::vector<std::uint32_t> survivalTerms;
    for (unsigned plane=0;plane<4u;++plane) {
        FractureVec3 n=request.selectedMaterialPlanes[plane].normal;
        const double nLength=std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
        if (!(std::isfinite(nLength)&&nLength>0.0))
            throw std::invalid_argument("material-frame damage plane normal must be nonzero");
        n={n.x/nLength,n.y/nLength,n.z/nLength};
        const auto n2=g.constant(n.x*n.x), ny2=g.constant(n.y*n.y), nz2=g.constant(n.z*n.z);
        const auto eN=g.sum({g.mul(n2,e11),g.mul(ny2,e22),g.mul(nz2,e33),
            g.scale(g.mul(g.constant(n.y*n.z),e23),2.0),
            g.scale(g.mul(g.constant(n.x*n.z),e13),2.0),
            g.scale(g.mul(g.constant(n.x*n.y),e12),2.0)});
        const std::array<std::uint32_t,3> en{
            g.sum({g.mul(e11,g.constant(n.x)),g.mul(e12,g.constant(n.y)),g.mul(e13,g.constant(n.z))}),
            g.sum({g.mul(e12,g.constant(n.x)),g.mul(e22,g.constant(n.y)),g.mul(e23,g.constant(n.z))}),
            g.sum({g.mul(e13,g.constant(n.x)),g.mul(e23,g.constant(n.y)),g.mul(e33,g.constant(n.z))})};
        std::vector<std::uint32_t> shearSquared;
        for (unsigned axis=0;axis<3u;++axis) {
            const double ni=axis==0u?n.x:(axis==1u?n.y:n.z);
            const auto tangential=g.sub(en[axis],g.mul(eN,g.constant(ni)));
            shearSquared.push_back(g.square(tangential));
        }
        const auto shearMagnitude=g.sub(g.sqrt(g.add(g.sum(shearSquared),widthSquare)),width);
        const auto normalModulus=detail::directionalNormalModulus(material,n);
        const auto shearModulus=detail::directionalShearModulus(material,n);
        for (unsigned localMode=0;localMode<2u;++localMode) {
            const unsigned mode=2u*plane+localMode;
            const auto state=g.addState("overlay_fracture_excess_"+std::to_string(mode));
            const auto next=scalar(ExprKind::candidateState,state);
            const auto old=scalar(ExprKind::internalState,state);
            const double modulus=localMode==0u?normalModulus:2.0*shearModulus;
            const auto strength=p.at("overlay_strength_pa_mode"+std::to_string(mode));
            const auto threshold=g.div(strength,g.constant(modulus,kPressure));
            const double generalizedStrength=localMode==0u?
                set.fractureModes[mode].strengthPa:2.0*set.fractureModes[mode].strengthPa;
            const double thresholdValue=set.fractureModes[mode].strengthPa/modulus;
            const double length=result.mapping.modeLengthM[plane];
            const double softeningStrain=set.fractureModes[mode].fractureEnergyJPerM2/
                (generalizedStrength*length)-0.5*thresholdValue;
            if (!(std::isfinite(thresholdValue)&&thresholdValue>0.0&&
                  std::isfinite(softeningStrain)&&softeningStrain>0.0))
                throw std::invalid_argument(
                    "selected element length/Gc/strength leaves no positive energy-derived crack-band softening interval at mode "+
                    std::to_string(mode)+" (Gc="+
                    std::to_string(set.fractureModes[mode].fractureEnergyJPerM2)+", sigma="+
                    std::to_string(set.fractureModes[mode].strengthPa)+", l="+
                    std::to_string(length)+", onset="+std::to_string(thresholdValue)+")");
            result.mapping.modeOnsetStrain[mode]=thresholdValue;
            result.mapping.modeSofteningStrain[mode]=softeningStrain;
            const auto totalDrive=localMode==0u?eN:shearMagnitude;
            const auto rate=p.at("overlay_rate_tau_s_mode"+std::to_string(mode));
            const auto rateStep=g.div(dt,rate);
            const auto rateResidual=g.sub(g.sub(next,old),g.mul(rateStep,
                g.smoothPositive(g.sub(g.sub(totalDrive,threshold),next),width)));
            material.stateUpdateRoots[state]=old;
            material.stateImplicitRoots[state]=rateResidual;

            const auto gc=p.at("overlay_gc_j_m2_mode"+std::to_string(mode));
            const auto generalizedStrengthExpr=localMode==0u?strength:g.scale(strength,2.0,kPressure);
            const auto strengthTimesLength=g.mul(generalizedStrengthExpr,
                g.constant(length,kLength),fractureEnergyDimension);
            const auto softeningWidth=g.sub(g.div(gc,strengthTimesLength),
                                             g.scale(threshold,0.5));
            const auto secantRatio=g.div(threshold,g.add(threshold,next));
            const auto survival=g.mul(secantRatio,g.exp(g.div(
                g.unary(ExprKind::negate,next,kDimensionless),softeningWidth)));
            survivalTerms.push_back(survival);
        }
    }
    const auto allModeSurvival=std::accumulate(survivalTerms.begin(),survivalTerms.end(),
        g.constant(1.0),[&](std::uint32_t a,std::uint32_t b){return g.mul(a,b);});
    // Preserve the undamaged fraction and surviving fraction as separate
    // terms.  The equivalent `1 - cap * (1 - survival)` loses most or all
    // FP32 precision when the damage cap is one and survival is near zero,
    // exactly where the crack-band softening branch is most active.
    const auto damageSurvival=g.add(
        g.sub(g.constant(1.0),p.at("overlay_damage_cap")),
        g.mul(p.at("overlay_damage_cap"),allModeSurvival));
    auto energy=g.mul(sourceEnergy,damageSurvival,kPressure);

    // Compression-only densification history is driven by accumulated source
    // plastic compression and adds explicit axis hardening to the Hill yield
    // threshold above. No extra free-energy term is added, avoiding an
    // unbalanced reversible crush force.
    for (unsigned axis=0;axis<3u;++axis) {
        const auto state=densificationState[axis];
        const auto next=scalar(ExprKind::candidateState,state);
        const auto old=scalar(ExprKind::internalState,state);
        const auto plasticCompression=g.unary(ExprKind::negate,ep[axis],kDimensionless);
        const auto parameterSuffix="_axis"+std::to_string(axis);
        const auto threshold=p.at("overlay_densification_onset"+parameterSuffix);
        const auto demand=g.smoothPositive(g.sub(plasticCompression,threshold),width);
        const auto rate=p.at("overlay_densification_tau_s"+parameterSuffix);
        const auto update=g.sub(g.sub(next,old),g.mul(g.div(dt,rate),
            g.smoothPositive(g.sub(demand,next),width)));
        material.stateUpdateRoots[state]=old;
        material.stateImplicitRoots[state]=update;
    }
    material.energyRoot=energy;
    material.name += "_explicit_retained_damage_"+set.sha256.substr(0,12);
    return result;
}

inline MeshProgramPool buildMeshProgramPool(const std::vector<ElementRequest>& requests) {
    if (requests.empty()) throw std::invalid_argument("damage overlay pool needs at least one element");
    MeshProgramPool pool;
    std::unordered_map<std::uint64_t,std::vector<std::uint32_t>> bySignature;
    std::unordered_map<std::uint32_t,bool> seenElement;
    bySignature.reserve(requests.size());
    seenElement.reserve(requests.size());
    for (const auto& request:requests) {
        if (!seenElement.emplace(request.elementIndex,true).second)
            throw std::invalid_argument("duplicate tetrahedron identity in retained-matrix damage pool");
        auto built=buildElementProgram(request);
        const auto signature=numi::cardboard::damage::nativeProgramSignature(built.program);
        std::uint32_t poolIndex=NM_INVALID_INDEX;
        for (const auto candidate:bySignature[signature]) {
            if (numi::cardboard::damage::sameNativeProgram(pool.overlayPrograms.at(candidate),built.program)) {
                poolIndex=candidate; break;
            }
        }
        if (poolIndex==NM_INVALID_INDEX) {
            poolIndex=static_cast<std::uint32_t>(pool.overlayPrograms.size());
            pool.overlayPrograms.push_back(std::move(built.program));
            pool.overlayDensitiesKgPerM3.push_back(built.densityKgPerM3);
            bySignature[signature].push_back(poolIndex);
        }
        built.mapping.overlayPoolIndex=poolIndex;
        pool.overlayMaterialIndices.push_back(poolIndex);
        pool.elementMappings.push_back(std::move(built.mapping));
    }
    return pool;
}

} // namespace numi::cardboard::retained_damage

