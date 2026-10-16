#include "cardboard_retained_matrix_damage.hpp"
#include "cardboard_retained_damage_fp64_oracle.hpp"
#include "numi/matter/detail.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;
using namespace numi::cardboard;
using namespace numi::cardboard::retained_damage;

using Matrix9 = std::array<double,9>;
using LocalState = std::vector<double>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string number(const double value) {
    std::ostringstream out;
    out<<std::setprecision(17)<<value;
    return out.str();
}

double evaluate(const ScalarBytecode& code,const MaterialProgram& material,
                const Matrix9& f,const Matrix9& df,const LocalState& accepted,
                const LocalState& candidate,const double dt) {
    std::vector<double> stack;
    const auto pop=[&]() {
        require(!stack.empty(),"constitutive bytecode stack underflow");
        const auto value=stack.back(); stack.pop_back(); return value;
    };
    for(const auto& instruction:code.instructions) {
        double value=0.0;
        switch(instruction.opcode) {
        case NM_EXPR_CONSTANT:value=instruction.immediate.x;break;
        case NM_EXPR_PARAMETER:value=static_cast<float>(material.parameters.at(instruction.index).defaultValue);break;
        case NM_EXPR_F:value=f.at(instruction.index);break;
        case NM_EXPR_DF:case NM_EXPR_RATE:value=df.at(instruction.index);break;
        case NM_EXPR_STATE:value=accepted.at(instruction.index);break;
        case NM_EXPR_NEXT_STATE:value=candidate.at(instruction.index);break;
        case NM_EXPR_DT:value=dt;break;
        case NM_EXPR_TEMPERATURE:value=293.15;break;
        case NM_EXPR_ADD:{const double b=pop();value=pop()+b;break;}
        case NM_EXPR_SUBTRACT:{const double b=pop();value=pop()-b;break;}
        case NM_EXPR_MULTIPLY:{const double b=pop();value=pop()*b;break;}
        case NM_EXPR_DIVIDE:{const double b=pop();const double a=pop();require(std::isfinite(b)&&b!=0.0,"constitutive division by zero or nonfinite value");value=a/b;break;}
        case NM_EXPR_NEGATE:value=-pop();break;
        case NM_EXPR_SQRT:value=std::sqrt(pop());break;
        case NM_EXPR_ABS:value=std::abs(pop());break;
        case NM_EXPR_LOG:value=std::log(pop());break;
        case NM_EXPR_EXP:value=std::exp(pop());break;
        case NM_EXPR_POW_INTEGER:{const double x=pop();value=std::pow(x,instruction.integer);break;}
        case NM_EXPR_MIN:{const double b=pop();value=std::min(pop(),b);break;}
        case NM_EXPR_MAX:{const double b=pop();value=std::max(pop(),b);break;}
        case NM_EXPR_CLAMP:{const double upper=pop();const double lower=pop();value=std::clamp(pop(),lower,upper);break;}
        case NM_EXPR_SELECT_NONNEGATIVE:{const double negative=pop();const double positive=pop();const double condition=pop();value=condition>=0.0?positive:negative;break;}
        case NM_EXPR_EXPM1_MINUS_X:{const double x=pop();value=std::expm1(x)-x;break;}
        default:throw std::runtime_error("unsupported expression opcode in retained-matrix CPU verifier: "+std::to_string(instruction.opcode));
        }
        require(std::isfinite(value),"nonfinite constitutive bytecode result");
        stack.push_back(value);
    }
    require(stack.size()==1u,"constitutive bytecode did not reduce to one scalar");
    return stack.front();
}

double evaluateEnergy(const MaterialProgram& material,const Matrix9& f,
                      const LocalState& accepted,const LocalState& candidate,
                      const double dt) {
    std::vector<double> cache(material.expressions.nodes.size(),
                              std::numeric_limits<double>::quiet_NaN());
    std::function<double(std::uint32_t)> visit=[&](const std::uint32_t index)->double {
        const auto& node=material.expressions.nodes.at(index);
        if(std::isfinite(cache[index])) return cache[index];
        const auto arg=[&](unsigned i){return visit(node.arguments.at(i));};
        double value=0.0;
        switch(node.kind) {
        case ExprKind::constant:value=static_cast<float>(node.constant);break;
        case ExprKind::parameter:value=static_cast<float>(material.parameters.at(node.index).defaultValue);break;
        case ExprKind::deformation:value=f.at(node.index);break;
        case ExprKind::deformationDirection:case ExprKind::deformationRate:value=0.0;break;
        case ExprKind::internalState:value=accepted.at(node.index);break;
        case ExprKind::candidateState:value=candidate.at(node.index);break;
        case ExprKind::timeStep:value=dt;break;
        case ExprKind::temperature:value=293.15;break;
        case ExprKind::add:value=arg(0)+arg(1);break;
        case ExprKind::subtract:value=arg(0)-arg(1);break;
        case ExprKind::multiply:value=arg(0)*arg(1);break;
        case ExprKind::divide:value=arg(0)/arg(1);break;
        case ExprKind::negate:value=-arg(0);break;
        case ExprKind::logarithm:value=std::log(arg(0));break;
        case ExprKind::exponential:value=std::exp(arg(0));break;
        case ExprKind::squareRoot:value=std::sqrt(arg(0));break;
        case ExprKind::absolute:value=std::abs(arg(0));break;
        case ExprKind::minimum:value=std::min(arg(0),arg(1));break;
        case ExprKind::maximum:value=std::max(arg(0),arg(1));break;
        case ExprKind::integerPower:value=std::pow(arg(0),node.integer);break;
        case ExprKind::clamp:value=std::clamp(arg(0),arg(1),arg(2));break;
        case ExprKind::expm1MinusArgument:{const double x=arg(0);value=std::expm1(x)-x;break;}
        case ExprKind::selectNonnegative:value=arg(0)>=0.0?arg(1):arg(2);break;
        case ExprKind::fiberExpLinear:throw std::runtime_error("unexpected fiber opcode in paper source material");
        }
        require(std::isfinite(value),"nonfinite source energy expression");
        cache[index]=value;
        return value;
    };
    return visit(material.energyRoot);
}

bool solveDense(std::vector<double> matrix,std::vector<double>& rhs,const unsigned n) {
    for(unsigned col=0;col<n;++col) {
        unsigned pivot=col;
        for(unsigned row=col+1;row<n;++row)
            if(std::abs(matrix[row*n+col])>std::abs(matrix[pivot*n+col])) pivot=row;
        const double magnitude=std::abs(matrix[pivot*n+col]);
        if(!(std::isfinite(magnitude)&&magnitude>1.0e-16)) return false;
        if(pivot!=col) {
            for(unsigned j=col;j<n;++j) std::swap(matrix[col*n+j],matrix[pivot*n+j]);
            std::swap(rhs[col],rhs[pivot]);
        }
        for(unsigned row=col+1;row<n;++row) {
            const double factor=matrix[row*n+col]/matrix[col*n+col];
            for(unsigned j=col;j<n;++j) matrix[row*n+j]-=factor*matrix[col*n+j];
            rhs[row]-=factor*rhs[col];
        }
    }
    for(int row=static_cast<int>(n)-1;row>=0;--row) {
        double value=rhs[static_cast<unsigned>(row)];
        for(unsigned col=static_cast<unsigned>(row)+1u;col<n;++col)
            value-=matrix[static_cast<unsigned>(row)*n+col]*rhs[col];
        rhs[static_cast<unsigned>(row)]=value/matrix[static_cast<unsigned>(row)*n+static_cast<unsigned>(row)];
        if(!std::isfinite(rhs[static_cast<unsigned>(row)])) return false;
    }
    return true;
}

struct LocalProjection { LocalState state; double residual=0.0; unsigned iterations=0u; };
double relativeError(const std::vector<double>& a,const std::vector<double>& b);

LocalProjection project(const ConstitutiveProgram& program,const Matrix9& f,
                        const LocalState& accepted,const double dt,
                        const double tolerance=2.0e-10) {
    const unsigned n=static_cast<unsigned>(accepted.size());
    require(program.implicitResiduals.size()==n && program.implicitJacobians.size()==n*n,
            "compiled overlay implicit system has the wrong size");
    const Matrix9 zero{};
    LocalState candidate=accepted;
    for(unsigned i=0;i<n;++i)
        candidate[i]=evaluate(program.stateUpdates.at(i),program.material,f,zero,accepted,candidate,dt);
    const auto residuals=[&](const LocalState& state) {
        std::vector<double> result(n);
        for(unsigned row=0;row<n;++row)
            result[row]=evaluate(program.implicitResiduals[row],program.material,f,zero,accepted,state,dt);
        return result;
    };
    const auto norm=[](const std::vector<double>& residual) {
        double value=0.0;for(double x:residual)value=std::max(value,std::abs(x));return value;
    };
    double residualNorm=norm(residuals(candidate));
    unsigned iteration=0u;
    for(;iteration<80u && residualNorm>tolerance;++iteration) {
        const auto r=residuals(candidate);
        std::vector<double> jacobian(n*n);
        for(unsigned row=0;row<n;++row) for(unsigned col=0;col<n;++col)
            jacobian[row*n+col]=evaluate(program.implicitJacobians[row*n+col],
                program.material,f,zero,accepted,candidate,dt);
        std::vector<double> delta(n);
        for(unsigned i=0;i<n;++i) delta[i]=-r[i];
        require(solveDense(std::move(jacobian),delta,n),"FP64 local overlay Newton Jacobian is singular");
        bool acceptedStep=false;
        for(double alpha=1.0;alpha>=1.0/4096.0;alpha*=0.5) {
            LocalState trial=candidate;
            for(unsigned i=0;i<n;++i) trial[i]+=alpha*delta[i];
            const double trialNorm=norm(residuals(trial));
            if(std::isfinite(trialNorm) && trialNorm<residualNorm) {
                candidate=std::move(trial);residualNorm=trialNorm;acceptedStep=true;break;
            }
        }
        require(acceptedStep,"FP64 local overlay Newton line search failed");
    }
    require(residualNorm<=tolerance,"FP64 local overlay projection did not converge");
    return {std::move(candidate),residualNorm,iteration};
}

std::array<double,9> stress(const ConstitutiveProgram& program,const Matrix9& f,
                            const LocalState& accepted,const LocalState& candidate,
                            const double dt) {
    std::array<double,9> result{};
    const Matrix9 zero{};
    for(unsigned i=0;i<9u;++i)
        result[i]=evaluate(program.stress[i],program.material,f,zero,accepted,candidate,dt);
    return result;
}

std::vector<double> consistentTangentAction(const ConstitutiveProgram& program,
    const Matrix9& f,const LocalState& accepted,const LocalState& candidate,
    const Matrix9& direction,const double dt) {
    const unsigned n=static_cast<unsigned>(accepted.size());
    const Matrix9 zero{};
    std::vector<double> jacobian(n*n),rhs(n),result(9u);
    for(unsigned row=0;row<n;++row) {
        for(unsigned col=0;col<n;++col)
            jacobian[row*n+col]=evaluate(program.implicitJacobians[row*n+col],
                program.material,f,zero,accepted,candidate,dt);
        rhs[row]=-evaluate(program.implicitDeformationDirections[row],
            program.material,f,direction,accepted,candidate,dt);
    }
    require(solveDense(std::move(jacobian),rhs,n),"consistent tangent implicit Jacobian is singular");
    for(unsigned component=0;component<9u;++component) {
        double value=evaluate(program.tangentVector[component],program.material,
            f,direction,accepted,candidate,dt);
        for(unsigned stateIndex=0;stateIndex<n;++stateIndex)
            value+=evaluate(program.stressStateDerivatives[component*n+stateIndex],
                program.material,f,zero,accepted,candidate,dt)*rhs[stateIndex];
        result[component]=value;
    }
    return result;
}

double reprojectedTangentError(const ConstitutiveProgram& program,const Matrix9& f,
    const LocalState& accepted,const LocalState& candidate,const Matrix9& direction,
    const double dt,const double h,const double projectionTolerance,
    double* plusResidual=nullptr,double* minusResidual=nullptr) {
    (void)candidate;
    const auto base=project(program,f,accepted,dt,projectionTolerance);
    Matrix9 plusF=f,minusF=f;
    for(unsigned i=0;i<9u;++i) {
        plusF[i]+=h*direction[i];
        minusF[i]-=h*direction[i];
    }
    const auto plus=project(program,plusF,accepted,dt,projectionTolerance);
    const auto minus=project(program,minusF,accepted,dt,projectionTolerance);
    if(plusResidual) *plusResidual=plus.residual;
    if(minusResidual) *minusResidual=minus.residual;
    const auto plusStress=stress(program,plusF,accepted,plus.state,dt);
    const auto minusStress=stress(program,minusF,accepted,minus.state,dt);
    std::vector<double> finiteDifference(9u);
    for(unsigned i=0;i<9u;++i)
        finiteDifference[i]=(plusStress[i]-minusStress[i])/(2.0*h);
    const auto tangent=consistentTangentAction(program,f,accepted,base.state,direction,dt);
    return relativeError(tangent,finiteDifference);
}

std::uint32_t stateIndex(const MaterialProgram& material,const std::string& name) {
    for(std::uint32_t i=0;i<material.internalState.size();++i)
        if(material.internalState[i].name==name) return i;
    throw std::runtime_error("test cannot locate generated state " + name);
}

struct WorkRecord {
    double externalWorkJ=0.0;
    double finalEnergyJ=0.0;
    LocalState finalState;
    std::vector<double> plasticStateAtMaximum;
    LocalState stateAtMaximum;
    std::vector<LocalState> trajectory;
    Matrix9 maximumF{};
    LocalState acceptedAtMaximum;
    LocalState candidateAtMaximum;
};

WorkRecord tensionHistory(const ConstitutiveProgram& program,
                         const double density,
                         const std::array<TetCrackPlaneGeometry,4>& planeGeometry,
                         const double maximumStretch,
                         const bool recordMaximumState,
                         const double maximumShear=0.0,
                         const unsigned loadingSteps=60u) {
    (void)density;
    const double dt=0.002;
    const auto n=static_cast<unsigned>(program.material.internalState.size());
    LocalState accepted(n);
    for(unsigned i=0;i<n;++i) accepted[i]=program.material.internalState[i].initialValue;
    Matrix9 f{1,0,0,0,1,0,0,0,1};
    const auto initialAccepted=accepted;
    double previousEnergy=evaluateEnergy(program.material,f,accepted,accepted,dt)*
                          planeGeometry[0].referenceVolumeM3;
    auto previousStress=stress(program,f,accepted,accepted,dt);
    WorkRecord record;
    std::vector<double> levels;
    levels.reserve(2u*loadingSteps);
    for(unsigned i=1u;i<=loadingSteps;++i)
        levels.push_back(1.0+(maximumStretch-1.0)*static_cast<double>(i)/loadingSteps);
    for(int i=static_cast<int>(loadingSteps)-1;i>=0;--i)
        levels.push_back(1.0+(maximumStretch-1.0)*static_cast<double>(i)/loadingSteps);
    for(unsigned step=0;step<levels.size();++step) {
        const double shearFraction=step<loadingSteps?
            static_cast<double>(step+1u)/loadingSteps:
            static_cast<double>(2u*loadingSteps-1u-step)/loadingSteps;
        Matrix9 nextF{levels[step],maximumShear*shearFraction,0,0,1,0,0,0,1};
        const auto projected=project(program,nextF,accepted,dt);
        const auto nextStress=stress(program,nextF,accepted,projected.state,dt);
        const double nextEnergy=evaluateEnergy(program.material,nextF,accepted,projected.state,dt)*
                                planeGeometry[0].referenceVolumeM3;
        double increment=0.0;
        for(unsigned component=0;component<9u;++component)
            increment+=0.5*(previousStress[component]+nextStress[component])*
                (nextF[component]-f[component]);
        record.externalWorkJ+=increment*planeGeometry[0].referenceVolumeM3;
        if(recordMaximumState && step==loadingSteps-1u) {
            record.plasticStateAtMaximum=projected.state;
            record.maximumF=nextF;
            record.acceptedAtMaximum=accepted;
            record.candidateAtMaximum=projected.state;
            record.stateAtMaximum=projected.state;
        }
        record.trajectory.push_back(projected.state);
        accepted=projected.state;
        f=nextF;
        previousStress=nextStress;
        previousEnergy=nextEnergy;
    }
    record.finalState=accepted;
    record.finalEnergyJ=previousEnergy;
    (void)initialAccepted;
    return record;
}

WorkRecord compressionHistory(const ConstitutiveProgram& program,
                              const double referenceVolumeM3,
                              const double minimumStretch) {
    constexpr double dt=0.002;
    const auto n=static_cast<unsigned>(program.material.internalState.size());
    LocalState accepted(n);
    for(unsigned i=0;i<n;++i) accepted[i]=program.material.internalState[i].initialValue;
    Matrix9 f{1,0,0,0,1,0,0,0,1};
    double previousEnergy=evaluateEnergy(program.material,f,accepted,accepted,dt)*referenceVolumeM3;
    auto previousStress=stress(program,f,accepted,accepted,dt);
    WorkRecord record;
    std::vector<double> levels;
    levels.reserve(31u);
    for(unsigned i=1u;i<=15u;++i)
        levels.push_back(1.0+(minimumStretch-1.0)*static_cast<double>(i)/15.0);
    for(int i=14;i>=0;--i)
        levels.push_back(1.0+(minimumStretch-1.0)*static_cast<double>(i)/15.0);
    for(unsigned step=0;step<levels.size();++step) {
        Matrix9 nextF{levels[step],0,0,0,1,0,0,0,1};
        const auto projected=project(program,nextF,accepted,dt);
        const auto nextStress=stress(program,nextF,accepted,projected.state,dt);
        const double nextEnergy=evaluateEnergy(program.material,nextF,accepted,projected.state,dt)*
                                referenceVolumeM3;
        record.externalWorkJ+=0.5*(previousStress[0]+nextStress[0])*
            (nextF[0]-f[0])*referenceVolumeM3;
        if(step==14u) {
            record.maximumF=nextF;
            record.acceptedAtMaximum=accepted;
            record.candidateAtMaximum=projected.state;
            record.stateAtMaximum=projected.state;
        }
        record.trajectory.push_back(projected.state);
        accepted=projected.state;
        f=nextF;
        previousStress=nextStress;
        previousEnergy=nextEnergy;
    }
    record.finalState=accepted;
    record.finalEnergyJ=previousEnergy;
    return record;
}

MaterialProgram parse(const std::filesystem::path& path) {
    const auto result = parseMatterFile(path);
    if (!result.succeeded()) throw std::runtime_error("material parse failed: " + path.string());
    return result.material;
}

ExplicitDamageSet syntheticSet() {
    ExplicitDamageSet set;
    set.setId = "retained_matrix_overlay_cpu_fixture_v1";
    set.sha256 = std::string(64, 'a');
    set.evidenceClass = "synthetic_test_only_not_calibration";
    set.plasticHardeningPa = {1.3e8, 2.4e7, 0.0, 0.0, 0.0, 0.0};
    set.hardeningBinding = {
        "haj_ali_2009_bilinear_tension_fit_nodes",
        "haj_ali_2009_bilinear_tension_fit_nodes",
        "f3f35358ca3bef1719b1ec62d52939951a6cdf5830d320bbefaae4087344727f",
        "synthetic CPU exercise; production values must be loaded from explicit data"};
    for (auto& mode : set.fractureModes)
        mode = {8.0e6, 1.0e6, 0.04};
    set.fractureBinding = {"synthetic_Gc_member",std::string(64,'b'),
                           "synthetic_test_only_not_calibration"};
    set.rateBinding = {"synthetic_rate_member",std::string(64,'c'),
                       "synthetic_test_only_not_calibration"};
    set.densificationModulusPa = {4.0e7, 3.0e7, 2.0e7};
    set.densificationOnsetStrain = {0.08, 0.10, 0.12};
    set.densificationRateTimeConstantSeconds = {0.04, 0.06, 0.08};
    set.densificationBinding = {"synthetic_densification_member",std::string(64,'d'),
                                "synthetic_test_only_not_calibration"};
    set.damageCap = 1.0;
    set.smoothStrainWidth = 1.0e-7;
    return set;
}

ElementRequest request(const MaterialProgram& source, const std::string& region,
                       std::uint32_t element, const FractureVec3 shift,
                       const double side) {
    ElementRequest result;
    result.elementIndex = element;
    result.sourceMaterialIndex = region == "liner" ? 0u : 1u;
    result.regionId = region;
    result.sourceMatrix = source;
    result.sourceDensityKgPerM3 = 0.0;
    for (const auto& parameter : source.parameters)
        if (parameter.name == "density") result.sourceDensityKgPerM3 = parameter.defaultValue;
    result.sourceBinding = {source.name,std::string(64,'e'),
                            "retained_source_asset_hash_test_placeholder"};
    const std::array<FractureVec3,4> vertices{{
        shift,
        {shift.x+side,shift.y,shift.z},
        {shift.x,shift.y+side,shift.z},
        {shift.x,shift.y,shift.z+side}}};
    result.referenceVerticesM = vertices;
    result.selectedMaterialPlanes = {{
        {{1,0,0},FractureNormalFrame::Reference,0.0,{},1.0e-12},
        {{0,1,0},FractureNormalFrame::Reference,0.0,{},1.0e-12},
        {{0,0,1},FractureNormalFrame::Reference,0.0,{},1.0e-12},
        {{1,1,1},FractureNormalFrame::Reference,0.0,{},1.0e-12}}};
    result.explicitSet = syntheticSet();
    return result;
}

FractureVec3 quarterTurnZ(const FractureVec3& value) {
    return {-value.y, value.x, value.z};
}

void checkComposition(const std::filesystem::path& materials) {
    const auto source = parse(materials / "hajali2009_liner_hill_ideal.nmatter");
    auto input = request(source,"liner",7u,{0,0,0},0.001);
    const auto built = buildElementProgram(input);
    require(built.program.name.find(source.name) == 0u,
            "overlay did not retain the source material identity");
    require(built.program.internalState.size() == 30u,
            "overlay must stay within the native 32-state material limit");
    require(built.densityKgPerM3 == input.sourceDensityKgPerM3,
            "overlay changed the regional source density");
    for (const std::string name : {"c11","c22","c33","c12","c13","c23",
                                   "g12","g13","g23","sigma_ref"}) {
        auto value = [&](const MaterialProgram& m) {
            for (const auto& p : m.parameters) if (p.name==name) return p.defaultValue;
            throw std::runtime_error("missing retained stiffness/yield parameter " + name);
        };
        require(value(built.program)==value(source),
                "overlay changed retained source matrix parameter " + name);
    }
    require(built.program.name.find("Robertsson") == std::string::npos &&
                built.program.name.find("robertsson") == std::string::npos,
            "shipping matrix overlay must not use Robertsson elasticity/density");
    require(built.mapping.sourceMaterialIndex==0u &&
                built.mapping.explicitSetSha256==input.explicitSet.sha256 &&
                built.mapping.modeLengthM[0]>0.0 &&
                built.mapping.fractureBinding.sha256==input.explicitSet.fractureBinding.sha256 &&
                built.mapping.rateBinding.sha256==input.explicitSet.rateBinding.sha256 &&
                built.mapping.densificationBinding.sha256==input.explicitSet.densificationBinding.sha256 &&
                built.mapping.fractureModes[0].fractureEnergyJPerM2==
                    input.explicitSet.fractureModes[0].fractureEnergyJPerM2 &&
                built.mapping.modeOnsetStrain[0]>0.0 &&
                built.mapping.modeSofteningStrain[0]>0.0,
            "per-element mapping lost source/set provenance, explicit laws, or selected-plane length");

    const auto compiled = numi::matter::detail::compileConstitutive(
        built.program,NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"retained matrix overlay failed constitutive compile");
    require(compiled.program.material.internalState.size()==30u &&
                compiled.program.implicitResiduals.size()==30u,
            "generated material did not expose all implicit history equations to AD");
}

void checkExactPool(const std::filesystem::path& materials) {
    const auto source = parse(materials / "hajali2009_medium_hill_ideal.nmatter");
    auto a=request(source,"NSSC",10u,{0,0,0},0.001);
    auto b=request(source,"NSSC",11u,{0.02,0,0},0.001);
    const auto pooled=buildMeshProgramPool({a,b});
    require(pooled.overlayPrograms.size()==1u &&
                pooled.overlayDensitiesKgPerM3.size()==1u &&
                pooled.overlayMaterialIndices==std::vector<std::uint32_t>{0u,0u},
            "translated, bit-identical overlay requests did not share exact native program");
    require(pooled.elementMappings.size()==2u &&
                pooled.elementMappings[0].elementIndex==10u &&
                pooled.elementMappings[1].elementIndex==11u,
            "program reuse lost per-tetrahedron source/geometry mappings");
    b.referenceVerticesM = request(source,"NSSC",11u,{0.02,0,0},0.002).referenceVerticesM;
    const auto refined=buildMeshProgramPool({a,b});
    require(refined.overlayPrograms.size()==2u &&
                refined.overlayMaterialIndices[0]!=refined.overlayMaterialIndices[1],
            "pool incorrectly quantized distinct physical characteristic lengths");
}

void checkFullFaceClassificationBoundary() {
    using numi::cardboard::FullFaceAmbiguityPolicy;
    using numi::cardboard::TetCrackPlane;
    using numi::cardboard::classifyTetCrackPlane;
    using numi::cardboard::computeTetCrackPlaneGeometry;
    using numi::cardboard::tetCrackPlaneClearsEveryFullFace;

    const TetCrackPlane zPlane{{0,0,1},FractureNormalFrame::Reference,0.0,{},1.0e-12};
    auto verticesForHeight = [](const double height) {
        return std::array<FractureVec3,4>{{
            {0,0,0}, {1,0,0}, {0,1,0}, {0,0,height}}};
    };

    const auto ordinary = verticesForHeight(1.0);
    const auto ordinaryClassification = classifyTetCrackPlane(ordinary,zPlane);
    require(tetCrackPlaneClearsEveryFullFace(ordinaryClassification),
            "ordinary tetrahedron unexpectedly aliases a full face within plane tolerance");
    (void)computeTetCrackPlaneGeometry(
        ordinary,zPlane,FullFaceAmbiguityPolicy::RejectWithinClassificationTolerance);

    // The existing volume degeneracy floor accepts this flattened tetrahedron,
    // while the ordinary polygon classifier classifies the entire base face as
    // on-plane. The retained builder must reject it before using that face as
    // a positive-area interior crack band.
    const auto sliver = verticesForHeight(2.0e-12);
    const auto sliverClassification = classifyTetCrackPlane(sliver,zPlane);
    require(sliverClassification.referenceVolumeM3 >
                128.0*std::numeric_limits<double>::epsilon()*
                    sliverClassification.edgeScaleM*sliverClassification.edgeScaleM*
                    sliverClassification.edgeScaleM,
            "sliver fixture no longer clears the existing volume floor");
    require(!tetCrackPlaneClearsEveryFullFace(sliverClassification),
            "sliver full-face ambiguity was not detected by the shared classifier");
    const auto legacySliverGeometry = computeTetCrackPlaneGeometry(sliver,zPlane);
    require(legacySliverGeometry.hasPositiveAreaCut() &&
                legacySliverGeometry.cutPolygonReferenceM.size()==3u,
            "fixture no longer demonstrates the old positive-area full-face cut");
    bool rejected = false;
    try {
        (void)computeTetCrackPlaneGeometry(
            sliver,zPlane,FullFaceAmbiguityPolicy::RejectWithinClassificationTolerance);
    } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"strict face policy accepted the ambiguous sliver cut");

    // Exact threshold equality is ambiguous: the classification itself uses
    // <= tolerance for its on-plane category, so admission is strictly greater.
    const double boundaryHeight = 4.0*ordinaryClassification.distanceToleranceM;
    const auto boundary = verticesForHeight(boundaryHeight);
    const auto boundaryClassification = classifyTetCrackPlane(boundary,zPlane);
    require(std::abs(boundaryClassification.signedDistanceM[0]) ==
                boundaryClassification.distanceToleranceM,
            "constructed equality fixture did not land exactly on classifier tolerance");
    require(!tetCrackPlaneClearsEveryFullFace(boundaryClassification),
            "exact tolerance boundary was not rejected");
    const auto outside = verticesForHeight(std::nextafter(
        boundaryHeight,std::numeric_limits<double>::infinity()));
    const auto outsideClassification = classifyTetCrackPlane(outside,zPlane);
    require(tetCrackPlaneClearsEveryFullFace(outsideClassification),
            "one-representable-step outside face boundary did not clear the strict gate");

    // Rotation is handled by the same classifier. A valid proper material
    // rotation of an ordinary tetrahedron remains admitted.
    TetCrackPlane rotated{{1,0,0},FractureNormalFrame::ElementMaterial,0.0,{},1.0e-12};
    rotated.materialToReference.value = {0,-1,0, 1,0,0, 0,0,1};
    const auto rotatedClassification = classifyTetCrackPlane(ordinary,rotated);
    require(tetCrackPlaneClearsEveryFullFace(rotatedClassification),
            "valid rotated material-plane control was rejected");
    (void)computeTetCrackPlaneGeometry(
        ordinary,rotated,FullFaceAmbiguityPolicy::RejectWithinClassificationTolerance);
}

void checkFailClosed(const std::filesystem::path& materials) {
    const auto source=parse(materials/"hajali2009_liner_hill_ideal.nmatter");
    auto bad=request(source,"liner",0u,{0,0,0},0.001);
    bad.explicitSet.rateBinding={};
    bool rejected=false;
    try {(void)buildElementProgram(bad);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"builder accepted a missing rate-evidence binding");
    bad=request(source,"liner",0u,{0,0,0},0.001);
    bad.sourceDensityKgPerM3+=1.0;
    rejected=false;
    try {(void)buildElementProgram(bad);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"builder accepted a changed source density");
    bad=request(source,"liner",0u,{0,0,0},0.001);
    bad.explicitSet.fractureModes[0].fractureEnergyJPerM2=1.0e-12;
    rejected=false;
    try {(void)buildElementProgram(bad);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"builder accepted a negative energy-derived softening interval");
    bad=request(source,"liner",1u,{0,0,0},0.001);
    bad.referenceVerticesM[3].z=2.0e-15;
    rejected=false;
    try {(void)buildElementProgram(bad);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"retained builder accepted a sliver whose zero-offset plane aliases a full tetrahedron face");
}

double relativeError(const std::vector<double>& a,const std::vector<double>& b) {
    require(a.size()==b.size(),"independent tangent vectors differ in size");
    double error2=0.0,reference2=0.0;
    for(std::size_t i=0;i<a.size();++i) {
        error2+=(a[i]-b[i])*(a[i]-b[i]);
        reference2+=b[i]*b[i];
    }
    return std::sqrt(error2)/std::max(1.0,std::sqrt(reference2));
}

void setParameter(MaterialProgram& material,const std::string& name,const double value) {
    const auto found=std::find_if(material.parameters.begin(),material.parameters.end(),
        [&](const Parameter& parameter){return parameter.name==name;});
    require(found!=material.parameters.end(),"test parameter not found: "+name);
    found->defaultValue=value;
    found->lower=value;
    found->upper=value;
}

std::uint32_t stateIndex(const MaterialProgram& material,const std::string& name);

double parameterValue(const MaterialProgram& material,const std::string& name) {
    for(const auto& parameter:material.parameters)
        if(parameter.name==name) return parameter.defaultValue;
    throw std::runtime_error("test cannot locate parameter "+name);
}

double hillYieldGap(const MaterialProgram& material,const LocalState& state) {
    const auto at=[&](const std::string& name) {
        return state[stateIndex(material,name)];
    };
    const double s11=at("sbar11"),s22=at("sbar22"),s33=at("sbar33");
    const double s23=at("sbar23"),s13=at("sbar13"),s12=at("sbar12");
    const double q2=parameterValue(material,"hill_F")*(s22-s33)*(s22-s33)+
        parameterValue(material,"hill_G")*(s33-s11)*(s33-s11)+
        parameterValue(material,"hill_H")*(s11-s22)*(s11-s22)+
        2.0*parameterValue(material,"hill_L")*s23*s23+
        2.0*parameterValue(material,"hill_M")*s13*s13+
        2.0*parameterValue(material,"hill_N")*s12*s12+
        std::pow(parameterValue(material,"q_floor"),2);
    double hardening=0.0;
    for(unsigned i=0;i<6u;++i)
        hardening+=parameterValue(material,"overlay_plastic_hardening_"+
            std::to_string(i))*at("overlay_plastic_history_"+std::to_string(i));
    for(unsigned axis=0;axis<3u;++axis)
        hardening+=parameterValue(material,"overlay_densification_modulus_pa_axis"+
            std::to_string(axis))*at("overlay_densification_"+std::to_string(axis));
    const double yieldRatio=1.0+hardening/parameterValue(material,"sigma_ref");
    return yieldRatio-std::sqrt(q2);
}

double hillYieldStrengthPa(const MaterialProgram& material,const LocalState& state) {
    double strength=parameterValue(material,"sigma_ref");
    for(unsigned i=0;i<6u;++i)
        strength+=parameterValue(material,"overlay_plastic_hardening_"+
            std::to_string(i))*state[stateIndex(material,
                "overlay_plastic_history_"+std::to_string(i))];
    for(unsigned axis=0;axis<3u;++axis)
        strength+=parameterValue(material,"overlay_densification_modulus_pa_axis"+
            std::to_string(axis))*state[stateIndex(material,
                "overlay_densification_"+std::to_string(axis))];
    return strength;
}

void checkStableNearCompleteDamageSurvival(const MaterialProgram& source,
                                           const ElementProgram& built,
                                           const ConstitutiveProgram& damaged) {
    // The crack-band energy multiplier is algebraically
    // (1-cap)+cap*product(survival).  At cap=1 and a nearly exhausted
    // exponential tail, `1-(1-survival)` suffers catastrophic FP32
    // cancellation. Verify both the algebra over ordinary values and the
    // generated stress response in that near-complete-damage regime.
    for (const float cap : {0.0f, 0.25f, 0.7f, 1.0f}) {
        for (const float survival : {0.0f, 1.0e-7f, 1.0e-4f, 0.1f, 0.9f}) {
            const float stable = (1.0f - cap) + cap * survival;
            const double expected = (1.0 - double(cap)) +
                                    double(cap) * double(survival);
            require(std::abs(double(stable) - expected) <= 6.0e-8,
                    "stable damage survival changed the cap/survival law");
        }
    }

    constexpr double dt = 0.002;
    const Matrix9 f{1.0038,0,0,0,1,0,0,0,1};
    LocalState accepted(damaged.material.internalState.size());
    for (std::size_t i=0;i<accepted.size();++i)
        accepted[i]=damaged.material.internalState[i].initialValue;
    auto candidate=project(damaged,f,accepted,dt).state;
    const auto damageIndex=stateIndex(damaged.material,"overlay_fracture_excess_0");
    candidate[damageIndex]=0.01;
    for (unsigned mode=1u;mode<8u;++mode)
        candidate[stateIndex(damaged.material,"overlay_fracture_excess_"+
                             std::to_string(mode))]=0.0;

    LocalState sourceAccepted(source.internalState.size());
    LocalState sourceCandidate(source.internalState.size());
    for (std::size_t i=0;i<source.internalState.size();++i) {
        sourceAccepted[i]=source.internalState[i].initialValue;
        sourceCandidate[i]=candidate[stateIndex(damaged.material,
                                                source.internalState[i].name)];
    }
    const auto sourceCompiled=numi::matter::detail::compileConstitutive(
        source,NM_EXPRESSION_STACK_CAPACITY);
    require(sourceCompiled.succeeded(),
            "retained source did not compile for near-complete-damage check");
    const auto sourceStress=stress(sourceCompiled.program,f,sourceAccepted,
                                   sourceCandidate,dt);
    const auto damagedStress=stress(damaged,f,accepted,candidate,dt);
    const double onset=built.mapping.modeOnsetStrain[0];
    const double softening=built.mapping.modeSofteningStrain[0];
    const double excess=candidate[damageIndex];
    const double survival=(onset/(onset+excess))*std::exp(-excess/softening);
    const double observedRatio=damagedStress[0]/sourceStress[0];
    require(std::abs(observedRatio-survival) <= 2.0e-6*std::max(survival,1.0e-12),
            "generated near-complete-damage stress lost the surviving energy fraction");

    const float survival32=static_cast<float>(survival);
    const float oldCancelled=1.0f-(1.0f-survival32);
    const float stable32=(1.0f-1.0f)+1.0f*survival32;
    require(survival32 < 1.0e-6f &&
                std::abs(oldCancelled-stable32) > 0.25f*stable32,
            "fixture did not expose the old near-complete-damage FP32 cancellation");
    std::cout<<"near_complete_damage_survival="<<std::setprecision(17)
             <<survival<<'\n'
             <<"near_complete_damage_observed_energy_ratio="<<observedRatio<<'\n'
             <<"near_complete_damage_old_fp32_survival="<<oldCancelled<<'\n'
             <<"near_complete_damage_stable_fp32_survival="<<stable32<<'\n';
}

void checkHistoryTangentAndWork(const std::filesystem::path& materials) {
    const auto source=parse(materials/"hajali2009_liner_hill_ideal.nmatter");
    auto input=request(source,"liner",41u,{0,0,0},0.001);
    input.explicitSet.damageCap=1.0;
    // Synthetic pure-mode fixture: leave the other seven plane channels far
    // below threshold, including their smoothed positive-part tails, so they
    // do not contaminate the mode-0 Gc*A budget.
    for(auto& mode:input.explicitSet.fractureModes) mode={1.0e8,1.0e6,1.0e-6};
    input.explicitSet.fractureModes[0]={1.0e6,0.4,1.0e-6};
    const auto built=buildElementProgram(input);
    const auto compiledResult=numi::matter::detail::compileConstitutive(
        built.program,NM_EXPRESSION_STACK_CAPACITY);
    require(compiledResult.succeeded(),"damage candidate did not compile for the tangent/work test");
    const auto& compiled=compiledResult.program;
    checkStableNearCompleteDamageSurvival(source,built,compiled);

    const auto damaged=tensionHistory(compiled,built.densityKgPerM3,
                                      built.mapping.planeGeometry,1.0038,true);
    for(std::size_t i=1;i<damaged.trajectory.size();++i)
        for(unsigned mode=0;mode<8u;++mode) {
            const auto state=stateIndex(compiled.material,
                "overlay_fracture_excess_"+std::to_string(mode));
            require(damaged.trajectory[i][state]+1.0e-10>=damaged.trajectory[i-1][state],
                    "fracture history decreased during load/unload");
        }
    const auto activeDamage=stateIndex(compiled.material,"overlay_fracture_excess_0");
    require(damaged.stateAtMaximum[activeDamage]>1.0e-5,
            "synthetic tensile history failed to activate the selected crack plane");
    for(unsigned mode=1u;mode<8u;++mode) {
        const auto state=stateIndex(compiled.material,
            "overlay_fracture_excess_"+std::to_string(mode));
        require(damaged.stateAtMaximum[state]<1.0e-5,
                "pure-mode energy fixture activated an unintended fracture channel "+
                std::to_string(mode));
    }
    auto rateMaterial=built.program;
    setParameter(rateMaterial,"overlay_rate_tau_s_mode0",0.004);
    const auto rateCompile=numi::matter::detail::compileConstitutive(
        rateMaterial,NM_EXPRESSION_STACK_CAPACITY);
    require(rateCompile.succeeded(),"explicit-rate comparison program did not compile");
    const auto shortRateStep=project(rateCompile.program,damaged.maximumF,
                                     damaged.acceptedAtMaximum,0.0005);
    const auto longRateStep=project(rateCompile.program,damaged.maximumF,
                                    damaged.acceptedAtMaximum,0.004);
    require(longRateStep.state[activeDamage]>shortRateStep.state[activeDamage],
            "fracture state did not respond monotonically to the explicit rate time step");
    for(unsigned state=0;state<6u;++state) {
        const auto hardening=stateIndex(compiled.material,
            "overlay_plastic_history_"+std::to_string(state));
        require(std::abs(damaged.stateAtMaximum[hardening])<1.0e-8,
                "sub-yield damage verification unexpectedly activated plastic hardening");
    }

    Matrix9 direction{};
    direction[0]=1.0;
    direction[4]=0.21;
    direction[1]=0.13;
    const auto tangent=consistentTangentAction(compiled,damaged.maximumF,
        damaged.acceptedAtMaximum,damaged.candidateAtMaximum,direction,0.002);
    constexpr double h=2.0e-6;
    Matrix9 plusF=damaged.maximumF,minusF=damaged.maximumF;
    for(unsigned i=0;i<9u;++i) {
        plusF[i]+=h*direction[i];
        minusF[i]-=h*direction[i];
    }
    const auto plus=project(compiled,plusF,damaged.acceptedAtMaximum,0.002);
    const auto minus=project(compiled,minusF,damaged.acceptedAtMaximum,0.002);
    const auto plusStress=stress(compiled,plusF,damaged.acceptedAtMaximum,plus.state,0.002);
    const auto minusStress=stress(compiled,minusF,damaged.acceptedAtMaximum,minus.state,0.002);
    std::vector<double> finiteDifference(9u);
    for(unsigned i=0;i<9u;++i) finiteDifference[i]=(plusStress[i]-minusStress[i])/(2.0*h);
    const double tangentError=relativeError(tangent,finiteDifference);
    require(tangentError<2.0e-3,
            "full implicit AD tangent disagrees with independent history re-projection: "+
            std::to_string(tangentError));

    fp64::Matrix oracleF{};
    fp64::Matrix oracleDirection{};
    fp64::State oracleAccepted=damaged.acceptedAtMaximum;
    for(unsigned i=0u;i<9u;++i) {
        oracleF[i]=static_cast<double>(static_cast<float>(damaged.maximumF[i]));
        oracleDirection[i]=static_cast<double>(static_cast<float>(direction[i]));
    }
    for(double& value:oracleAccepted)
        value=static_cast<double>(static_cast<float>(value));
    const double oracleDt=static_cast<double>(static_cast<float>(0.002));
    const auto oracleProjection=fp64::project(compiled,oracleF,oracleAccepted,oracleDt);
    const auto oracleStress=fp64::stress(compiled,oracleF,oracleAccepted,
                                         oracleProjection.state,oracleDt);
    const auto oracleTangent=fp64::tangentAction(compiled,oracleF,oracleAccepted,
        oracleProjection.state,oracleDirection,oracleDt);
    const auto referenceProjection=project(compiled,oracleF,oracleAccepted,oracleDt);
    const auto referenceStress=stress(compiled,oracleF,oracleAccepted,
                                      referenceProjection.state,oracleDt);
    const auto referenceTangent=consistentTangentAction(compiled,oracleF,
        oracleAccepted,referenceProjection.state,oracleDirection,oracleDt);
    fp64::Matrix referenceTangentMatrix{};
    std::copy(referenceTangent.begin(),referenceTangent.end(),
              referenceTangentMatrix.begin());
    const double oracleStressSelfError=fp64::differenceNorm(oracleStress,referenceStress)/
        std::max(1.0,fp64::norm2(referenceStress));
    const double oracleTangentSelfError=fp64::differenceNorm(oracleTangent,
        referenceTangentMatrix)/std::max(1.0,fp64::norm2(referenceTangentMatrix));
    require(oracleProjection.residualInf<=2.0e-11&&
            relativeError(oracleProjection.state,referenceProjection.state)<1.0e-12&&
            oracleStressSelfError<1.0e-12&&oracleTangentSelfError<1.0e-12,
            "independent FP64 oracle did not reproduce its source-program projection/stress/tangent");
    std::cout<<"fp64_native_capture_oracle_selfcheck_residual="
             <<oracleProjection.residualInf<<"\n"
             <<"fp64_native_capture_oracle_tangent_selfcheck="
             <<oracleTangentSelfError<<"\n";

    // Compare actual closed-cycle external work minus free-energy change with
    // an otherwise identical undamaged run. This is an integrated-work test,
    // not a claim that this phenomenological overlay dissipates exactly Gc*A.
    auto undamagedMaterial=built.program;
    setParameter(undamagedMaterial,"overlay_damage_cap",0.0);
    const auto undamagedCompile=numi::matter::detail::compileConstitutive(
        undamagedMaterial,NM_EXPRESSION_STACK_CAPACITY);
    require(undamagedCompile.succeeded(),"undamaged comparison did not compile");
    const auto control=tensionHistory(undamagedCompile.program,built.densityKgPerM3,
                                     built.mapping.planeGeometry,1.0038,true);
    const auto sourceHillCompile=numi::matter::detail::compileConstitutive(
        source,NM_EXPRESSION_STACK_CAPACITY);
    require(sourceHillCompile.succeeded(),"retained source Hill control did not compile");
    const auto sourcePlasticCycle=tensionHistory(sourceHillCompile.program,
        built.densityKgPerM3,built.mapping.planeGeometry,1.012,true);
    const double sourcePlasticDissipation=sourcePlasticCycle.externalWorkJ-
        sourcePlasticCycle.finalEnergyJ;
    const double damagedDissipation=damaged.externalWorkJ-damaged.finalEnergyJ;
    const double controlDissipation=control.externalWorkJ-control.finalEnergyJ;
    const double damageIncrement=damagedDissipation-controlDissipation;
    const double expectedGcArea=input.explicitSet.fractureModes[0].fractureEnergyJPerM2*
                                built.mapping.planeGeometry[0].referenceAreaM2;
    const double terminalDamageExcess=damaged.stateAtMaximum[activeDamage];
    const double terminalSofteningTailFraction=
        (input.explicitSet.fractureModes[0].strengthPa*
         built.mapping.modeLengthM[0]*built.mapping.modeSofteningStrain[0]*
         std::exp(-terminalDamageExcess/built.mapping.modeSofteningStrain[0]))/
        input.explicitSet.fractureModes[0].fractureEnergyJPerM2;
    std::vector<std::pair<unsigned,double>> workQuadrature;
    for(const unsigned steps:{120u,240u}) {
        const auto resolvedDamage=tensionHistory(compiled,built.densityKgPerM3,
            built.mapping.planeGeometry,1.0038,false,0.0,steps);
        const auto resolvedControl=tensionHistory(undamagedCompile.program,
            built.densityKgPerM3,built.mapping.planeGeometry,1.0038,false,0.0,steps);
        const double resolvedWork=(resolvedDamage.externalWorkJ-resolvedDamage.finalEnergyJ)-
            (resolvedControl.externalWorkJ-resolvedControl.finalEnergyJ);
        workQuadrature.emplace_back(steps,resolvedWork/expectedGcArea);
    }
    const auto upperRangeDamage=tensionHistory(compiled,built.densityKgPerM3,
        built.mapping.planeGeometry,1.0040,true,0.0,120u);
    const auto upperRangeControl=tensionHistory(undamagedCompile.program,
        built.densityKgPerM3,built.mapping.planeGeometry,1.0040,true,0.0,120u);
    const auto plasticMultiplierIndex=stateIndex(compiled.material,"accumulated_multiplier");
    require(upperRangeDamage.stateAtMaximum[plasticMultiplierIndex]<1.0e-8&&
            upperRangeControl.stateAtMaximum[plasticMultiplierIndex]<1.0e-8,
            "upper pure-damage loading range crossed into the Hill plastic branch");
    const double upperRangeDamageWork=(upperRangeDamage.externalWorkJ-
            upperRangeDamage.finalEnergyJ)-
        (upperRangeControl.externalWorkJ-upperRangeControl.finalEnergyJ);
    const double upperRangeGcRatio=upperRangeDamageWork/expectedGcArea;
    // The 1.0038, 1 MPa fixture deliberately retains a finite opening; its
    // sub-Gc*A loop work includes an exponential tail and recoverable endpoint
    // energy. For a separate energy-closure test, use an explicitly synthetic
    // 2 MPa onset (below source Hill yield) and drive farther so the
    // exponential damage tail is nearly exhausted without activating source
    // plasticity. This is a law-integration check, not a grade parameter.
    auto completeInput=request(source,"liner",45u,{0,0,0},0.001);
    completeInput.explicitSet=input.explicitSet;
    completeInput.explicitSet.fractureModes[0].strengthPa=2.0e6;
    const auto completeBuilt=buildElementProgram(completeInput);
    const auto completeCompile=numi::matter::detail::compileConstitutive(
        completeBuilt.program,NM_EXPRESSION_STACK_CAPACITY);
    require(completeCompile.succeeded(),"full-softening energy candidate did not compile");
    constexpr double fullOpeningStretch=1.0040;
    const auto fullOpening120=tensionHistory(completeCompile.program,
        completeBuilt.densityKgPerM3,completeBuilt.mapping.planeGeometry,
        fullOpeningStretch,true,0.0,120u);
    const auto fullOpening240=tensionHistory(completeCompile.program,
        completeBuilt.densityKgPerM3,completeBuilt.mapping.planeGeometry,
        fullOpeningStretch,true,0.0,240u);
    const auto fullOpeningMultiplier=stateIndex(completeCompile.program.material,
                                                "accumulated_multiplier");
    require(fullOpening120.stateAtMaximum[fullOpeningMultiplier]<1.0e-8&&
            fullOpening240.stateAtMaximum[fullOpeningMultiplier]<1.0e-8,
            "full-softening energy path crossed into the retained Hill plastic branch ("+
            number(fullOpening120.stateAtMaximum[fullOpeningMultiplier])+", "+
            number(fullOpening240.stateAtMaximum[fullOpeningMultiplier])+")");
    const double fullOpeningWork120=fullOpening120.externalWorkJ-
                                    fullOpening120.finalEnergyJ;
    const double fullOpeningWork240=fullOpening240.externalWorkJ-
                                    fullOpening240.finalEnergyJ;
    const double fullOpeningRatio120=fullOpeningWork120/expectedGcArea;
    const double fullOpeningRatio240=fullOpeningWork240/expectedGcArea;
    const auto completeDamageIndex=stateIndex(completeCompile.program.material,
                                               "overlay_fracture_excess_0");
    const double fullOpeningExcess=fullOpening240.stateAtMaximum[completeDamageIndex];
    const double fullOpeningSofteningWidth=completeBuilt.mapping.modeSofteningStrain[0];
    require(std::isfinite(fullOpeningRatio120)&&std::isfinite(fullOpeningRatio240)&&
            fullOpeningWork120>0.0&&fullOpeningWork240>0.0&&
            std::abs(fullOpeningRatio240-1.0)<=0.02&&
            std::abs(fullOpeningWork240/fullOpeningWork120-1.0)<=0.02,
            "full-softening pure-mode cycle missed the Gc*A or 120/240-step 2% gates");

    // Exercise the V/A mapping through the constitutive history and integrated
    // work, not only through a geometry-only helper. Rotation and winding
    // must preserve the actual Gc*A dissipation budget.
    auto rotatedEnergyInput=completeInput;
    for (auto& vertex:rotatedEnergyInput.referenceVerticesM)
        vertex=quarterTurnZ(vertex);
    for (auto& plane:rotatedEnergyInput.selectedMaterialPlanes)
        plane.normal=quarterTurnZ(plane.normal);
    const auto rotatedEnergyBuilt=buildElementProgram(rotatedEnergyInput);
    const auto rotatedEnergyCompile=numi::matter::detail::compileConstitutive(
        rotatedEnergyBuilt.program,NM_EXPRESSION_STACK_CAPACITY);
    require(rotatedEnergyCompile.succeeded(),
            "rotated crack-band energy fixture did not compile");
    const auto rotatedEnergy=tensionHistory(rotatedEnergyCompile.program,
        rotatedEnergyBuilt.densityKgPerM3,rotatedEnergyBuilt.mapping.planeGeometry,
        fullOpeningStretch,true,0.0,240u);
    const double rotatedEnergyWork=rotatedEnergy.externalWorkJ-rotatedEnergy.finalEnergyJ;
    const double rotatedGcArea=rotatedEnergyInput.explicitSet.fractureModes[0].fractureEnergyJPerM2*
        rotatedEnergyBuilt.mapping.planeGeometry[0].referenceAreaM2;
    const double rotatedEnergyRatio=rotatedEnergyWork/rotatedGcArea;
    require(rotatedEnergyBuilt.mapping.planeGeometry[0].referenceAreaM2>0.0&&
            rotatedEnergyBuilt.mapping.modeLengthM[0]>0.0&&
            std::abs(rotatedEnergyBuilt.mapping.modeLengthM[0]/
                     completeBuilt.mapping.modeLengthM[0]-1.0)<=1.0e-12&&
            std::abs(rotatedEnergyRatio-1.0)<=0.02&&
            std::abs(rotatedEnergyWork/fullOpeningWork240-1.0)<=0.02,
            "rotated crack-plane damage work missed the invariant Gc*A 2% gate");

    auto reversedEnergyInput=completeInput;
    std::swap(reversedEnergyInput.referenceVerticesM[1],
              reversedEnergyInput.referenceVerticesM[2]);
    const auto reversedEnergyBuilt=buildElementProgram(reversedEnergyInput);
    const auto reversedEnergyCompile=numi::matter::detail::compileConstitutive(
        reversedEnergyBuilt.program,NM_EXPRESSION_STACK_CAPACITY);
    require(reversedEnergyCompile.succeeded(),
            "reversed-winding crack-band energy fixture did not compile");
    const auto reversedEnergy=tensionHistory(reversedEnergyCompile.program,
        reversedEnergyBuilt.densityKgPerM3,reversedEnergyBuilt.mapping.planeGeometry,
        fullOpeningStretch,true,0.0,240u);
    const double reversedEnergyWork=reversedEnergy.externalWorkJ-reversedEnergy.finalEnergyJ;
    const double reversedGcArea=reversedEnergyInput.explicitSet.fractureModes[0].fractureEnergyJPerM2*
        reversedEnergyBuilt.mapping.planeGeometry[0].referenceAreaM2;
    const double reversedEnergyRatio=reversedEnergyWork/reversedGcArea;
    require(std::abs(reversedEnergyRatio-1.0)<=0.02&&
            std::abs(reversedEnergyWork/fullOpeningWork240-1.0)<=0.02,
            "reversed-winding crack-plane damage work missed the invariant Gc*A 2% gate");

    require(std::isfinite(damageIncrement)&&damageIncrement>0.0,
            "damaging closed-cycle did not produce positive integrated dissipation above control");
    require(std::isfinite(expectedGcArea)&&expectedGcArea>0.0,
            "selected-plane Gc*A diagnostic is invalid");

    auto coarseInput=request(source,"liner",42u,{0,0,0},0.002);
    coarseInput.explicitSet=input.explicitSet;
    const auto coarseBuilt=buildElementProgram(coarseInput);
    const auto coarseCompiled=numi::matter::detail::compileConstitutive(
        coarseBuilt.program,NM_EXPRESSION_STACK_CAPACITY);
    require(coarseCompiled.succeeded(),"coarse damage/refinement case did not compile");
    const auto coarse=tensionHistory(coarseCompiled.program,coarseBuilt.densityKgPerM3,
        coarseBuilt.mapping.planeGeometry,1.0038,false);
    auto coarseControlMaterial=coarseBuilt.program;
    setParameter(coarseControlMaterial,"overlay_damage_cap",0.0);
    const auto coarseControlCompiled=numi::matter::detail::compileConstitutive(
        coarseControlMaterial,NM_EXPRESSION_STACK_CAPACITY);
    require(coarseControlCompiled.succeeded(),"coarse undamaged control did not compile");
    const auto coarseControl=tensionHistory(coarseControlCompiled.program,
        coarseBuilt.densityKgPerM3,coarseBuilt.mapping.planeGeometry,1.0038,false);
    const double coarseDamageWork=(coarse.externalWorkJ-coarse.finalEnergyJ)-
        (coarseControl.externalWorkJ-coarseControl.finalEnergyJ);
    const double finePerArea=damageIncrement/built.mapping.planeGeometry[0].referenceAreaM2;
    const double coarsePerArea=coarseDamageWork/
        coarseBuilt.mapping.planeGeometry[0].referenceAreaM2;
    require(std::isfinite(finePerArea)&&std::isfinite(coarsePerArea)&&
            damageIncrement>0.0&&coarseDamageWork>0.0,
            "refinement sweep did not produce finite positive integrated work");
    const double refinementDifference=std::abs(finePerArea/coarsePerArea-1.0);
    require(refinementDifference<0.02,
            "pure-mode crack-band work failed the separate 2x-refinement screen");

    auto obliqueInput=request(source,"liner",43u,{0,0,0},0.001);
    obliqueInput.explicitSet=input.explicitSet;
    obliqueInput.explicitSet.fractureModes[0]={1.0e6,0.4,1.0e-6};
    const double invRoot2=1.0/std::sqrt(2.0);
    obliqueInput.selectedMaterialPlanes[0].normal={invRoot2,invRoot2,0.0};
    const auto obliqueBuilt=buildElementProgram(obliqueInput);
    const auto obliqueCompiled=numi::matter::detail::compileConstitutive(
        obliqueBuilt.program,NM_EXPRESSION_STACK_CAPACITY);
    require(obliqueCompiled.succeeded(),"oblique selected-plane case did not compile");
    const auto oblique=tensionHistory(obliqueCompiled.program,obliqueBuilt.densityKgPerM3,
        obliqueBuilt.mapping.planeGeometry,1.0038,true);
    const double obliqueWork=oblique.externalWorkJ-oblique.finalEnergyJ;
    require(oblique.stateAtMaximum[activeDamage]>1.0e-7&&obliqueWork>0.0&&
            std::isfinite(obliqueWork),
            "rotated material-frame selected plane did not produce finite history/dissipation");

    auto mixedInput=request(source,"liner",44u,{0,0,0},0.001);
    mixedInput.explicitSet=input.explicitSet;
    for(unsigned mode=2u;mode<8u;++mode)
        mixedInput.explicitSet.fractureModes[mode]={1.0e8,1.0e8,1.0e-6};
    mixedInput.explicitSet.fractureModes[1]={0.1e6,0.05,1.0e-6};
    const auto mixedBuilt=buildElementProgram(mixedInput);
    const auto mixedCompiled=numi::matter::detail::compileConstitutive(
        mixedBuilt.program,NM_EXPRESSION_STACK_CAPACITY);
    require(mixedCompiled.succeeded(),"mixed normal/shear case did not compile");
    const auto mixed=tensionHistory(mixedCompiled.program,mixedBuilt.densityKgPerM3,
        mixedBuilt.mapping.planeGeometry,1.002,true,0.0014);
    const auto shearDamage=stateIndex(mixedCompiled.program.material,"overlay_fracture_excess_1");
    require(mixed.stateAtMaximum[activeDamage]>1.0e-5&&
            mixed.stateAtMaximum[shearDamage]>1.0e-5,
            "mixed-mode history failed to activate both normal and shear channels");
    require(mixed.stateAtMaximum[stateIndex(mixedCompiled.program.material,
                "accumulated_multiplier")]<1.0e-8,
            "mixed-mode budget check crossed the source Hill plastic yield surface");
    Matrix9 mixedDirection{};
    mixedDirection[0]=1.0;
    mixedDirection[1]=0.5;
    const auto mixedTangent=consistentTangentAction(mixedCompiled.program,mixed.maximumF,
        mixed.acceptedAtMaximum,mixed.candidateAtMaximum,mixedDirection,0.002);
    Matrix9 mixedPlusF=mixed.maximumF,mixedMinusF=mixed.maximumF;
    for(unsigned i=0;i<9u;++i) {
        mixedPlusF[i]+=h*mixedDirection[i];
        mixedMinusF[i]-=h*mixedDirection[i];
    }
    const auto mixedPlus=project(mixedCompiled.program,mixedPlusF,mixed.acceptedAtMaximum,0.002);
    const auto mixedMinus=project(mixedCompiled.program,mixedMinusF,mixed.acceptedAtMaximum,0.002);
    const auto mixedPlusStress=stress(mixedCompiled.program,mixedPlusF,
        mixed.acceptedAtMaximum,mixedPlus.state,0.002);
    const auto mixedMinusStress=stress(mixedCompiled.program,mixedMinusF,
        mixed.acceptedAtMaximum,mixedMinus.state,0.002);
    std::vector<double> mixedFD(9u);
    for(unsigned i=0;i<9u;++i) mixedFD[i]=(mixedPlusStress[i]-mixedMinusStress[i])/(2.0*h);
    const double mixedTangentError=relativeError(mixedTangent,mixedFD);
    require(mixedTangentError<2.0e-3,
            "mixed-mode history AD tangent disagrees with independent reprojection");
    auto mixedControlMaterial=mixedBuilt.program;
    setParameter(mixedControlMaterial,"overlay_damage_cap",0.0);
    const auto mixedControlCompile=numi::matter::detail::compileConstitutive(
        mixedControlMaterial,NM_EXPRESSION_STACK_CAPACITY);
    require(mixedControlCompile.succeeded(),"mixed-mode undamaged control did not compile");
    const auto mixedControl=tensionHistory(mixedControlCompile.program,
        mixedBuilt.densityKgPerM3,mixedBuilt.mapping.planeGeometry,1.002,false,0.0014);
    const double mixedDamageWork=(mixed.externalWorkJ-mixed.finalEnergyJ)-
        (mixedControl.externalWorkJ-mixedControl.finalEnergyJ);
    double mixedGcBudget=0.0;
    std::string mixedActiveModes;
    for(unsigned mode=0;mode<8u;++mode) {
        const auto state=stateIndex(mixedCompiled.program.material,
            "overlay_fracture_excess_"+std::to_string(mode));
        if(mixed.stateAtMaximum[state]>1.0e-5) {
            const unsigned plane=mode/2u;
            mixedGcBudget+=mixedInput.explicitSet.fractureModes[mode].fractureEnergyJPerM2*
                mixedBuilt.mapping.planeGeometry[plane].referenceAreaM2;
            mixedActiveModes+=std::to_string(mode)+",";
        }
    }
    const double mixedBudgetRatio=mixedDamageWork/mixedGcBudget;
    require(std::isfinite(mixedBudgetRatio)&&mixedDamageWork>0.0,
            "mixed-mode integrated damage work or explicit plane budget is invalid");
    require(mixedBudgetRatio<=1.02,
            "mixed-mode synthetic path exceeded the selected-plane additive Gc budget");
    require(mixedBudgetRatio<=1.02,
            "mixed-mode synthetic path exceeded the sum of active selected-plane Gc area budgets");

    const auto compression=compressionHistory(compiled,built.mapping.referenceVolumeM3,0.80);
    const auto densification=stateIndex(compiled.material,"overlay_densification_0");
    require(compression.stateAtMaximum[densification]>1.0e-5&&
            compression.finalEnergyJ>0.0&&std::isfinite(compression.externalWorkJ),
            "compression cycle failed to activate finite densification history and endpoint energy");
    const auto compressionMultiplier=stateIndex(compiled.material,"accumulated_multiplier");
    std::array<std::uint32_t,6> compressionHardening{};
    std::array<std::uint32_t,3> compressionDensification{};
    for(unsigned i=0;i<6u;++i)
        compressionHardening[i]=stateIndex(compiled.material,
            "overlay_plastic_history_"+std::to_string(i));
    for(unsigned i=0;i<3u;++i)
        compressionDensification[i]=stateIndex(compiled.material,
            "overlay_densification_"+std::to_string(i));
    double maximumDensificationYieldStrength=hillYieldStrengthPa(
        compiled.material,compression.trajectory.front());
    for(std::size_t i=1;i<compression.trajectory.size();++i) {
        require(compression.trajectory[i][compressionMultiplier]+1.0e-10>=
                    compression.trajectory[i-1u][compressionMultiplier],
                "compression accumulated plastic multiplier decreased during unloading");
        for(const auto state:compressionHardening)
            require(compression.trajectory[i][state]+1.0e-10>=
                        compression.trajectory[i-1u][state],
                    "compression plastic hardening history decreased during unloading");
        for(const auto state:compressionDensification)
            require(compression.trajectory[i][state]+1.0e-10>=
                        compression.trajectory[i-1u][state],
                    "compression densification history decreased during unloading");
        const double strength=hillYieldStrengthPa(compiled.material,
                                                   compression.trajectory[i]);
        require(strength+1.0e-6>=maximumDensificationYieldStrength,
                "compression hardening/densification reduced the retained Hill yield threshold");
        maximumDensificationYieldStrength=strength;
    }
    double maximumCompressionDamage=0.0;
    for(unsigned mode=0;mode<8u;++mode) {
        const auto damageState=stateIndex(compiled.material,
            "overlay_fracture_excess_"+std::to_string(mode));
        for(const auto& state:compression.trajectory) {
            maximumCompressionDamage=std::max(maximumCompressionDamage,
                                               state[damageState]);
            require(state[damageState]<1.0e-7,
                    "compression-only budget activated an unintended tensile/shear damage mode");
        }
    }
    Matrix9 compressionDirection{};
    compressionDirection[0]=1.0;
    const auto compressionTangent=consistentTangentAction(compiled,compression.maximumF,
        compression.acceptedAtMaximum,compression.candidateAtMaximum,
        compressionDirection,0.002);
    Matrix9 compressionPlusF=compression.maximumF,compressionMinusF=compression.maximumF;
    compressionPlusF[0]+=h;
    compressionMinusF[0]-=h;
    const auto compressionPlus=project(compiled,compressionPlusF,
        compression.acceptedAtMaximum,0.002);
    const auto compressionMinus=project(compiled,compressionMinusF,
        compression.acceptedAtMaximum,0.002);
    const auto compressionPlusStress=stress(compiled,compressionPlusF,
        compression.acceptedAtMaximum,compressionPlus.state,0.002);
    const auto compressionMinusStress=stress(compiled,compressionMinusF,
        compression.acceptedAtMaximum,compressionMinus.state,0.002);
    std::vector<double> compressionFD(9u);
    for(unsigned i=0;i<9u;++i)
        compressionFD[i]=(compressionPlusStress[i]-compressionMinusStress[i])/(2.0*h);
    const double compressionTangentError=relativeError(compressionTangent,compressionFD);
    require(compressionTangentError<2.0e-3,
            "full compression/densification AD tangent disagrees with independent reprojection");

    auto undensifiedMaterial=built.program;
    for(unsigned axis=0;axis<3u;++axis)
        setParameter(undensifiedMaterial,
            "overlay_densification_modulus_pa_axis"+std::to_string(axis),1.0e-6);
    const auto undensifiedCompile=numi::matter::detail::compileConstitutive(
        undensifiedMaterial,NM_EXPRESSION_STACK_CAPACITY);
    require(undensifiedCompile.succeeded(),"no-added-densification control did not compile");
    const auto compressionControl=compressionHistory(undensifiedCompile.program,
        built.mapping.referenceVolumeM3,0.80);
    const auto sourceHillCompression=compressionHistory(sourceHillCompile.program,
        built.mapping.referenceVolumeM3,0.80);
    const double compressionDissipation=compression.externalWorkJ-compression.finalEnergyJ;
    const double compressionControlDissipation=
        compressionControl.externalWorkJ-compressionControl.finalEnergyJ;
    const double sourceHillCompressionDissipation=sourceHillCompression.externalWorkJ-
        sourceHillCompression.finalEnergyJ;
    const double densificationDissipation=compressionDissipation-compressionControlDissipation;
    const double densificationMinusSourceHill=
        compressionDissipation-sourceHillCompressionDissipation;
    const double initialDensificationYieldStrength=hillYieldStrengthPa(
        compiled.material,compression.trajectory.front());
    const double peakDensificationYieldStrength=hillYieldStrengthPa(
        compiled.material,compression.stateAtMaximum);
    const double finalDensificationYieldStrength=hillYieldStrengthPa(
        compiled.material,compression.finalState);
    require(std::isfinite(compressionDissipation)&&compressionDissipation>0.0&&
            std::isfinite(compressionControlDissipation)&&compressionControlDissipation>0.0&&
            std::isfinite(sourceHillCompressionDissipation)&&
            sourceHillCompressionDissipation>0.0&&
            std::isfinite(densificationDissipation),
            "compression/densification cycle did not produce finite positive total dissipation");

    const auto plasticCycle=tensionHistory(undamagedCompile.program,built.densityKgPerM3,
        built.mapping.planeGeometry,1.012,true);
    const auto multiplier=stateIndex(undamagedCompile.program.material,"accumulated_multiplier");
    const auto mdHardening=stateIndex(undamagedCompile.program.material,
                                     "overlay_plastic_history_0");
    require(plasticCycle.stateAtMaximum[multiplier]>1.0e-5&&
            plasticCycle.stateAtMaximum[mdHardening]>1.0e-5,
            "above-yield cycle failed to exercise the explicit MD hardening history");
    Matrix9 plasticDirection{};
    plasticDirection[0]=1.0;
    const auto plasticTangent=consistentTangentAction(undamagedCompile.program,
        plasticCycle.maximumF,plasticCycle.acceptedAtMaximum,
        plasticCycle.candidateAtMaximum,plasticDirection,0.002);
    Matrix9 plasticPlusF=plasticCycle.maximumF,plasticMinusF=plasticCycle.maximumF;
    plasticPlusF[0]+=h;
    plasticMinusF[0]-=h;
    const auto plasticPlus=project(undamagedCompile.program,plasticPlusF,
        plasticCycle.acceptedAtMaximum,0.002);
    const auto plasticMinus=project(undamagedCompile.program,plasticMinusF,
        plasticCycle.acceptedAtMaximum,0.002);
    const auto plasticPlusStress=stress(undamagedCompile.program,plasticPlusF,
        plasticCycle.acceptedAtMaximum,plasticPlus.state,0.002);
    const auto plasticMinusStress=stress(undamagedCompile.program,plasticMinusF,
        plasticCycle.acceptedAtMaximum,plasticMinus.state,0.002);
    std::vector<double> plasticFD(9u);
    for(unsigned i=0;i<9u;++i)
        plasticFD[i]=(plasticPlusStress[i]-plasticMinusStress[i])/(2.0*h);
    const double plasticTangentError=relativeError(plasticTangent,plasticFD);
    require(plasticTangentError<2.0e-3,
            "full plastic/hardening AD tangent disagrees with independent reprojection");
    const double plasticDissipation=plasticCycle.externalWorkJ-plasticCycle.finalEnergyJ;
    require(std::isfinite(plasticDissipation)&&plasticDissipation>0.0,
            "above-yield plastic cycle did not produce positive integrated dissipation");

    constexpr double tightProjectionTolerance=2.0e-12;
    const double compressionGateError1=reprojectedTangentError(compiled,
        compression.maximumF,compression.acceptedAtMaximum,
        compression.candidateAtMaximum,compressionDirection,0.002,1.0e-6,
        tightProjectionTolerance);
    const double compressionGateError2=reprojectedTangentError(compiled,
        compression.maximumF,compression.acceptedAtMaximum,
        compression.candidateAtMaximum,compressionDirection,0.002,3.0e-8,
        tightProjectionTolerance);
    const double mixedGateError1=reprojectedTangentError(mixedCompiled.program,
        mixed.maximumF,mixed.acceptedAtMaximum,mixed.candidateAtMaximum,
        mixedDirection,0.002,3.0e-7,tightProjectionTolerance);
    const double mixedGateError2=reprojectedTangentError(mixedCompiled.program,
        mixed.maximumF,mixed.acceptedAtMaximum,mixed.candidateAtMaximum,
        mixedDirection,0.002,3.0e-8,tightProjectionTolerance);
    require(compressionGateError1<=1.0e-6&&compressionGateError2<=1.0e-6,
            "compression/densification tangent missed the two-step smooth FP64 gate");
    require(mixedGateError1<=1.0e-6&&mixedGateError2<=1.0e-6,
            "mixed-mode tangent missed the two-step smooth FP64 gate");

    std::vector<std::string> tangentSweep;
    std::vector<std::string> tangentSweepFailures;
    const std::array<std::pair<std::string,const WorkRecord*>,2> sweepCases{{
        {"compression_densification",&compression},{"mixed_mode",&mixed}}};
    const std::array<double,8> sweepSteps{{1.0e-4,3.0e-5,1.0e-5,3.0e-6,
                                            1.0e-6,3.0e-7,1.0e-7,3.0e-8}};
    const std::array<double,3> projectionTolerances{{2.0e-10,2.0e-12,2.0e-14}};
    for(const auto& [caseName,history]:sweepCases) {
        Matrix9 sweepDirection{};
        if(caseName=="mixed_mode") {
            sweepDirection[0]=1.0;sweepDirection[1]=0.5;
        } else sweepDirection[0]=1.0;
        for(const double tolerance:projectionTolerances)
            for(const double step:sweepSteps) {
                double plusResidual=0.0,minusResidual=0.0;
                double error=std::numeric_limits<double>::quiet_NaN();
                try {
                    error=reprojectedTangentError(compiled,history->maximumF,
                        history->acceptedAtMaximum,history->candidateAtMaximum,
                        sweepDirection,0.002,step,tolerance,&plusResidual,&minusResidual);
                } catch(const std::exception& e) {
                    tangentSweepFailures.push_back(caseName+","+number(step)+","+
                        number(tolerance)+",nonconverged,"+e.what());
                    break;
                }
                tangentSweep.push_back(caseName+","+number(step)+","+
                    number(tolerance)+","+number(error)+","+
                    number(plusResidual)+","+number(minusResidual));
            }
    }
    std::vector<std::string> compressionBranchRows;
    for(const double step:{1.0e-6,3.0e-8}) {
        const double projectionTolerance=2.0e-12;
        const auto base=project(compiled,compression.maximumF,
            compression.acceptedAtMaximum,0.002,projectionTolerance);
        Matrix9 plusF=compression.maximumF,minusF=compression.maximumF;
        plusF[0]+=step;minusF[0]-=step;
        const auto plus=project(compiled,plusF,compression.acceptedAtMaximum,
                                0.002,projectionTolerance);
        const auto minus=project(compiled,minusF,compression.acceptedAtMaximum,
                                 0.002,projectionTolerance);
        const auto plusStress=stress(compiled,plusF,compression.acceptedAtMaximum,
                                    plus.state,0.002);
        const auto minusStress=stress(compiled,minusF,compression.acceptedAtMaximum,
                                     minus.state,0.002);
        Matrix9 compressionDirection{};compressionDirection[0]=1.0;
        const auto ad=consistentTangentAction(compiled,compression.maximumF,
            compression.acceptedAtMaximum,base.state,compressionDirection,0.002);
        unsigned worstComponent=0u;double worstDifference=-1.0;
        for(unsigned component=0;component<9u;++component) {
            const double fd=(plusStress[component]-minusStress[component])/(2.0*step);
            const double difference=std::abs(ad[component]-fd);
            if(difference>worstDifference) {worstDifference=difference;worstComponent=component;}
        }
        const auto lambda=stateIndex(compiled.material,"accumulated_multiplier");
        const auto densification0=stateIndex(compiled.material,"overlay_densification_0");
        const auto damage0=stateIndex(compiled.material,"overlay_fracture_excess_0");
        require(hillYieldGap(compiled.material,plus.state)<0.0&&
                hillYieldGap(compiled.material,minus.state)<0.0&&
                plus.state[lambda]>compression.acceptedAtMaximum[lambda]&&
                minus.state[lambda]>compression.acceptedAtMaximum[lambda]&&
                plus.state[densification0]>0.0&&minus.state[densification0]>0.0,
                "compression finite-difference probes changed the active plastic/densification branch");
        compressionBranchRows.push_back(number(step)+","+
            number(hillYieldGap(compiled.material,plus.state))+","+
            number(hillYieldGap(compiled.material,minus.state))+","+
            number(plus.state[lambda]-compression.acceptedAtMaximum[lambda])+","+
            number(minus.state[lambda]-compression.acceptedAtMaximum[lambda])+","+
            number(plus.state[densification0])+","+number(minus.state[densification0])+","+
            number(plus.state[damage0])+","+number(minus.state[damage0])+","+
            std::to_string(worstComponent)+","+number(worstDifference));
    }
    std::vector<std::string> mixedBranchRows;
    for(const double step:{3.0e-7,3.0e-8}) {
        Matrix9 plusF=mixed.maximumF,minusF=mixed.maximumF;
        plusF[0]+=step*mixedDirection[0];plusF[1]+=step*mixedDirection[1];
        minusF[0]-=step*mixedDirection[0];minusF[1]-=step*mixedDirection[1];
        const auto plus=project(mixedCompiled.program,plusF,mixed.acceptedAtMaximum,
                                0.002,tightProjectionTolerance);
        const auto minus=project(mixedCompiled.program,minusF,mixed.acceptedAtMaximum,
                                 0.002,tightProjectionTolerance);
        const auto pMultiplier=plus.state[stateIndex(mixedCompiled.program.material,
            "accumulated_multiplier")];
        const auto mMultiplier=minus.state[stateIndex(mixedCompiled.program.material,
            "accumulated_multiplier")];
        const auto pNormal=plus.state[activeDamage];
        const auto mNormal=minus.state[activeDamage];
        const auto pShear=plus.state[shearDamage];
        const auto mShear=minus.state[shearDamage];
        require(pMultiplier<1.0e-8&&mMultiplier<1.0e-8&&
                pNormal>1.0e-5&&mNormal>1.0e-5&&pShear>1.0e-5&&mShear>1.0e-5,
                "mixed-mode finite-difference probes changed the active smooth damage/plastic branch");
        mixedBranchRows.push_back(number(step)+","+number(pMultiplier)+","+
            number(mMultiplier)+","+number(pNormal)+","+number(mNormal)+","+
            number(pShear)+","+number(mShear));
    }

    std::cout<<std::setprecision(17)
             <<"retained_overlay_tangent_relative_error="<<tangentError<<'\n'
             <<"compression_densification_tangent_relative_error="<<compressionTangentError<<'\n'
             <<"plastic_hardening_tangent_relative_error="<<plasticTangentError<<'\n'
             <<"mixed_mode_tangent_relative_error="<<mixedTangentError<<'\n'
             <<"plastic_multiplier_at_peak="<<plasticCycle.stateAtMaximum[multiplier]<<'\n'
             <<"md_hardening_history_at_peak="<<plasticCycle.stateAtMaximum[mdHardening]<<'\n'
             <<"plastic_closed_cycle_dissipation_j="<<plasticDissipation<<'\n'
             <<"closed_cycle_damage_dissipation_j="<<damageIncrement<<'\n'
             <<"compression_cycle_work_j="<<compression.externalWorkJ<<'\n'
             <<"compression_final_energy_j="<<compression.finalEnergyJ<<'\n'
             <<"compression_candidate_net_irreversible_work_j="<<compressionDissipation<<'\n'
             <<"compression_no_added_densification_control_work_j="<<
                compressionControl.externalWorkJ<<'\n'
             <<"compression_no_added_densification_control_recoverable_energy_j="<<
                compressionControl.finalEnergyJ<<'\n'
             <<"compression_no_added_densification_control_net_work_j="<<
                compressionControlDissipation<<'\n'
             <<"compression_minus_no_added_densification_control_j="<<
                densificationDissipation<<'\n'
             <<"compression_source_hill_work_j="<<sourceHillCompression.externalWorkJ<<'\n'
             <<"compression_source_hill_recoverable_energy_j="<<
                sourceHillCompression.finalEnergyJ<<'\n'
             <<"compression_source_hill_net_work_j="<<
                sourceHillCompressionDissipation<<'\n'
             <<"compression_candidate_minus_source_hill_net_work_j="<<
                densificationMinusSourceHill<<'\n'
             <<"compression_hill_yield_strength_start_pa="<<
                initialDensificationYieldStrength<<'\n'
             <<"compression_hill_yield_strength_peak_pa="<<
                peakDensificationYieldStrength<<'\n'
             <<"compression_hill_yield_strength_final_pa="<<
                finalDensificationYieldStrength<<'\n'
             <<"compression_accumulated_multiplier_peak="<<
                compression.stateAtMaximum[compressionMultiplier]<<'\n'
             <<"compression_accumulated_multiplier_final="<<
                compression.finalState[compressionMultiplier]<<'\n'
             <<"compression_md_plastic_history_peak="<<
                compression.stateAtMaximum[compressionHardening[0]]<<'\n'
             <<"compression_md_plastic_history_final="<<
                compression.finalState[compressionHardening[0]]<<'\n'
             <<"compression_maximum_fracture_damage_history="<<
                maximumCompressionDamage<<'\n'
             <<"source_hill_plastic_cycle_dissipation_j="<<sourcePlasticDissipation<<'\n'
             <<"overlay_hardening_minus_source_plastic_dissipation_j="<<
                (plasticDissipation-sourcePlasticDissipation)<<'\n'
             <<"matched_no_damage_elastic_cycle_dissipation_j="<<controlDissipation<<'\n'
             <<"densification_history_at_peak="<<compression.stateAtMaximum[densification]<<'\n'
             <<"selected_plane_gc_area_j="<<expectedGcArea<<'\n'
             <<"dissipation_to_gc_area_ratio="<<damageIncrement/expectedGcArea<<'\n'
             <<"terminal_mode0_damage_excess="<<terminalDamageExcess<<'\n'
             <<"terminal_mode0_softening_tail_fraction="<<terminalSofteningTailFraction<<'\n'
             <<"damage_excess_short_dt="<<shortRateStep.state[activeDamage]<<'\n'
             <<"damage_excess_long_dt="<<longRateStep.state[activeDamage]<<'\n'
             <<"refined_damage_work_per_area_j_m2="<<finePerArea<<'\n'
             <<"coarse_damage_work_per_area_j_m2="<<coarsePerArea<<'\n'
             <<"refinement_relative_difference="<<refinementDifference<<'\n'
             <<"refinement_screen_2pct="<<(refinementDifference<=0.02?"pass":"fail")<<'\n'
             <<"oblique_plane_closed_cycle_work_j="<<obliqueWork<<'\n'
             <<"mixed_mode_damage_work_j="<<mixedDamageWork<<'\n'
             <<"mixed_mode_gc_budget_j="<<mixedGcBudget<<'\n'
             <<"mixed_mode_budget_ratio="<<mixedBudgetRatio<<'\n'
             <<"mixed_mode_active_channels="<<mixedActiveModes<<'\n'
             <<"gc_area_2pct_gate="<<
                (std::abs(damageIncrement/expectedGcArea-1.0)<=0.02?"pass":"fail")<<'\n'
             <<"damage_work_ratio_60_steps="<<damageIncrement/expectedGcArea<<'\n';
    for(const auto& [steps,ratio]:workQuadrature)
        std::cout<<"damage_work_ratio_"<<steps<<"_steps="<<ratio<<'\n';
    for(const auto& row:tangentSweep)
        std::cout<<"tangent_sweep="<<row<<'\n';
    for(const auto& row:tangentSweepFailures)
        std::cout<<"tangent_sweep_failure="<<row<<'\n';
    for(const auto& row:compressionBranchRows)
        std::cout<<"compression_branch_indicator="<<row<<'\n';
    std::cout<<"upper_range_120_step_gc_ratio="<<upperRangeGcRatio<<'\n'
             <<"full_softening_fixture_strength_pa=2e6_synthetic_only\n"
             <<"full_softening_excess_over_width="<<
                fullOpeningExcess/fullOpeningSofteningWidth<<'\n'
             <<"full_softening_damage_work_ratio_120_steps="<<fullOpeningRatio120<<'\n'
             <<"full_softening_damage_work_ratio_240_steps="<<fullOpeningRatio240<<'\n'
             <<"full_softening_damage_work_j_120_steps="<<fullOpeningWork120<<'\n'
             <<"full_softening_damage_work_j_240_steps="<<fullOpeningWork240<<'\n'
             <<"rotated_full_softening_damage_work_j="<<rotatedEnergyWork<<'\n'
             <<"rotated_full_softening_gc_area_j="<<rotatedGcArea<<'\n'
             <<"rotated_full_softening_gc_area_ratio="<<rotatedEnergyRatio<<'\n'
             <<"reversed_winding_full_softening_damage_work_j="<<reversedEnergyWork<<'\n'
             <<"reversed_winding_full_softening_gc_area_j="<<reversedGcArea<<'\n'
             <<"reversed_winding_full_softening_gc_area_ratio="<<reversedEnergyRatio<<'\n'
             <<"full_softening_maximum_hill_multiplier="<<
                fullOpening240.stateAtMaximum[fullOpeningMultiplier]<<'\n'
             <<"full_softening_work_convergence_relative_difference="<<
                std::abs(fullOpeningWork240-fullOpeningWork120)/fullOpeningWork240<<'\n'
             <<"compression_gate_error_h1e-6="<<compressionGateError1<<'\n'
             <<"compression_gate_error_h3e-8="<<compressionGateError2<<'\n'
             <<"mixed_gate_error_h3e-7="<<mixedGateError1<<'\n'
             <<"mixed_gate_error_h3e-8="<<mixedGateError2<<'\n';
    for(const auto& row:mixedBranchRows)
        std::cout<<"mixed_branch_indicator="<<row<<'\n';
    std::cout
             <<"dissipation_status=measured_cpu_synthetic_candidate_not_calibration\n";
}
} // namespace

int main(int argc,char** argv) {
    try {
        const std::filesystem::path materials=argc>1?argv[1]:
            std::filesystem::path(__FILE__).parent_path().parent_path()/"materials";
        checkComposition(materials);
        checkExactPool(materials);
        checkFailClosed(materials);
        checkFullFaceClassificationBoundary();
        checkHistoryTangentAndWork(materials);
        std::cout << "cardboard_retained_matrix_damage_cpu_passed\n"
                  << "matrix=Haj-Ali liner/NSSC source program and density retained\n"
                  << "fracture/rate/densification values=explicit synthetic test bindings only\n"
                  << "calibration=not_claimed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
