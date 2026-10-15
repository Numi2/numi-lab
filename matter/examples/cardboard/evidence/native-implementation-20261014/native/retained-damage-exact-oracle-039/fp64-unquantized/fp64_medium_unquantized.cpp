#include "cardboard_retained_matrix_damage.hpp"
#include "cardboard_retained_damage_fp64_oracle.hpp"
#include "numi/matter/detail.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using namespace numi::matter;
using namespace numi::cardboard::retained_damage;
namespace fp64 = numi::cardboard::retained_damage::fp64;
using Matrix = std::array<double, 9>;
using State = std::vector<double>;
struct PairInput { double epsilon; Matrix plus; Matrix minus; };
static constexpr const char* kMaterialPath = "/Users/home/cardboard-implementation-20261014/snapshots/damage-oracle-025/matter/materials/hajali2009_medium_hill_ideal.nmatter";
static constexpr const char* kMaterialSha = "ecfd9de3f4550a6a269d7a6559781b16e15316cf3d587093f2e11a2e6942a8e8";
static constexpr const char* kCsvSha = "83f5778db797f0fb9996416848abb91c18f9102f4c3707aea9ea267b1eeb0d82";
static constexpr const char* kSyntheticSetSha = "a7e7e865a24da8486c047325cf1766a9f7e8fdb2921b80b89b6e4b3c27ce09aa";
static constexpr const char* kRegion = "medium";
static const std::array<float, 9> kF32 = {{1.0038000300000001,0,0,0,1,0,0,0,1}};
static const std::array<float, 9> kDirection32 = {{1,0.119999997,0,0,0.15000000599999999,0,0,0,0}};
static const std::array<float, 30> kAccepted32 = {{0.00051405147000000003,0.000932644936,-0.00144669635,0,0,0,1.1708396700000001,-0.084095589799999995,0.0027950461999999999,0,0,0,0.00051470549099999995,0.00051385158400000001,0.000932444935,0.0014464963499999999,0,0,0,7.1490484599999996e-09,4.8406225700000003e-09,3.6593608100000002e-09,0.0012835721699999999,0,0,0,0,0,0,0}};
static const std::vector<PairInput> kPairs = {{0.001,{{1.0048000800000001,0.000119999997,0,0,1.0001499700000001,0,0,0,1}},{{1.00279999,-0.000119999997,0,0,0.99984997499999995,0,0,0,1}}},{0.00029999999999999997,{{1.00410008,3.5999997600000002e-05,0,0,1.00004494,0,0,0,1}},{{1.00349998,-3.5999997600000002e-05,0,0,0.99995499799999998,0,0,0,1}}},{0.0001,{{1.0039000499999999,1.1999999500000001e-05,0,0,1.00001502,0,0,0,1}},{{1.0037000199999999,-1.1999999500000001e-05,0,0,0.99998498000000002,0,0,0,1}}},{3.0000000000000001e-05,{{1.00383008,3.5999998999999999e-06,0,0,1.00000453,0,0,0,1}},{{1.0037699899999999,-3.5999998999999999e-06,0,0,0.99999552999999997,0,0,0,1}}},{1.0000000000000001e-05,{{1.00381005,1.1999999299999999e-06,0,0,1.0000015499999999,0,0,0,1}},{{1.0037900200000001,-1.1999999299999999e-06,0,0,0.99999850999999995,0,0,0,1}}},{3.0000000000000001e-06,{{1.0038030099999999,3.60000001e-07,0,0,1.00000048,0,0,0,1}},{{1.00379705,-3.60000001e-07,0,0,0.99999952299999995,0,0,0,1}}},{9.9999999999999995e-07,{{1.00380099,1.19999996e-07,0,0,1.0000001199999999,0,0,0,1}},{{1.0037990800000001,-1.19999996e-07,0,0,0.99999982099999996,0,0,0,1}}},{2.9999999999999999e-07,{{1.0038003900000001,3.5999999400000001e-08,0,0,1,0,0,0,1}},{{1.00379968,-3.5999999400000001e-08,0,0,0.99999994000000003,0,0,0,1}}}};
static const std::vector<double> kExactEpsilons = {0.001,0.00050000000000000001,0.00029999999999999997,0.00014999999999999999,0.0001,5.0000000000000002e-05,3.0000000000000001e-05,1.5e-05,1.0000000000000001e-05,5.0000000000000004e-06,3.0000000000000001e-06,1.5e-06,9.9999999999999995e-07,4.9999999999999998e-07,2.9999999999999999e-07,2.4999999999999999e-07,1.4999999999999999e-07};
static constexpr double kDt = static_cast<double>(static_cast<float>(0.002));
static constexpr double kProjectionTolerance = 2.0e-13;

static Matrix asDouble(const std::array<float, 9>& input) {
    Matrix output{};
    for (std::size_t i = 0; i < output.size(); ++i) output[i] = input[i];
    return output;
}
static double norm(const Matrix& a) {
    double sum = 0.0;
    for (double value : a) sum += value * value;
    return std::sqrt(sum);
}
static double relativeError(const Matrix& a, const Matrix& b) {
    Matrix difference{};
    for (std::size_t i = 0; i < a.size(); ++i) difference[i] = a[i] - b[i];
    return norm(difference) / std::max(std::numeric_limits<double>::min(), norm(b));
}
static Matrix subtract(const Matrix& a, const Matrix& b) {
    Matrix out{}; for (std::size_t i = 0; i < out.size(); ++i) out[i] = a[i] - b[i]; return out;
}
static Matrix addScaled(const Matrix& a, const Matrix& b, double scale) {
    Matrix out{}; for (std::size_t i = 0; i < out.size(); ++i) out[i] = a[i] + scale * b[i]; return out;
}
static Matrix scale(const Matrix& a, double s) {
    Matrix out{}; for (std::size_t i = 0; i < out.size(); ++i) out[i] = s * a[i]; return out;
}
static Matrix divide(const Matrix& a, double d) { return scale(a, 1.0 / d); }
static void writeMatrix(const Matrix& x) {
    std::cout << '['; for (std::size_t i = 0; i < x.size(); ++i) { if (i) std::cout << ','; std::cout << x[i]; } std::cout << ']';
}
static void writeState(const State& x) {
    std::cout << '['; for (std::size_t i = 0; i < x.size(); ++i) { if (i) std::cout << ','; std::cout << x[i]; } std::cout << ']';
}
static double stateValue(const MaterialProgram& material, const State& state, const std::string& name) {
    for (std::size_t i = 0; i < material.internalState.size(); ++i)
        if (material.internalState[i].name == name) return state.at(i);
    throw std::runtime_error("missing state " + name);
}
static double paramValue(const MaterialProgram& material, const std::string& name) {
    for (const auto& p : material.parameters) if (p.name == name) return static_cast<float>(p.defaultValue);
    throw std::runtime_error("missing parameter " + name);
}
static double yieldGap(const MaterialProgram& m, const State& s) {
    const double s11=stateValue(m,s,"sbar11"), s22=stateValue(m,s,"sbar22"), s33=stateValue(m,s,"sbar33");
    const double s23=stateValue(m,s,"sbar23"), s13=stateValue(m,s,"sbar13"), s12=stateValue(m,s,"sbar12");
    const double q2 = paramValue(m,"hill_F")*std::pow(s22-s33,2.0) +
        paramValue(m,"hill_G")*std::pow(s33-s11,2.0) + paramValue(m,"hill_H")*std::pow(s11-s22,2.0) +
        2.0*paramValue(m,"hill_L")*s23*s23 + 2.0*paramValue(m,"hill_M")*s13*s13 +
        2.0*paramValue(m,"hill_N")*s12*s12 + paramValue(m,"q_floor")*paramValue(m,"q_floor");
    double hardening=0.0;
    for (unsigned i=0;i<6;++i) hardening += paramValue(m,"overlay_plastic_hardening_"+std::to_string(i))*stateValue(m,s,"overlay_plastic_history_"+std::to_string(i));
    for (unsigned i=0;i<3;++i) hardening += paramValue(m,"overlay_densification_modulus_pa_axis"+std::to_string(i))*stateValue(m,s,"overlay_densification_"+std::to_string(i));
    return 1.0 + hardening/paramValue(m,"sigma_ref") - std::sqrt(std::max(0.0,q2));
}
static double lambdaValue(const MaterialProgram& m, const State& s) { return stateValue(m,s,"accumulated_multiplier"); }
static double projection(const ConstitutiveProgram& p, const Matrix& f, const State& accepted, Matrix* stress, Matrix* tangent, const Matrix* direction, State* state, double* gap) {
    const auto result=fp64::project(p,f,accepted,kDt,kProjectionTolerance);
    if (stress) *stress=fp64::stress(p,f,accepted,result.state,kDt);
    if (tangent) *tangent=fp64::tangentAction(p,f,accepted,result.state,*direction,kDt);
    if (state) *state=result.state;
    if (gap) *gap=result.residualInf;
    return result.residualInf;
}
static ExplicitDamageSet makeSet() {
    ExplicitDamageSet set;
    set.setId="retained_matrix_native_instrument_synthetic_v1";
    set.sha256=kSyntheticSetSha;
    set.evidenceClass="synthetic_instrument_only_not_calibration";
    set.plasticHardeningPa={1.3e8,2.4e7,0.0,0.0,0.0,0.0};
    set.hardeningBinding={"Haj-Ali 2009 fitted-tension-node dataset; overlay coefficients are synthetic",
        "haj_ali_2009_bilinear_tension_fit_nodes.csv",kCsvSha,"instrument fixture values are not the published fit values"};
    for (auto& mode:set.fractureModes) mode={1.0e8,1.0e6,0.04};
    set.fractureModes[0]={2.0e6,0.4,0.004};
    set.fractureBinding={"native_instrument_synthetic_fracture_parameters",std::string(64,'a'),"synthetic_instrument_only_not_calibration"};
    set.rateBinding={"native_instrument_synthetic_rate_parameters",std::string(64,'b'),"synthetic_instrument_only_not_calibration"};
    set.densificationModulusPa={4.0e7,3.0e7,2.0e7};
    set.densificationOnsetStrain={0.08,0.10,0.12};
    set.densificationRateTimeConstantSeconds={0.04,0.06,0.08};
    set.densificationBinding={"native_instrument_synthetic_compaction_parameters",std::string(64,'c'),"synthetic_instrument_only_not_calibration"};
    set.damageCap=1.0; set.smoothStrainWidth=1.0e-7; return set;
}
static ElementRequest makeRequest(const MaterialProgram& source) {
    ElementRequest r; r.elementIndex=0; r.sourceMaterialIndex=0; r.regionId=kRegion;
    r.sourceMatrix=source; r.sourceDensityKgPerM3=paramValue(source,"density");
    r.sourceBinding={source.name,kMaterialSha,"exact_local_material_file_sha256"};
    r.referenceVerticesM={{{0.0,0.0,0.0},{0.001,0.0,0.0},{0.0,0.001,0.0},{0.0,0.0,0.001}}};
    r.selectedMaterialPlanes={{{{1,0,0},numi::cardboard::FractureNormalFrame::Reference,0.0,{},1.0e-12},
        {{0,1,0},numi::cardboard::FractureNormalFrame::Reference,0.0,{},1.0e-12},
        {{0,0,1},numi::cardboard::FractureNormalFrame::Reference,0.0,{},1.0e-12},
        {{1,1,1},numi::cardboard::FractureNormalFrame::Reference,0.0,{},1.0e-12}}};
    r.explicitSet=makeSet(); return r;
}
int main() {
    std::cout << std::setprecision(17);
    const auto parsed=parseMatterFile(kMaterialPath);
    if (!parsed.succeeded()) throw std::runtime_error("material parse failed");
    auto request=makeRequest(parsed.material);
    auto built=buildElementProgram(request);
    const auto compiled=numi::matter::detail::compileConstitutive(built.program,NM_EXPRESSION_STACK_CAPACITY);
    if (!compiled.succeeded()) throw std::runtime_error("source material compile failed");
    Matrix f=asDouble(kF32), direction=asDouble(kDirection32);
    State accepted(kAccepted32.size()); for (std::size_t i=0;i<accepted.size();++i) accepted[i]=kAccepted32[i];
    Matrix centerStress{}, centerTangent{}; State centerState; double centerResidual=0.0;
    projection(compiled.program,f,accepted,&centerStress,&centerTangent,&direction,&centerState,&centerResidual);
    std::cout << "{\"region\":\"" << kRegion << "\",\"material_name\":\"" << built.program.name
              << "\",\"center\":{";
    std::cout << "\"residual_inf\":" << centerResidual << ",\"stress_pa\":"; writeMatrix(centerStress);
    std::cout << ",\"tangent_pa\":"; writeMatrix(centerTangent); std::cout << ",\"state\":"; writeState(centerState);
    std::cout << ",\"hill_yield_gap\":" << yieldGap(built.program,centerState)
              << ",\"lambda\":" << lambdaValue(built.program,centerState) << "},\"representable_pairs\":[";
    for (std::size_t i=0;i<kPairs.size();++i) {
        const auto& pair=kPairs[i]; Matrix sp{},sm{},midTangent{}; State xp,xm,xmid; double rp=0,rm=0,rmid=0;
        projection(compiled.program,pair.plus,accepted,&sp,nullptr,nullptr,&xp,&rp);
        projection(compiled.program,pair.minus,accepted,&sm,nullptr,nullptr,&xm,&rm);
        const Matrix mid=scale(addScaled(pair.plus,pair.minus,1.0),0.5);
        const Matrix realized=divide(subtract(pair.plus,pair.minus),2.0*pair.epsilon);
        projection(compiled.program,mid,accepted,nullptr,&midTangent,&realized,&xmid,&rmid);
        Matrix fd=divide(subtract(sp,sm),2.0*pair.epsilon);
        Matrix onePlusDir=divide(subtract(pair.plus,f),pair.epsilon);
        Matrix oneMinusDir=divide(subtract(f,pair.minus),pair.epsilon);
        Matrix onePlusStress{},oneMinusStress{},tanPlus{},tanMinus{}; State xop,xom; double rop=0,rom=0;
        projection(compiled.program,pair.plus,accepted,&onePlusStress,nullptr,nullptr,&xop,&rop);
        projection(compiled.program,pair.minus,accepted,&oneMinusStress,nullptr,nullptr,&xom,&rom);
        projection(compiled.program,f,accepted,nullptr,&tanPlus,&onePlusDir,&xop,&rop);
        projection(compiled.program,f,accepted,nullptr,&tanMinus,&oneMinusDir,&xom,&rom);
        const Matrix fdPlus=divide(subtract(sp,centerStress),pair.epsilon);
        const Matrix fdMinus=divide(subtract(centerStress,sm),pair.epsilon);
        if(i) std::cout << ',';
        std::cout << "{\"epsilon\":" << pair.epsilon << ",\"plus_residual\":" << rp << ",\"minus_residual\":" << rm
            << ",\"plus_yield_gap\":" << yieldGap(built.program,xp) << ",\"minus_yield_gap\":" << yieldGap(built.program,xm)
            << ",\"plus_lambda\":" << lambdaValue(built.program,xp) << ",\"minus_lambda\":" << lambdaValue(built.program,xm)
            << ",\"midpoint_residual\":" << rmid << ",\"midpoint_yield_gap\":" << yieldGap(built.program,xmid)
            << ",\"midpoint_lambda\":" << lambdaValue(built.program,xmid) << ",\"realized_direction\":"; writeMatrix(realized);
        std::cout << ",\"midpoint_tangent_pa\":"; writeMatrix(midTangent);
        std::cout << ",\"fp64_fd\":"; writeMatrix(fd);
        std::cout << ",\"fp64_fd_vs_midpoint_ad_relative\":" << relativeError(fd,midTangent)
            << ",\"one_sided_plus_fd\":"; writeMatrix(fdPlus);
        std::cout << ",\"one_sided_plus_ad\":"; writeMatrix(tanPlus);
        std::cout << ",\"one_sided_plus_error\":" << relativeError(fdPlus,tanPlus)
            << ",\"one_sided_minus_fd\":"; writeMatrix(fdMinus);
        std::cout << ",\"one_sided_minus_ad\":"; writeMatrix(tanMinus);
        std::cout << ",\"one_sided_minus_error\":" << relativeError(fdMinus,tanMinus) << '}';
    }
    std::cout << "],\"unquantized_fp64_sweep\":[";
    std::vector<Matrix> fdValues; std::vector<State> plusStates,minusStates; std::vector<double> residualsPlus,residualsMinus;
    fdValues.reserve(kExactEpsilons.size());
    for (double eps:kExactEpsilons) {
        const Matrix fp=addScaled(f,direction,eps), fm=addScaled(f,direction,-eps);
        Matrix sp{},sm{}; State xp,xm; double rp=0,rm=0;
        projection(compiled.program,fp,accepted,&sp,nullptr,nullptr,&xp,&rp);
        projection(compiled.program,fm,accepted,&sm,nullptr,nullptr,&xm,&rm);
        fdValues.push_back(divide(subtract(sp,sm),2.0*eps)); plusStates.push_back(xp); minusStates.push_back(xm);
        residualsPlus.push_back(rp); residualsMinus.push_back(rm);
    }
    for(std::size_t i=0;i<kExactEpsilons.size();++i) {
        const double eps=kExactEpsilons[i]; const Matrix& fd=fdValues[i];
        const double err=relativeError(fd,centerTangent);
        if(i) std::cout << ',';
        std::cout << "{\"epsilon\":" << eps << ",\"plus_residual\":" << residualsPlus[i]
            << ",\"minus_residual\":" << residualsMinus[i]
            << ",\"plus_yield_gap\":" << yieldGap(built.program,plusStates[i])
            << ",\"minus_yield_gap\":" << yieldGap(built.program,minusStates[i])
            << ",\"plus_lambda\":" << lambdaValue(built.program,plusStates[i])
            << ",\"minus_lambda\":" << lambdaValue(built.program,minusStates[i])
            << ",\"fd\":"; writeMatrix(fd);
        std::cout << ",\"fd_vs_ad_relative\":" << err;
        const auto halfIt=std::find_if(kExactEpsilons.begin(),kExactEpsilons.end(),
            [&](double candidate){return std::abs(candidate-0.5*eps)<eps*1.0e-12;});
        if(halfIt!=kExactEpsilons.end()) {
            const auto halfIndex=static_cast<std::size_t>(std::distance(kExactEpsilons.begin(),halfIt));
            const Matrix& half=fdValues[halfIndex]; Matrix rich{};
            for(std::size_t k=0;k<rich.size();++k) rich[k]=(4.0*half[k]-fd[k])/3.0;
            std::cout << ",\"half_step_spread_relative\":" << relativeError(fd,half)
                << ",\"richardson_fd\":"; writeMatrix(rich);
            std::cout << ",\"richardson_vs_ad_relative\":" << relativeError(rich,centerTangent);
        } else {
            std::cout << ",\"half_step_spread_relative\":null,\"richardson_fd\":null,\"richardson_vs_ad_relative\":null";
        }
        std::cout << '}';
    }
    std::cout << "]}\n";
}
