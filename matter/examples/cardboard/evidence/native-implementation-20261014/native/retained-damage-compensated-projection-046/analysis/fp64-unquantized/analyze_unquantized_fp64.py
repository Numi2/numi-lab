#!/usr/bin/env python3
"""CPU-only exact-FP64 constitutive FD sweep for native tangent captures."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path('/Users/home/cardboard-implementation-20261014/native/retained-damage-compensated-projection-046')
HERE = pathlib.Path(__file__).resolve().parent
SNAPSHOT = pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/generic-damage-regional-block-032')
BUILD = pathlib.Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-032')
LIB = BUILD / 'libnumi_matter_compiler.a'
INCLUDES = [SNAPSHOT / 'matter' / 'include', SNAPSHOT / 'include']
CAPTURES = sorted((ROOT / 'capture').glob('*.tangent-capture.json'))
FP64_TOLERANCE = 2.0e-13
EPSILONS = [1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 5.0e-7, 3.0e-7]

DRIVER = r'''#include "cardboard_retained_matrix_damage.hpp"
#include "cardboard_retained_damage_fp64_oracle.hpp"
#include "numi/matter/detail.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
using namespace numi::matter;
using namespace numi::cardboard::retained_damage;
namespace fp64 = numi::cardboard::retained_damage::fp64;
using Matrix = std::array<double, 9>;
using State = std::vector<double>;
struct PairInput { double epsilon; Matrix plus; Matrix minus; };
static constexpr const char* kMaterialPath = @MATERIAL_PATH@;
static constexpr const char* kMaterialSha = @MATERIAL_SHA@;
static constexpr const char* kCsvSha = @CSV_SHA@;
static constexpr const char* kSyntheticSetSha = @SET_SHA@;
static constexpr const char* kRegion = @REGION@;
static const std::array<float, 9> kF32 = @F32@;
static const std::array<float, 9> kDirection32 = @DIRECTION32@;
static const std::array<float, 30> kAccepted32 = @ACCEPTED32@;
static const std::vector<PairInput> kPairs = @PAIR_INPUTS@;
static const std::vector<double> kExactEpsilons = @EXACT_EPSILONS@;
static const std::vector<std::pair<std::string,float>> kCookedParameters = @COOKED_PARAMETERS@;
static constexpr std::uint64_t kExpectedConstitutiveFingerprint = @CONSTITUTIVE_FINGERPRINT@;
static constexpr double kDt = static_cast<double>(static_cast<float>(0.002));
static constexpr double kProjectionTolerance = @PROJECTION_TOLERANCE@;

static CompiledWorld compileInstrumentWorld(const MaterialProgram& material) {
    WorldSource source;
    source.materials = {material};
    source.gravity = {0.0, 0.0, 0.0};
    source.environmentCount = 1u;
    source.frameTimestep = 2.0e-3;
    ObjectSource object;
    object.name = "retained_matrix_constitutive_instrument";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.femNodes = {{{0.0, 0.0, 0.0}, {0.001, 0.0, 0.0},
                        {0.0, 0.001, 0.0}, {0.0, 0.0, 0.001}}};
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    object.characteristicLength = 0.001;
    source.objects = {object};
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const auto result = compileWorld(source, options);
    if (!result.succeeded()) throw std::runtime_error("instrument world compile failed");
    return result.world;
}
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
    for(auto& p:built.program.parameters) { bool found=false; for(const auto& cp:kCookedParameters) if(p.name==cp.first) { p.defaultValue=static_cast<double>(cp.second); found=true; break; } if(!found) throw std::runtime_error("missing captured cooked parameter: "+p.name); }
    built.program.genericCompensatedImplicitProjection=true;
    const auto world=compileInstrumentWorld(built.program);
    const auto& compiled=world.constitutive.at(0u);
    if(compiled.fingerprint!=kExpectedConstitutiveFingerprint) throw std::runtime_error("reconstructed constitutive fingerprint mismatch: "+std::to_string(compiled.fingerprint)+" expected "+std::to_string(kExpectedConstitutiveFingerprint));
    Matrix f=asDouble(kF32), direction=asDouble(kDirection32);
    State accepted(kAccepted32.size()); for (std::size_t i=0;i<accepted.size();++i) accepted[i]=kAccepted32[i];
    Matrix centerStress{}, centerTangent{}; State centerState; double centerResidual=0.0;
    projection(compiled,f,accepted,&centerStress,&centerTangent,&direction,&centerState,&centerResidual);
    std::cout << "{\"region\":\"" << kRegion << "\",\"material_name\":\"" << built.program.name
              << "\",\"center\":{";
    std::cout << "\"residual_inf\":" << centerResidual << ",\"stress_pa\":"; writeMatrix(centerStress);
    std::cout << ",\"tangent_pa\":"; writeMatrix(centerTangent); std::cout << ",\"state\":"; writeState(centerState);
    std::cout << ",\"hill_yield_gap\":" << yieldGap(built.program,centerState)
              << ",\"lambda\":" << lambdaValue(built.program,centerState) << "},\"representable_pairs\":[";
    for (std::size_t i=0;i<kPairs.size();++i) {
        const auto& pair=kPairs[i]; Matrix sp{},sm{},midTangent{}; State xp,xm,xmid; double rp=0,rm=0,rmid=0;
        projection(compiled,pair.plus,accepted,&sp,nullptr,nullptr,&xp,&rp);
        projection(compiled,pair.minus,accepted,&sm,nullptr,nullptr,&xm,&rm);
        const Matrix mid=scale(addScaled(pair.plus,pair.minus,1.0),0.5);
        const Matrix realized=divide(subtract(pair.plus,pair.minus),2.0*pair.epsilon);
        projection(compiled,mid,accepted,nullptr,&midTangent,&realized,&xmid,&rmid);
        Matrix fd=divide(subtract(sp,sm),2.0*pair.epsilon);
        Matrix onePlusDir=divide(subtract(pair.plus,f),pair.epsilon);
        Matrix oneMinusDir=divide(subtract(f,pair.minus),pair.epsilon);
        Matrix onePlusStress{},oneMinusStress{},tanPlus{},tanMinus{}; State xop,xom; double rop=0,rom=0;
        projection(compiled,pair.plus,accepted,&onePlusStress,nullptr,nullptr,&xop,&rop);
        projection(compiled,pair.minus,accepted,&oneMinusStress,nullptr,nullptr,&xom,&rom);
        projection(compiled,f,accepted,nullptr,&tanPlus,&onePlusDir,&xop,&rop);
        projection(compiled,f,accepted,nullptr,&tanMinus,&oneMinusDir,&xom,&rom);
        const Matrix fdPlus=divide(subtract(sp,centerStress),pair.epsilon);
        const Matrix fdMinus=divide(subtract(centerStress,sm),pair.epsilon);
        if(i) std::cout << ',';
        std::cout << "{\"epsilon\":" << pair.epsilon << ",\"plus_residual\":" << rp << ",\"minus_residual\":" << rm
            << ",\"plus_yield_gap\":" << yieldGap(built.program,xp) << ",\"minus_yield_gap\":" << yieldGap(built.program,xm) << ",\"plus_state\":";
        writeState(xp); std::cout << ",\"minus_state\":"; writeState(xm);
        std::cout << ",\"plus_lambda\":" << lambdaValue(built.program,xp) << ",\"minus_lambda\":" << lambdaValue(built.program,xm)
            << ",\"midpoint_residual\":" << rmid << ",\"midpoint_yield_gap\":" << yieldGap(built.program,xmid)
            << ",\"midpoint_lambda\":" << lambdaValue(built.program,xmid) << ",\"realized_direction\":"; writeMatrix(realized);
        std::cout << ",\"midpoint_tangent_pa\":"; writeMatrix(midTangent);
        std::cout << ",\"plus_stress\":"; writeMatrix(sp); std::cout << ",\"minus_stress\":"; writeMatrix(sm); std::cout << ",\"fp64_fd\":"; writeMatrix(fd);
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
        projection(compiled,fp,accepted,&sp,nullptr,nullptr,&xp,&rp);
        projection(compiled,fm,accepted,&sm,nullptr,nullptr,&xm,&rm);
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
'''


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)

def cpp_float(value) -> str:
    rendered = f'{float(value):.9g}'
    if '.' not in rendered and 'e' not in rendered and 'E' not in rendered:
        rendered += '.0'
    return rendered + 'f'

def cpp_array(values, cast='double') -> str:
    return '{{' + ','.join(f'{float(v):.17g}' for v in values) + '}}'

def generate_case(capture_path: pathlib.Path, capture: dict) -> tuple[pathlib.Path, pathlib.Path]:
    material = capture['source_material']
    source_path = pathlib.Path(material['path'])
    if not source_path.is_file() or sha(source_path) != material['sha256']:
        raise RuntimeError(f"source material missing/hash mismatch: {source_path}")
    csv = pathlib.Path(capture['hardening_dataset']['path'])
    if not csv.is_file() or sha(csv) != capture['hardening_dataset']['sha256']:
        raise RuntimeError(f"hardening CSV missing/hash mismatch: {csv}")
    region = 'liner' if '_liner_' in material['name'] else 'medium'
    accepted_obj = capture['accepted_history_f32']
    accepted = [float(v) for v in accepted_obj.values()]
    if len(accepted) != 30:
        raise RuntimeError(f"expected 30 captured FP32 history components, got {len(accepted)}")
    f0 = [float(v) for v in capture['F_f32']]
    d = [float(v) for v in capture['direction_f32']]
    pairs=[]
    for row in capture['perturbation_sweep']:
        pairs.append('{'+f"{row['requested_epsilon']:.17g},"+cpp_array(row['plus_F_f32'])+','+cpp_array(row['minus_F_f32'])+'}')
    eps = sorted(set(EPSILONS + [row['requested_epsilon'] for row in capture['perturbation_sweep']]), reverse=True)
    # Add each exact half-step needed for an adjacent Richardson pair.
    exact = sorted(set(eps + [e*0.5 for e in eps]), reverse=True)
    replacements = {
        '@MATERIAL_PATH@': cpp_string(str(source_path)),
        '@MATERIAL_SHA@': cpp_string(material['sha256']),
        '@CSV_SHA@': cpp_string(capture['hardening_dataset']['sha256']),
        '@SET_SHA@': cpp_string(capture['synthetic_parameter_set_sha256']),
        '@REGION@': cpp_string(region),
        '@F32@': cpp_array(f0, 'float'),
        '@DIRECTION32@': cpp_array(d, 'float'),
        '@ACCEPTED32@': cpp_array(accepted, 'float'),
        '@PAIR_INPUTS@': '{'+','.join(pairs)+'}',
        '@EXACT_EPSILONS@': '{'+','.join(f'{e:.17g}' for e in exact)+'}',
        '@COOKED_PARAMETERS@': '{'+','.join('{'+cpp_string(k)+','+cpp_float(v)+'}' for k,v in capture['cooked_parameters_f32'].items())+'}',
        '@PROJECTION_TOLERANCE@': f'{FP64_TOLERANCE:.17g}',
        '@CONSTITUTIVE_FINGERPRINT@': str(capture['native_artifacts']['compiled_constitutive_fingerprint_u64'])+'ULL',
    }
    source = DRIVER
    for key, value in replacements.items():
        source = source.replace(key, value)
    if '@' in source:
        raise RuntimeError('unexpanded code template marker')
    cpp_path = HERE / f'fp64_{region}_unquantized.cpp'
    exe_path = HERE / f'fp64_{region}_unquantized'
    cpp_path.write_text(source)
    command = ['/usr/bin/clang++', '-x', 'objective-c++', '-O3', '-DNDEBUG', '-std=c++23', '-arch', 'arm64', '-fPIE', '-fobjc-arc', '-Wall', '-Wextra', '-Wpedantic', '-Werror']
    for include in INCLUDES:
        command += ['-I', str(include)]
    command += ['-I', str(SNAPSHOT / 'matter' / 'tools'), str(cpp_path), '-o', str(exe_path), '-x', 'none', str(LIB)]
    subprocess.run(command, check=True, cwd=HERE, capture_output=False)
    output = subprocess.check_output([str(exe_path)], text=True, cwd=HERE)
    output_path = HERE / f'fp64_{region}_unquantized.json'
    output_path.write_text(output)
    return cpp_path, output_path

def relative(a, b):
    na = math.sqrt(sum(float(x)*float(x) for x in a))
    nb = math.sqrt(sum(float(x)*float(x) for x in b))
    nd = math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(a,b)))
    return nd / nb if nb > 0 else None

def analyze_capture(capture_path: pathlib.Path):
    raw=capture_path.read_bytes(); capture=json.loads(raw)
    cpp_path, fp64_path=generate_case(capture_path,capture)
    result=json.loads(fp64_path.read_text())
    center_native=capture['center_native']; center_fp64=capture['center_fp64_source_reprojection']
    state_names=list(capture['accepted_history_f32'].keys())
    native_state=center_native['projected_state']; fp64_state=center_fp64['projected_state']
    state_diffs=[(abs(float(native_state[name])-float(fp64_state[name])),name,float(native_state[name]),float(fp64_state[name])) for name in state_names]
    state_diffs.sort(reverse=True)
    by_eps={float(row['epsilon']):row for row in result['representable_pairs']}
    exact_rows={float(row['epsilon']):row for row in result['unquantized_fp64_sweep']}
    pair_audit=[]
    for native in capture['perturbation_sweep']:
        eps=float(native['requested_epsilon']); oracle=by_eps[eps]
        same_pair_fd_error=relative(native['native_fd_action_pa'],oracle['fp64_fd'])
        original_pair_oracle_drift=relative(native['fp64_fd_action_pa'],oracle['fp64_fd'])
        requested_center_tangent=capture['center_fp64_source_reprojection']['tangent_action_Pa']
        midpoint_tangent_direction_error=relative(oracle['midpoint_tangent_pa'],requested_center_tangent)
        pair_audit.append({
            'epsilon':eps,
            'native_projection_residual_inf_plus_minus':[native['native_plus']['projection_residual_inf'],native['native_minus']['projection_residual_inf']],
            'fp64_projection_residual_inf_plus_minus':[oracle['plus_residual'],oracle['minus_residual']],
            'plus_minus_yield_gap_fp64':[oracle['plus_yield_gap'],oracle['minus_yield_gap']],
            'plus_minus_multiplier_fp64':[oracle['plus_lambda'],oracle['minus_lambda']],
            'midpoint_yield_gap_fp64':oracle['midpoint_yield_gap'],
            'midpoint_multiplier_fp64':oracle['midpoint_lambda'],
            'realized_direction_relative_l2_error':native['direction_realization_relative_l2'],
            'same_fp32_pair_native_fd_vs_tight_fp64_fd_relative_l2':same_pair_fd_error,
            'tight_fp64_fd_vs_capture_fp64_fd_relative_l2':original_pair_oracle_drift,
            'midpoint_ad_at_realized_direction_vs_requested_center_ad_relative_l2':midpoint_tangent_direction_error,
            'fp64_center_ad_vs_exact_same_fp32_pair_fd':native['fp64_ad_vs_fp64_fd_scaled_error'],
            'fp64_midpoint_ad_vs_same_pair_fd_relative_l2':oracle['fp64_fd_vs_midpoint_ad_relative'],
            'one_sided_plus_actual_pair_fd_vs_ad_relative_l2':oracle['one_sided_plus_error'],
            'one_sided_minus_actual_pair_fd_vs_ad_relative_l2':oracle['one_sided_minus_error'],
        })
    center_stress_match=relative(result['center']['stress_pa'],center_fp64['stress_Pa'])
    center_tangent_match=relative(result['center']['tangent_pa'],center_fp64['tangent_action_Pa'])
    if center_stress_match > 1.0e-7 or center_tangent_match > 1.0e-7:
        raise RuntimeError(f"FP64 reconstruction does not match retained capture: {capture_path}")
    by_exact_eps={float(row['epsilon']):row for row in result['unquantized_fp64_sweep']}
    exact_5e7=by_exact_eps[5.0e-7]
    exact_25e8=by_exact_eps[2.5e-7]
    exact_1e6=by_exact_eps[1.0e-6]
    oracle_gate = all(value is not None and value <= 1.0e-6 for value in (
        exact_5e7['fd_vs_ad_relative'], exact_5e7['half_step_spread_relative'],
        exact_1e6['richardson_vs_ad_relative']))
    result.update({
        'fp64_self_oracle_1e-6_gate':{
            'threshold':1.0e-6,
            'unquantized_direction_pair_epsilon':5.0e-7,
            'raw_centered_fd_vs_ad_relative':exact_5e7['fd_vs_ad_relative'],
            'step_spread_5e-7_vs_2.5e-7_relative':exact_5e7['half_step_spread_relative'],
            'richardson_1e-6_vs_ad_relative':exact_1e6['richardson_vs_ad_relative'],
            'pass':oracle_gate,
            'scope':'independent FP64 oracle self-check only; not native FP32/material qualification'},
        'reconstruction_check':{
            'captured_center_fp64_stress_relative_l2':center_stress_match,
            'captured_center_fp64_tangent_relative_l2':center_tangent_match,
            'pass':True},
    })
    result.update({
        'schema':'numi.cardboard.fp64-unquantized-tangent-audit.v1',
        'evidence_class':'CPU independent oracle over captured source-bound FP32 accepted history and inputs; synthetic instrument only, not calibration',
        'native_capture_path':str(capture_path),
        'native_capture_sha256':hashlib.sha256(raw).hexdigest(),
        'native_capture_material_sha256':capture['source_material']['sha256'],
        'native_capture_executable_sha256':capture['native_artifacts']['executable_sha256'],
        'native_capture_metallib_sha256':capture['native_artifacts']['metallib_sha256'],
        'analysis_cpp_path':str(cpp_path),
        'analysis_cpp_sha256':sha(cpp_path),
        'fp64_oracle_header_sha256':sha(SNAPSHOT/'matter'/'tools'/'cardboard_retained_damage_fp64_oracle.hpp'),
        'builder_header_sha256':sha(SNAPSHOT/'matter'/'tools'/'cardboard_retained_matrix_damage.hpp'),
        'compiled_library_path':str(LIB),
        'compiled_library_sha256':sha(LIB),
        'center_native_projection_residual_inf':center_native['projection_residual_inf'],
        'center_fp64_projection_residual_inf':center_fp64['projection_residual_inf'],
        'center_native_fp64_projected_state_error_max':{'component':state_diffs[0][1],'absolute':state_diffs[0][0],'native':state_diffs[0][2],'fp64':state_diffs[0][3]},
        'same_fp32_pair_audit':pair_audit,
        'strict_requested_direction_finite_difference_gate':'inconclusive_due_FP32_direction_realization_and_no_exactly_representable_pair',
        'unquantized_fp64_fd_threshold_1e-6_is_software_oracle_self_check_only':True,
    })
    return result

def main():
    global FP64_TOLERANCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--projection-tolerance', type=float, default=2.0e-13)
    args = parser.parse_args()
    if not (math.isfinite(args.projection_tolerance) and args.projection_tolerance > 0.0):
        raise RuntimeError('projection tolerance must be finite and positive')
    FP64_TOLERANCE = args.projection_tolerance
    if not LIB.is_file(): raise RuntimeError(f'compiler library not found: {LIB}')
    outputs=[]
    for capture_path in CAPTURES:
        outputs.append(analyze_capture(capture_path))
    combined={
        'schema':'numi.cardboard.fp64-unquantized-tangent-audit-bundle.v1',
        'evidence_class':'CPU-only numerical analysis; not native Metal or physical calibration evidence',
        'source_bound_inputs':['native capture JSON with SHA','material source SHA','hardening source CSV SHA','oracle and builder header SHA','compiler library SHA'],
        'cases':outputs,
    }
    out=HERE/f'analysis_tol_{FP64_TOLERANCE:.0e}.json'; out.write_text(json.dumps(combined,indent=2,sort_keys=True)+'\n')
    print(f'analysis={out} cases={len(outputs)}')
    for case in outputs:
        print(case['region'], 'center residual native/fp64', case['center_native_projection_residual_inf'],case['center_fp64_projection_residual_inf'],
              'largest state parity',case['center_native_fp64_projected_state_error_max'])
        for row in case['unquantized_fp64_sweep']:
            if row['epsilon'] in (1.0e-5,5.0e-6,1.0e-6,5.0e-7,2.5e-7):
                print(' ',row['epsilon'],'FD/AD',row['fd_vs_ad_relative'],'Richardson/AD',row['richardson_vs_ad_relative'],'spread',row['half_step_spread_relative'],
                      'branches',row['plus_lambda'],row['minus_lambda'],'res',row['plus_residual'],row['minus_residual'])
if __name__=='__main__': main()
