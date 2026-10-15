#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"
#include "cardboard_asymmetric_c1_material_oracle.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace numi::matter;
namespace oracle = numi_cardboard_asymmetric_c1_oracle;
using Matrix = oracle::Matrix;
using State = oracle::State;

constexpr Matrix kIdentity{1,0,0, 0,1,0, 0,0,1};
constexpr double kRelativeGate = 1.0e-3;
constexpr std::size_t kDiagnosticBase=25u+2u*NM_MAX_MATERIAL_STATE;
constexpr std::size_t kDiagnosticSlots=4u+NM_MAX_MATERIAL_STATE+
    NM_MAX_MATERIAL_STATE*NM_MAX_MATERIAL_STATE+NM_MAX_MATERIAL_STATE+
    9u*NM_MAX_MATERIAL_STATE+9u+NM_MAX_MATERIAL_STATE+9u;
constexpr std::size_t kProjectionTraceStateCapacity=16u;
constexpr std::size_t kProjectionTraceIterationCapacity=16u;
constexpr std::size_t kProjectionTraceHeaderSlots=8u;
constexpr std::size_t kProjectionTraceRecordSlots=6u+
    7u*kProjectionTraceStateCapacity+
    kProjectionTraceStateCapacity*kProjectionTraceStateCapacity;
constexpr std::size_t kProjectionTraceSlots=kProjectionTraceHeaderSlots+
    kProjectionTraceIterationCapacity*kProjectionTraceRecordSlots;
constexpr char kLinerHash[] = "f5951cd7fb63684a4fdcca58457af5d769fe466cc640370be54d9a4ac4309a43";
constexpr char kMediumHash[] = "40e8bd9aa61ed946661c162c395397f958bab71306df8bec8df1140e78ae95bb";
constexpr char kSourceCsvHash[] = "83f5778db797f0fb9996416848abb91c18f9102f4c3707aea9ea267b1eeb0d82";
constexpr char kTransferCsvHash[] = "5fccfea7ca49776cef6782f6fa8eff1876de38f643c6993c3b2829135f446e48";
bool gDetailedTangentDiagnostic=false;
bool gObservedTangentMismatch=false;
bool gObservedTangentCapture=false;
bool gProjectionIterationTrace=false;
bool gCompensatedProjection=false;
unsigned gCompletedProjectionTraceMaterials=0u;
unsigned gMaterialIterationBudget=16u;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string sha256(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot read hash-bound input: " + path.string());
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    require(bytes.size() <= static_cast<std::size_t>(UINT32_MAX), "hash input exceeds CommonCrypto limit");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    require(CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest)!=nullptr,
            "CommonCrypto SHA256 failed");
    static constexpr char hex[]="0123456789abcdef";
    std::string result;
    result.reserve(2u*CC_SHA256_DIGEST_LENGTH);
    for (const unsigned char value : digest) {
        result.push_back(hex[value>>4]);
        result.push_back(hex[value&0xf]);
    }
    return result;
}

std::vector<std::string> parseCsvRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted=false;
    for (std::size_t i=0;i<line.size();++i) {
        const char c=line[i];
        if (c=='"') {
            if (quoted && i+1u<line.size() && line[i+1u]=='"') { field.push_back('"'); ++i; }
            else quoted=!quoted;
        } else if (c==',' && !quoted) {
            fields.push_back(field); field.clear();
        } else field.push_back(c);
    }
    require(!quoted,"unterminated quoted field in source CSV");
    fields.push_back(field);
    return fields;
}

struct Curve {
    double yieldStrain=0.0;
    double endpointStrain=0.0;
    double tangentMPa=0.0;
    double tangentLowMPa=0.0;
    double tangentHighMPa=0.0;
    bool hasYield=false;
    bool hasEndpoint=false;
};

using Curves = std::map<std::pair<std::string,std::string>,Curve>;

Curves readSourceCurves(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(input.good(),"cannot read registered source CSV");
    std::string line;
    require(bool(std::getline(input,line)),"empty registered source CSV");
    const auto names=parseCsvRow(line);
    std::map<std::string,std::size_t> columns;
    for (std::size_t i=0;i<names.size();++i) columns.emplace(names[i],i);
    for (const char* name : {"component","axis","point_role","strain_mm_per_mm",
                             "postyield_tangent_MPa","postyield_tangent_low_MPa",
                             "postyield_tangent_high_MPa"})
        require(columns.contains(name),std::string("source CSV missing column ")+name);
    Curves curves;
    while (std::getline(input,line)) {
        if (line.empty()) continue;
        const auto row=parseCsvRow(line);
        require(row.size()==names.size(),"registered source CSV row width changed");
        const std::string& component=row[columns.at("component")];
        const std::string& axis=row[columns.at("axis")];
        const std::string& role=row[columns.at("point_role")];
        if (role!="published_Table_1_tensile_yield_anchor" &&
            role!="digitized_bilinear_fit_endpoint") continue;
        const double strain=std::stod(row[columns.at("strain_mm_per_mm")]);
        auto& curve=curves[{component,axis}];
        curve.tangentMPa=std::stod(row[columns.at("postyield_tangent_MPa")]);
        curve.tangentLowMPa=std::stod(row[columns.at("postyield_tangent_low_MPa")]);
        curve.tangentHighMPa=std::stod(row[columns.at("postyield_tangent_high_MPa")]);
        if (role=="published_Table_1_tensile_yield_anchor") {
            require(!curve.hasYield,"duplicate Table 1 source anchor");
            curve.yieldStrain=strain; curve.hasYield=true;
        } else {
            require(!curve.hasEndpoint,"duplicate digitized endpoint");
            curve.endpointStrain=strain; curve.hasEndpoint=true;
        }
    }
    for (const auto& key : {std::pair<std::string,std::string>{"liner","MD"},
                            {"liner","CD"},{"medium_NSSC","MD"},{"medium_NSSC","CD"}}) {
        const auto it=curves.find(key);
        require(it!=curves.end() && it->second.hasYield && it->second.hasEndpoint &&
                it->second.endpointStrain>it->second.yieldStrain,
                "registered CSV lacks a valid yield/endpoint pair");
    }
    return curves;
}

double cpuStressFreeAxialMPa(const ConstitutiveProgram& program,const unsigned axis,
                             const double axialStrain) {
    Matrix f=kIdentity;
    const unsigned axialComponent=axis==0u?0u:4u;
    f[axialComponent]=1.0+axialStrain;
    const std::array<unsigned,2> lateralComponents=axis==0u
        ?std::array<unsigned,2>{4u,8u}:std::array<unsigned,2>{0u,8u};
    const std::array<unsigned,2> lateralF=axis==0u
        ?std::array<unsigned,2>{4u,8u}:std::array<unsigned,2>{0u,8u};
    f[lateralF[0]]=1.0-0.18*axialStrain;
    f[lateralF[1]]=1.0-0.01*axialStrain;
    const State accepted{};
    oracle::Point point;
    for (unsigned iteration=0;iteration<12u;++iteration) {
        point=oracle::project(program,f,accepted);
        const std::array<double,2> residual{
            point.firstPiolaPa[lateralComponents[0]],
            point.firstPiolaPa[lateralComponents[1]]};
        const double norm=std::max(std::abs(residual[0]),std::abs(residual[1]));
        if (norm<2.0e-2) break;
        std::array<double,4> jac{};
        for (unsigned column=0;column<2u;++column) {
            Matrix direction{}; direction[lateralF[column]]=1.0;
            const Matrix tangent=oracle::tangent(program,f,direction,accepted,point.state);
            for (unsigned row=0;row<2u;++row)
                jac[2u*row+column]=tangent[lateralComponents[row]];
        }
        const double det=jac[0]*jac[3]-jac[1]*jac[2];
        require(std::isfinite(det)&&std::abs(det)>1.0e-10,
                "CPU bilinear stress-free lateral tangent is singular");
        const std::array<double,2> correction{
            (-jac[3]*residual[0]+jac[1]*residual[1])/det,
            (jac[2]*residual[0]-jac[0]*residual[1])/det};
        bool acceptedStep=false;
        double step=1.0;
        for (unsigned backtrack=0;backtrack<16u;++backtrack) {
            Matrix trialF=f;
            for (unsigned c=0;c<2u;++c) trialF[lateralF[c]]+=step*correction[c];
            const auto trial=oracle::project(program,trialF,accepted);
            const double trialNorm=std::max(
                std::abs(trial.firstPiolaPa[lateralComponents[0]]),
                std::abs(trial.firstPiolaPa[lateralComponents[1]]));
            if (trialNorm<norm) { f=trialF; acceptedStep=true; break; }
            step*=0.5;
        }
        require(acceptedStep,"CPU bilinear stress-free lateral solve stalled");
    }
    point=oracle::project(program,f,accepted);
    return point.firstPiolaPa[axialComponent]*1.0e-6;
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string result;
    for (const auto& value : values) result+=value.message+"; ";
    return result;
}

CompiledWorld cook(const MaterialProgram& material) {
    WorldSource source;
    MaterialProgram selected=material;
    selected.genericCompensatedImplicitProjection=gCompensatedProjection;
    source.materials={selected};
    source.gravity={0,0,0};
    source.environmentCount=1;
    source.frameTimestep=1.0e-4;
    ObjectSource object;
    object.name="asymmetric_c1_material_point_only";
    object.representation=Representation::fem;
    object.deformableContact=false;
    object.deformableSelfContact=false;
    object.mixedFEM=false;
    object.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}}};
    object.tetrahedra={{{0,1,2,3}}};
    object.characteristicLength=.01;
    source.objects={object};
    const auto result=compileWorld(source,{.maximumRateExponent=0u,.emitSpecializedMetal=false,
        .localMaterialNewtonIterations=gMaterialIterationBudget});
    require(result.succeeded(),"C1 material compile failed: "+diagnostics(result.diagnostics));
    require(result.world.materials.size()==1u && result.world.constitutive.size()==1u,
            "material-point fixture did not compile one constitutive program");
    const auto& compiled=result.world.constitutive[0];
    require(compiled.material.internalState.size()==15u &&
            compiled.implicitResiduals.size()==15u &&
            compiled.implicitJacobians.size()==225u &&
            compiled.implicitDeformationDirections.size()==15u &&
            compiled.stressStateDerivatives.size()==135u,
            "expected full generic 15-state residual/Jacobian/tangent chain");
    require(!compiled.material.paperHillScalarInitializer &&
            !compiled.material.paperCompensatedArithmetic &&
            !compiled.material.paperHillAccurateProjection &&
            compiled.material.genericCompensatedImplicitProjection==gCompensatedProjection,
            "compiled material did not select the explicitly requested ordinary/compensated generic policy");
    require(result.world.materials[0].stateCount==15u,
            "compiled Metal material does not expose all 15 persistent states");
    require(result.world.stateInitials.size()==15u,
            "one-tet fixture has unexpected state-initial arena layout");
    return result.world;
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device,const std::vector<T>& values) {
    T zero{};
    const void* bytes=values.empty()?static_cast<const void*>(&zero):static_cast<const void*>(values.data());
    return [device newBufferWithBytes:bytes
                               length:sizeof(T)*std::max<std::size_t>(1u,values.size())
                              options:MTLResourceStorageModeShared];
}

struct Response {
    bool valid=false;
    bool projectionValid=false;
    Matrix p{};
    Matrix tangent{};
    State state{};
    float residual=0.0f;
    unsigned iterations=0u;
    unsigned flags=0u;
    unsigned failure=0u;
    bool localDiagnosticAvailable=false;
    bool localDiagnosticEvaluated=false;
    bool localDiagnosticSolveValid=false;
    std::array<double,oracle::kStateCount> localResidual{};
    std::array<double,oracle::kStateCount*oracle::kStateCount> localJacobian{};
    std::array<double,oracle::kStateCount> localDeformationResidualDerivative{};
    std::array<double,9u*oracle::kStateCount> localStressStateDerivative{};
    Matrix localFixedStateTangent{};
    State localStateDirection{};
    Matrix localReconstructedTangent{};
    std::array<float,kProjectionTraceSlots> projectionTrace{};
};

struct MetalInstrument {
    id<MTLDevice> device=nil;
    id<MTLCommandQueue> queue=nil;
    id<MTLComputePipelineState> pipeline=nil;
    id<MTLComputePipelineState> diagnosticPipeline=nil;

    MetalInstrument() {
        device=MTLCreateSystemDefaultDevice();
        require(device!=nil,"Metal device unavailable");
        const std::string name=[[device name] UTF8String];
        require(name.find("Apple")!=std::string::npos &&
                name.find("Paravirtual")==std::string::npos,
                "physical Apple GPU required for this native material check");
        queue=[device newCommandQueue];
        NSError* error=nil;
        id<MTLLibrary> library=[device newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&error];
        require(library!=nil,"cardboard_paper_check.metallib is missing");
        id<MTLFunction> function=[library newFunctionWithName:@"cardboard_paper_check"];
        require(function!=nil,"existing production material-point kernel is missing");
        pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline!=nil,"could not build existing production material-point pipeline");
        id<MTLFunction> diagnosticFunction=[library newFunctionWithName:
            @"cardboard_paper_tangent_diagnostic_check"];
        require(diagnosticFunction!=nil,"material tangent diagnostic kernel is missing");
        diagnosticPipeline=[device newComputePipelineStateWithFunction:diagnosticFunction error:&error];
        require(diagnosticPipeline!=nil,"could not build material tangent diagnostic pipeline");
        std::cout<<"device=\""<<name<<"\" kernel=cardboard_paper_check"
            <<" scope=native_metal_generic_material_point_only physical_steps=0\n";
    }

    Response evaluate(const CompiledWorld& world,const Matrix& inputF,const Matrix& inputH,
                      const State& accepted,const bool detailedDiagnostic=false,
                      const bool traceProjection=false) {
        const bool runDiagnosticKernel=detailedDiagnostic||traceProjection;
        std::vector<float> inputs;
        inputs.reserve(29u);
        Matrix realizedF{},realizedH{};
        for (unsigned i=0;i<9u;++i) { realizedF[i]=static_cast<float>(inputF[i]); inputs.push_back(static_cast<float>(inputF[i])); }
        for (unsigned i=0;i<9u;++i) { realizedH[i]=static_cast<float>(inputH[i]); inputs.push_back(static_cast<float>(inputH[i])); }
        for (unsigned i=0;i<9u;++i) inputs.push_back(0.0f);
        inputs.push_back(0.0f); // ordinary generic projection, no 13-state-only accurate policy
        inputs.push_back(traceProjection?1.0f:0.0f);
        std::vector<float> acceptedFloat(accepted.begin(),accepted.end());
        std::vector<float> parameterValues;
        parameterValues.reserve(world.parameters.size());
        for (const auto& value : world.parameters) parameterValues.push_back(value.valueAndBounds.x);
        const std::vector<float> outputZero(25u+2u*NM_MAX_MATERIAL_STATE+
            (runDiagnosticKernel?kDiagnosticSlots:0u)+
            (traceProjection?kProjectionTraceSlots:0u),0.0f);
        std::array<id<MTLBuffer>,7> buffers{
            makeBuffer(device,std::vector<NMMaterialGPU>{world.materials.at(0)}),
            makeBuffer(device,world.scalarPrograms),
            makeBuffer(device,world.instructions),
            makeBuffer(device,parameterValues),
            makeBuffer(device,acceptedFloat),
            makeBuffer(device,inputs),
            makeBuffer(device,outputZero)};
        require(std::all_of(buffers.begin(),buffers.end(),[](id<MTLBuffer> x){return x!=nil;}),
                "failed allocating material-point Metal buffers");
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:runDiagnosticKernel?diagnosticPipeline:pipeline];
        for (NSUInteger i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,
                command.error?std::string([[command.error description] UTF8String]):
                    "Metal material-point command did not complete");
        const auto* out=static_cast<const float*>(buffers[6].contents);
        Response result;
        result.valid=out[0]==1.0f;
        for (unsigned i=0;i<9u;++i) { result.p[i]=out[1u+i]; result.tangent[i]=out[10u+i]; }
        for (unsigned i=0;i<oracle::kStateCount;++i) result.state[i]=out[19u+i];
        result.projectionValid=out[19u+NM_MAX_MATERIAL_STATE]==1.0f;
        result.residual=out[20u+NM_MAX_MATERIAL_STATE];
        result.iterations=static_cast<unsigned>(out[22u+NM_MAX_MATERIAL_STATE]);
        result.flags=static_cast<unsigned>(out[23u+NM_MAX_MATERIAL_STATE]);
        result.failure=static_cast<unsigned>(out[24u+NM_MAX_MATERIAL_STATE]);
        if (detailedDiagnostic) {
            const float* raw=out+kDiagnosticBase;
            result.localDiagnosticAvailable=raw[0]==1.0f;
            result.localDiagnosticEvaluated=raw[1]==1.0f;
            result.localDiagnosticSolveValid=raw[2]==1.0f;
            require(static_cast<unsigned>(raw[3])==oracle::kStateCount,
                    "diagnostic material state count mismatch");
            std::size_t cursor=4u;
            for(double& x:result.localResidual)x=raw[cursor++];
            for(double& x:result.localJacobian)x=raw[cursor++];
            for(double& x:result.localDeformationResidualDerivative)x=raw[cursor++];
            for(double& x:result.localStressStateDerivative)x=raw[cursor++];
            for(double& x:result.localFixedStateTangent)x=raw[cursor++];
            for(double& x:result.localStateDirection)x=raw[cursor++];
            for(double& x:result.localReconstructedTangent)x=raw[cursor++];
            require(cursor<=kDiagnosticSlots,"diagnostic tangent payload overflow");
        }
        if(traceProjection) {
            const float* trace=out+25u+2u*NM_MAX_MATERIAL_STATE+kDiagnosticSlots;
            std::copy(trace,trace+kProjectionTraceSlots,result.projectionTrace.begin());
        }
        // Keep exact input roundoff visible to the finite-difference check.
        for (unsigned i=0;i<9u;++i) {
            require(std::isfinite(realizedF[i]) && std::isfinite(realizedH[i]),
                    "nonfinite Metal input conversion");
        }
        return result;
    }
};

double norm(const Matrix& values) {
    double sum=0.0;
    for (double value : values) sum+=value*value;
    return std::sqrt(sum);
}

double stateNorm(const State& values) {
    double sum=0.0;
    for (double value : values) sum+=value*value;
    return std::sqrt(sum);
}

double relativeError(const Matrix& actual,const Matrix& expected) {
    Matrix difference{};
    for (unsigned i=0;i<9u;++i) difference[i]=actual[i]-expected[i];
    return norm(difference)/norm(expected);
}

double relativeStateError(const State& actual,const State& expected) {
    State difference{};
    for (unsigned i=0;i<oracle::kStateCount;++i) difference[i]=actual[i]-expected[i];
    return stateNorm(difference)/stateNorm(expected);
}

template<class Range>
void writeArray(std::ostream& out,const Range& values) {
    out<<'[';
    bool first=true;
    for(const auto value:values) {
        if(!first)out<<',';
        first=false;
        if(!std::isfinite(static_cast<double>(value))) out<<"null";
        else out<<std::setprecision(10)<<value;
    }
    out<<']';
}

const char* compressionBranch(const double stress,const double width) {
    if(stress>=width)return "tensile_zero_compression_term";
    if(stress<=-width)return "linear_compression_term";
    return "C1_transition";
}

void emitTangentDiagnostic(const std::string& label,const unsigned column,
                           const Matrix& f,const Matrix& h,const State& accepted,
                           const Response& metal,const CompiledWorld& world,
                           const Matrix& oracleTangent) {
    // Metal receives float buffers. Rebuild the independent oracle inputs from
    // those exact representable values so this record isolates local
    // arithmetic instead of mixing in host-to-device conversion.
    Matrix realizedF{},realizedH{};
    State realizedAccepted{};
    for(unsigned i=0;i<9u;++i) {
        realizedF[i]=static_cast<float>(f[i]);
        realizedH[i]=static_cast<float>(h[i]);
    }
    for(unsigned i=0;i<oracle::kStateCount;++i)
        realizedAccepted[i]=static_cast<float>(accepted[i]);
    const auto captureExact=oracle::project(world.constitutive.at(0),realizedF,
                                             realizedAccepted);
    const Matrix captureTangent=oracle::tangent(world.constitutive.at(0),realizedF,
        realizedH,realizedAccepted,captureExact.state);
    const auto exactLinearization=oracle::linearize(
        world.constitutive.at(0),realizedF,realizedH,realizedAccepted,captureExact.state);
    const double width=oracle::parameter(world.constitutive.at(0).material,
                                          "compression_smoothing_width");
    std::cout<<"{\"schema\":\"numi.cardboard.asymmetric-c1-tangent-diagnostic.v1\","
        <<"\"case\":\""<<label<<"\",\"column\":"<<column<<",\"F_row_major\":";
    writeArray(std::cout,realizedF);
    std::cout<<",\"H_row_major\":";writeArray(std::cout,realizedH);
    std::cout<<",\"accepted_state\":";writeArray(std::cout,realizedAccepted);
    std::cout<<",\"metal_state\":";writeArray(std::cout,metal.state);
    std::cout<<",\"oracle_state\":";writeArray(std::cout,captureExact.state);
    std::cout<<",\"metal_tangent_Pa_per_F\":";writeArray(std::cout,metal.tangent);
    std::cout<<",\"oracle_tangent_Pa_per_F\":";writeArray(std::cout,captureTangent);
    std::cout<<",\"oracle_tangent_before_input_rounding_Pa_per_F\":";
    writeArray(std::cout,oracleTangent);
    std::cout<<",\"metal_projection_residual\":"<<metal.residual
        <<",\"metal_projection_iterations\":"<<metal.iterations
        <<",\"metal_stage_flags\":"<<metal.flags
        <<",\"metal_projection_failure\":"<<metal.failure
        <<",\"metal_local_capture_available\":"
        <<(metal.localDiagnosticAvailable?"true":"false")
        <<",\"metal_local_evaluation_valid\":"
        <<(metal.localDiagnosticEvaluated?"true":"false")
        <<",\"metal_local_solve_valid\":"
        <<(metal.localDiagnosticSolveValid?"true":"false");
    if(metal.localDiagnosticAvailable) {
        std::cout<<",\"metal_local_residual\":";writeArray(std::cout,metal.localResidual);
        std::cout<<",\"metal_local_jacobian_row_major\":";writeArray(std::cout,metal.localJacobian);
        std::cout<<",\"metal_rF_for_H\":";writeArray(std::cout,metal.localDeformationResidualDerivative);
        std::cout<<",\"metal_dP_dz_row_major\":";writeArray(std::cout,metal.localStressStateDerivative);
        std::cout<<",\"metal_fixed_state_tangent\":";writeArray(std::cout,metal.localFixedStateTangent);
        std::cout<<",\"metal_dz_dF_for_H\":";writeArray(std::cout,metal.localStateDirection);
        std::cout<<",\"metal_reconstructed_tangent\":";writeArray(std::cout,metal.localReconstructedTangent);
    }
    std::cout<<",\"oracle_local_residual\":";writeArray(std::cout,exactLinearization.residual);
    std::cout<<",\"oracle_local_jacobian_row_major\":";writeArray(std::cout,exactLinearization.jacobian);
    std::cout<<",\"oracle_rF_for_H\":";writeArray(std::cout,exactLinearization.deformationResidualDerivative);
    std::cout<<",\"oracle_dP_dz_row_major\":";writeArray(std::cout,exactLinearization.stressStateDerivative);
    std::cout<<",\"oracle_fixed_state_tangent\":";writeArray(std::cout,exactLinearization.fixedStateTangent);
    State oracleStateDirection{};
    for(unsigned i=0;i<oracle::kStateCount;++i)
        oracleStateDirection[i]=-exactLinearization.deformationResidualDerivative[i];
    require(oracle::solve(exactLinearization.jacobian,oracleStateDirection,oracleStateDirection),
            "diagnostic oracle tangent solve failed");
    std::cout<<",\"oracle_dz_dF_for_H\":";writeArray(std::cout,oracleStateDirection);
    std::array<double,3> metalPrincipal{metal.state[6],metal.state[7],metal.state[8]};
    std::array<double,3> oraclePrincipal{captureExact.state[6],captureExact.state[7],captureExact.state[8]};
    std::cout<<",\"compression_width\":"<<width<<",\"metal_principal_sbar\":";
    writeArray(std::cout,metalPrincipal);
    std::cout<<",\"oracle_principal_sbar\":";writeArray(std::cout,oraclePrincipal);
    std::cout<<",\"metal_compression_branches\":[\""
        <<compressionBranch(metalPrincipal[0],width)<<"\",\""
        <<compressionBranch(metalPrincipal[1],width)<<"\",\""
        <<compressionBranch(metalPrincipal[2],width)<<"\"],\"oracle_compression_branches\":[\""
        <<compressionBranch(oraclePrincipal[0],width)<<"\",\""
        <<compressionBranch(oraclePrincipal[1],width)<<"\",\""
        <<compressionBranch(oraclePrincipal[2],width)<<"\"],\"local_rF_sign\":\"positive_derivative_of_residual\","
        <<"\"capture_source\":\"production_generic_material_entrypoint_plus_explicit_bytecode_operands\"}\n";
    gObservedTangentCapture=true;
}

State asDouble(const Response& value) { return value.state; }

void validateStateAndStress(const std::string& label,const Response& metal,
                            const oracle::Point& exact,const State& accepted) {
    if (!metal.valid || !metal.projectionValid) {
        std::ostringstream message;
        message<<label<<" Metal projection/stress validity failed"
            <<" valid="<<metal.valid<<" projection_valid="<<metal.projectionValid
            <<" residual="<<metal.residual<<" iterations="<<metal.iterations
            <<" flags=0x"<<std::hex<<metal.flags<<std::dec
            <<" failure="<<metal.failure;
        throw std::runtime_error(message.str());
    }
    Matrix actualP=metal.p;
    const double exactPNorm=norm(exact.firstPiolaPa);
    if (exactPNorm<=1.0e-2) {
        std::cout<<"{\"case\":\""<<label<<"\",\"quantity\":\"first_piola\","
            <<"\"status\":\"inconclusive_zero_response\",\"oracle_norm_Pa\":"<<exactPNorm
            <<",\"noise_floor\":null}\n";
    } else {
        const double error=relativeError(actualP,exact.firstPiolaPa);
        std::cout<<"{\"case\":\""<<label<<"\",\"quantity\":\"first_piola\","
            <<"\"relative_error\":"<<error<<",\"status\":\""
            <<(error<=kRelativeGate?"pass":"fail")<<"\"}\n";
        require(error<=kRelativeGate,label+" Metal first Piola differs from FP64 bytecode oracle by more than 1e-3");
    }
    const double exactStateNorm=stateNorm(exact.state);
    if (exactStateNorm<=1.0e-10) {
        std::cout<<"{\"case\":\""<<label<<"\",\"quantity\":\"state\","
            <<"\"status\":\"inconclusive_zero_response\",\"oracle_norm\":"<<exactStateNorm
            <<",\"noise_floor\":null}\n";
    } else {
        const double error=relativeStateError(asDouble(metal),exact.state);
        std::cout<<"{\"case\":\""<<label<<"\",\"quantity\":\"all_15_states\","
            <<"\"relative_error\":"<<error<<",\"history13\":"<<metal.state[13]
            <<",\"history14\":"<<metal.state[14]<<",\"status\":\""
            <<(error<=kRelativeGate?"pass":"fail")<<"\"}\n";
        require(error<=kRelativeGate,label+" complete projected state differs from FP64 bytecode oracle by more than 1e-3");
    }
    for (unsigned state=12u;state<15u;++state)
        require(metal.state[state]+2.0e-7>=accepted[state],
                label+" irreversible multiplier/history regressed at state "+std::to_string(state));
}

void validateFullTangent(const std::string& label,MetalInstrument& metal,
                         const CompiledWorld& world,const Matrix& f,
                         const State& accepted,const oracle::Point& exact) {
    unsigned resolved=0u;
    double worst=0.0;
    bool allPassed=true;
    for (unsigned column=0;column<9u;++column) {
        Matrix direction{}; direction[column]=1.0;
        const bool selectedDiagnostic=gDetailedTangentDiagnostic&&column==2u&&
            label.find("liner_hill_asymmetric_c1_transfer_MD_compression")!=std::string::npos;
        const Response actual=metal.evaluate(world,f,direction,accepted,selectedDiagnostic);
        require(actual.valid && actual.projectionValid,label+" tangent column projection failed");
        const Matrix expected=oracle::tangent(world.constitutive.at(0),f,direction,
                                              accepted,exact.state);
        if(selectedDiagnostic)emitTangentDiagnostic(label,column,f,direction,accepted,
                                                     actual,world,expected);
        const double expectedNorm=norm(expected);
        if (expectedNorm<=1.0e-2) {
            std::cout<<"{\"case\":\""<<label<<"\",\"tangent_column\":"<<column
                     <<",\"status\":\"inconclusive_zero_response\",\"oracle_norm_Pa_per_F\":"
                     <<expectedNorm<<"}\n";
            continue;
        }
        const double error=relativeError(actual.tangent,expected);
        worst=std::max(worst,error); ++resolved;
        std::cout<<"{\"case\":\""<<label<<"\",\"tangent_column\":"<<column
                 <<",\"relative_error\":"<<error<<",\"gate\":"<<kRelativeGate
                 <<",\"status\":\""<<(error<=kRelativeGate?"pass":"fail")<<"\"}\n";
        if(error>kRelativeGate) {
            allPassed=false;
            if(gDetailedTangentDiagnostic)gObservedTangentMismatch=true;
            else require(false,label+" full algorithmic tangent column exceeds 1e-3 gate");
        }
    }
    require(resolved>0u,label+" produced no resolved tangent columns");
    std::cout<<"{\"case\":\""<<label<<"\",\"quantity\":\"complete_9x9_algorithmic_tangent\","
        <<"\"resolved_columns\":"<<resolved<<",\"worst_relative_error\":"<<worst
        <<",\"includes_implicit_states\":15,\"status\":\""<<(allPassed?"pass":"fail")<<"\"}\n";
}

void emitBranchAwareFDSide(const std::string& label,const unsigned component,
                           const unsigned level,const char* side,
                           const double requestedEpsilon,const double realizedEpsilon,
                           const Matrix& f,const Response& response) {
    // This record is emitted before the caller enforces projection validity.
    // It preserves the first failing perturbation instead of reducing a local
    // return-map failure to a single checker exception.
    const auto& trace=response.projectionTrace;
    const bool traceSupported=trace[0]==1.0f && trace[7]!=-1.0f;
    const unsigned stateCount=traceSupported?static_cast<unsigned>(trace[1]):0u;
    const unsigned iterations=traceSupported?std::min<unsigned>(
        static_cast<unsigned>(trace[2]),
        static_cast<unsigned>(kProjectionTraceIterationCapacity)):0u;
    std::cout<<std::setprecision(10)
        <<"{\"schema\":\"numi.cardboard.c1-fd-side-projection.v1\","
        <<"\"case\":\""<<label<<"\",\"component\":"<<component
        <<",\"fd_level\":"<<level<<",\"side\":\""<<side
        <<"\",\"compiled_material_local_newton_iterations\":"<<gMaterialIterationBudget
        <<",\"projection_trace_capacity_iterations\":"<<kProjectionTraceIterationCapacity
        <<",\"projection_trace_truncated\":"
        <<(gMaterialIterationBudget>kProjectionTraceIterationCapacity?"true":"false")
        <<",\"requested_epsilon\":"<<requestedEpsilon
        <<",\"realized_epsilon\":"<<realizedEpsilon
        <<",\"F_row_major\":";
    writeArray(std::cout,f);
    std::cout<<",\"projection_valid\":"<<(response.projectionValid?"true":"false")
        <<",\"stress_tangent_valid\":"<<(response.valid?"true":"false")
        <<",\"terminal_failure\":"<<response.failure
        <<",\"iterations\":"<<response.iterations
        <<",\"residual_inf\":"<<response.residual
        <<",\"stage_flags\":"<<response.flags
        <<",\"state\":";
    writeArray(std::cout,response.state);
    std::cout<<",\"projection_trace_supported\":"
        <<(traceSupported?"true":"false");
    if(traceSupported) {
        std::cout<<",\"trace_terminal_failure\":"<<trace[7]
            <<",\"trace_final_residual_inf\":"<<trace[4]
            <<",\"trace_projection_tolerance\":"<<trace[6]
            <<",\"trace_iterations\":[";
        for(unsigned iteration=0u;iteration<iterations;++iteration) {
            if(iteration)std::cout<<',';
            const std::size_t base=kProjectionTraceHeaderSlots+
                iteration*kProjectionTraceRecordSlots;
            const auto span=[&](std::size_t offset,std::size_t count) {
                return std::span<const float>(trace.data()+base+offset,count);
            };
            std::cout<<"{\"index\":"<<iteration
                <<",\"record_flags\":"<<trace[base]
                <<",\"accepted_backtrack_index\":"<<trace[base+1u]
                <<",\"accepted_alpha\":"<<trace[base+2u]
                <<",\"residual_inf\":"<<trace[base+3u]
                <<",\"state_inf\":"<<trace[base+4u]
                <<",\"candidate_state_high\":";
            writeArray(std::cout,span(6u,stateCount));
            std::cout<<",\"candidate_state_low\":";
            writeArray(std::cout,span(6u+kProjectionTraceStateCapacity,stateCount));
            std::cout<<",\"residual_rows\":";
            writeArray(std::cout,span(6u+2u*kProjectionTraceStateCapacity,stateCount));
            std::cout<<",\"newton_delta\":";
            writeArray(std::cout,span(6u+3u*kProjectionTraceStateCapacity,stateCount));
            std::cout<<",\"trial_residual_norms\":";
            writeArray(std::cout,span(6u+4u*kProjectionTraceStateCapacity,
                                      kProjectionTraceStateCapacity));
            std::cout<<'}';
        }
        std::cout<<']';
    }
    std::cout<<",\"physical_steps\":0,\"diagnostic_only\":true}\n";
    std::cout.flush();
}

void validateBranchAwareFD(const std::string& label,MetalInstrument& metal,
                           const CompiledWorld& world,const Matrix& centerF,
                           const unsigned component,const State& accepted,
                           const Response& center) {
    Matrix direction{};
    direction[component]=1.0;
    constexpr std::array<double,2> epsilons{5.0e-4,2.5e-4};
    std::array<Matrix,2> estimates{};
    for (unsigned level=0;level<epsilons.size();++level) {
        Matrix plusF=centerF,minusF=centerF;
        plusF[component]+=epsilons[level];
        minusF[component]-=epsilons[level];
        const float centerInput=static_cast<float>(centerF[component]);
        const float plusInput=static_cast<float>(plusF[component]);
        const float minusInput=static_cast<float>(minusF[component]);
        const double realizedPlus=double(plusInput)-double(centerInput);
        const double realizedMinus=double(centerInput)-double(minusInput);
        require(realizedPlus>0.0 && realizedMinus>0.0,
                label+" finite-difference perturbation is not representable in FP32");
        const bool captureSideTrace=gDetailedTangentDiagnostic &&
            label.find("_F13")!=std::string::npos;
        const Response plus=metal.evaluate(world,plusF,direction,accepted,
                                            false,captureSideTrace);
        const Response minus=metal.evaluate(world,minusF,direction,accepted,
                                             false,captureSideTrace);
        if(captureSideTrace) {
            emitBranchAwareFDSide(label,component,level,"plus",epsilons[level],
                                  realizedPlus,plusF,plus);
            emitBranchAwareFDSide(label,component,level,"minus",epsilons[level],
                                  realizedMinus,minusF,minus);
        }
        require(plus.valid&&plus.projectionValid&&minus.valid&&minus.projectionValid,
                label+" branch-aware finite-difference projection failed");
        const bool plusPlastic=plus.state[12]-accepted[12]>1.0e-8;
        const bool minusPlastic=minus.state[12]-accepted[12]>1.0e-8;
        const bool centerPlastic=center.state[12]-accepted[12]>1.0e-8;
        require(centerPlastic&&plusPlastic&&minusPlastic,
                label+" FD perturbations did not remain on the loaded plastic branch");
        for (unsigned row=0;row<9u;++row)
            estimates[level][row]=(plus.p[row]-minus.p[row])/(realizedPlus+realizedMinus);
        std::cout<<"{\"case\":\""<<label<<"\",\"fd_level\":"<<level
            <<",\"requested_epsilon\":"<<epsilons[level]
            <<",\"realized_plus\":"<<realizedPlus<<",\"realized_minus\":"<<realizedMinus
            <<",\"branch\":\"plastic\",\"projection_residual_plus\":"<<plus.residual
            <<",\"projection_residual_minus\":"<<minus.residual<<"}\n";
    }
    Matrix richardson{};
    for (unsigned i=0;i<9u;++i)
        richardson[i]=(4.0*estimates[1][i]-estimates[0][i])/3.0;
    const double noise=relativeError(estimates[0],estimates[1]);
    const double tangentError=relativeError(center.tangent,richardson);
    std::cout<<"{\"case\":\""<<label<<"\",\"quantity\":\"branch_aware_centered_fd\","
        <<"\"step_halving_noise_relative\":"<<noise<<",\"metal_tangent_relative_error\":"
        <<tangentError<<",\"gate\":"<<kRelativeGate<<",\"status\":\""
        <<(noise<=kRelativeGate?(tangentError<=kRelativeGate?"pass":"fail_resolved_tangent_mismatch"):
                                  "inconclusive_fd_noise")
        <<"\"}\n";
    // The exact FP64 compiled tangent is also checked for every column. This
    // FD gate is enforced only when step-halving measured its own noise below
    // 1e-3; noisier comparisons remain explicitly inconclusive.
    if (noise<=kRelativeGate)
        require(tangentError<=kRelativeGate,
                label+" branch-preserving resolved Metal FD disagrees with tangent by more than 1e-3");
}

struct MaterialFixture {
    std::string role;
    std::string assetName;
    std::string assetHash;
    std::string csvComponent;
    std::filesystem::path path;
};

void emitProjectionIterationTrace(const MaterialFixture& fixture,
                                  const CompiledWorld& world,
                                  const State& acceptedHistory,
                                  const Response& unload,
                                  const oracle::Point& oracleUnload,
                                  const std::string& digest) {
    const auto& trace=unload.projectionTrace;
    const unsigned n=static_cast<unsigned>(trace[1]);
    const unsigned iterationCount=static_cast<unsigned>(trace[2]);
    const double residualScale=oracle::parameter(
        world.constitutive.at(0).material,"residual_scale");
    std::cout<<std::setprecision(10)
        <<"{\"schema\":\"numi.cardboard.c1-local-projection-trace.v1\","
        <<"\"material\":\""<<fixture.assetName<<"\",\"asset_sha256\":\""
        <<digest<<"\",\"source_csv_sha256\":\""<<kSourceCsvHash
        <<"\",\"axis\":\"CD\",\"stage\":\"unload_to_identity\","
        <<"\"compiled_material_local_newton_iterations\":"
        <<world.materials.at(0).localNewtonIterations<<','
        <<"\"compiled_generic_compensated_implicit\":"
        <<(gCompensatedProjection?"true":"false")<<','
        <<"\"F_row_major\":";writeArray(std::cout,kIdentity);
    std::cout<<",\"F_low_row_major\":";
    const Matrix zeroLow{};writeArray(std::cout,zeroLow);
    std::cout<<",\"accepted_state\":";writeArray(std::cout,acceptedHistory);
    std::cout<<",\"trace_supported\":"<<(trace[7]!=-1.0f?"true":"false")
        <<",\"state_count\":"<<n<<",\"attempted_iterations\":"
        <<iterationCount<<",\"converged\":"<<(trace[3]==1.0f?"true":"false")
        <<",\"terminal_projection_failure\":"<<trace[7]
        <<",\"projection_tolerance\":"<<trace[6]
        <<",\"residual_scale_factor\":"<<residualScale
        <<",\"residual_row_names\":[\"ep11\",\"ep22\",\"ep33\",\"ep23\",\"ep13\",\"ep12\","
          "\"sbar11\",\"sbar22\",\"sbar33\",\"sbar23\",\"sbar13\",\"sbar12\","
          "\"accumulated_multiplier\",\"hardening_md_accumulation\",\"hardening_cd_accumulation\"]"
        <<",\"residual_units\":\"dimensionless; rows scaled by residual_scale_factor\""
        <<",\"timestep_s\":1e-4,\"temperature_K\":293.15"
        <<",\"final_residual_inf_scaled\":"<<trace[4]
        <<",\"final_residual_inf_unscaled\":"<<trace[4]/residualScale
        <<",\"final_state_norm_inf\":"<<trace[5]
        <<",\"fp64_oracle_iterations\":"<<oracleUnload.iterations
        <<",\"fp64_oracle_residual_inf\":"<<oracleUnload.residualInfinity
        <<",\"fp64_oracle_state\":";writeArray(std::cout,oracleUnload.state);
    std::cout<<",\"native_projection_valid\":"<<(unload.projectionValid?"true":"false")
        <<",\"native_stress_tangent_valid\":"<<(unload.valid?"true":"false")
        <<",\"iterations\":[";
    const unsigned copied=std::min<unsigned>(iterationCount,
        static_cast<unsigned>(kProjectionTraceIterationCapacity));
    for(unsigned iteration=0u;iteration<copied;++iteration) {
        if(iteration)std::cout<<',';
        const std::size_t base=kProjectionTraceHeaderSlots+
            iteration*kProjectionTraceRecordSlots;
        const auto span=[&](std::size_t offset,std::size_t count) {
            return std::span<const float>(trace.data()+base+offset,count);
        };
        std::cout<<"{\"index\":"<<iteration
            <<",\"record_flags\":"<<trace[base]
            <<",\"accepted_backtrack_index\":"<<trace[base+1u]
            <<",\"accepted_alpha\":"<<trace[base+2u]
            <<",\"local_solve_succeeded\":"<<((trace[base]==3.0f||trace[base]==7.0f)?"true":"false")
            <<",\"update_accepted\":"<<(trace[base]==7.0f?"true":"false")
            <<",\"residual_inf_scaled\":"<<trace[base+3u]
            <<",\"residual_inf_unscaled\":"<<trace[base+3u]/residualScale
            <<",\"state_inf\":"<<trace[base+4u]
            <<",\"candidate_state_high\":";
        writeArray(std::cout,span(6u,n));
        std::cout<<",\"candidate_state_low\":";
        writeArray(std::cout,span(6u+kProjectionTraceStateCapacity,n));
        std::cout<<",\"residual_rows_scaled\":";
        writeArray(std::cout,span(6u+2u*kProjectionTraceStateCapacity,n));
        std::cout<<",\"residual_row_scale\":";
        std::array<double,oracle::kStateCount> rowScale{};
        rowScale.fill(residualScale);
        writeArray(std::cout,std::span<const double>(rowScale.data(),n));
        std::cout<<",\"newton_delta\":";
        writeArray(std::cout,span(6u+3u*kProjectionTraceStateCapacity,n));
        std::cout<<",\"backtrack_residual_inf_scaled\":";
        writeArray(std::cout,span(6u+4u*kProjectionTraceStateCapacity,
                                  kProjectionTraceStateCapacity));
        std::cout<<",\"jacobian_row_major_scaled\":";
        writeArray(std::cout,span(6u+5u*kProjectionTraceStateCapacity,n*n));
        std::cout<<",\"candidate_after_high\":";
        writeArray(std::cout,span(6u+5u*kProjectionTraceStateCapacity+
            kProjectionTraceStateCapacity*kProjectionTraceStateCapacity,n));
        std::cout<<",\"candidate_after_low\":";
        writeArray(std::cout,span(6u+6u*kProjectionTraceStateCapacity+
            kProjectionTraceStateCapacity*kProjectionTraceStateCapacity,n));
        std::cout<<'}';
    }
    std::cout<<"],\"physical_steps\":0,\"physical_validation\":false,"
        <<"\"production_equations_unchanged\":true}\n";
}

void runCdUnloadProjectionTrace(MetalInstrument& metal,
                                const MaterialFixture& fixture,
                                const Curves& curves,
                                const CompiledWorld& world,
                                const std::string& digest) {
    const auto& program=world.constitutive.at(0);
    State initial{};
    for(unsigned i=0u;i<oracle::kStateCount;++i)
        initial[i]=world.stateInitials.at(i);
    const Matrix zeroDirection{};
    const Response rest=metal.evaluate(world,kIdentity,zeroDirection,initial);
    const State restState=rest.projectionValid?asDouble(rest):initial;

    const auto& cdCurve=curves.at({fixture.csvComponent,"CD"});
    Matrix cdLoadF=kIdentity;
    cdLoadF[4]=1.0+0.5*(cdCurve.yieldStrain+cdCurve.endpointStrain);
    const Response cdLoad=metal.evaluate(world,cdLoadF,zeroDirection,restState);
    const oracle::Point oracleLoad=oracle::project(program,cdLoadF,restState);
    std::cout<<std::setprecision(10)
        <<"{\"schema\":\"numi.cardboard.c1-local-projection-trace-setup.v1\","
        <<"\"material\":\""<<fixture.assetName<<"\",\"asset_sha256\":\""
        <<digest<<"\",\"source_csv_sha256\":\""<<kSourceCsvHash
        <<"\",\"compiled_material_local_newton_iterations\":"
        <<world.materials.at(0).localNewtonIterations<<','
        <<"\"compiled_generic_compensated_implicit\":"
        <<(gCompensatedProjection?"true":"false")
        <<",\"rest_projection_valid\":"<<(rest.projectionValid?"true":"false")
        <<",\"cd_load_projection_valid\":"<<(cdLoad.projectionValid?"true":"false")
        <<",\"cd_load_stress_tangent_valid\":"<<(cdLoad.valid?"true":"false")
        <<",\"cd_load_F_row_major\":";writeArray(std::cout,cdLoadF);
    std::cout<<",\"cd_load_native_state\":";writeArray(std::cout,cdLoad.state);
    std::cout<<",\"cd_load_fp64_state\":";writeArray(std::cout,oracleLoad.state);
    std::cout<<",\"cd_load_fp64_iterations\":"<<oracleLoad.iterations
        <<",\"cd_load_fp64_residual_inf\":"<<oracleLoad.residualInfinity
        <<",\"physical_steps\":0,\"diagnostic_only\":true}\n";
    if(!rest.projectionValid||!cdLoad.projectionValid) return;

    const State accepted=asDouble(cdLoad);
    const Response unload=metal.evaluate(world,kIdentity,zeroDirection,accepted,
                                         false,true);
    const oracle::Point oracleUnload=oracle::project(program,kIdentity,accepted);
    if(gMaterialIterationBudget<=kProjectionTraceIterationCapacity) {
        emitProjectionIterationTrace(fixture,world,accepted,unload,oracleUnload,digest);
    } else {
        std::cout<<std::setprecision(10)
            <<"{\"schema\":\"numi.cardboard.c1-local-projection-budget.v1\","
            <<"\"material\":\""<<fixture.assetName<<"\",\"asset_sha256\":\""
            <<digest<<"\",\"source_csv_sha256\":\""<<kSourceCsvHash
            <<"\",\"compiled_material_local_newton_iterations\":"
            <<world.materials.at(0).localNewtonIterations
            <<",\"projection_trace_capacity_iterations\":"
            <<kProjectionTraceIterationCapacity
            <<",\"projection_trace_truncated\":true"
            <<",\"compiled_generic_compensated_implicit\":"
            <<(gCompensatedProjection?"true":"false")
            <<",\"stage\":\"CD_unload_to_identity\",\"native_projection_valid\":"
            <<(unload.projectionValid?"true":"false")
            <<",\"native_stress_tangent_valid\":"<<(unload.valid?"true":"false")
            <<",\"native_terminal_failure\":"<<unload.failure
            <<",\"native_iterations\":"<<unload.iterations
            <<",\"native_residual_inf_scaled\":"<<unload.residual
            <<",\"native_state\":";writeArray(std::cout,unload.state);
        std::cout<<",\"fp64_iterations\":"<<oracleUnload.iterations
            <<",\"fp64_residual_inf\":"<<oracleUnload.residualInfinity
            <<",\"fp64_state\":";writeArray(std::cout,oracleUnload.state);
        std::cout<<",\"iteration_trace_supported\":false,\"physical_steps\":0,"
            "\"physical_validation\":false,\"diagnostic_only\":true}\n";
    }
    if(unload.projectionTrace[7]!=-1.0f || unload.iterations>0u || unload.projectionValid)
        ++gCompletedProjectionTraceMaterials;
}

void runMaterial(MetalInstrument& metal,const MaterialFixture& fixture,
                 const Curves& curves) {
    const std::string digest=sha256(fixture.path);
    require(digest==fixture.assetHash,fixture.role+" C1 transfer asset SHA256 does not match registered source identity");
    const auto parsed=parseMatterFile(fixture.path.string());
    require(parsed.succeeded(),fixture.role+" parse failed: "+diagnostics(parsed.diagnostics));
    const MaterialProgram& material=parsed.material;
    require(material.name==fixture.assetName,fixture.role+" parsed material name is not the registered asset");
    const CompiledWorld world=cook(material);
    const auto& program=world.constitutive.at(0);
    if(gProjectionIterationTrace) {
        runCdUnloadProjectionTrace(metal,fixture,curves,world,digest);
        return;
    }
    std::cout<<"{\"material\":\""<<material.name<<"\",\"asset_sha256\":\""<<digest
        <<"\",\"state_count\":15,\"source_csv_sha256\":\""<<kSourceCsvHash
        <<"\",\"cpu_digitized_slope_check\":\"separately_recorded_in_compiler_test\"}\n";

    State accepted{};
    for (unsigned i=0;i<oracle::kStateCount;++i) accepted[i]=world.stateInitials[i];
    const Matrix zeroDirection{};
    const Response rest=metal.evaluate(world,kIdentity,zeroDirection,accepted);
    const oracle::Point restExact=oracle::project(program,kIdentity,accepted);
    validateStateAndStress(material.name+"_rest",rest,restExact,accepted);
    validateFullTangent(material.name+"_rest",metal,world,kIdentity,accepted,restExact);
    accepted=asDouble(rest);

    // A one-ULP change in the weak through-thickness axis produces a small,
    // nonzero normalized principal stress inside the authored C1 transition.
    // This complements exact rest (all principal stresses zero) and verifies
    // that the native material-point path sees the regularized near-zero law.
    Matrix nearZeroF=kIdentity;
    nearZeroF[8]=static_cast<double>(std::nextafter(1.0f,2.0f));
    const Response nearZero=metal.evaluate(world,nearZeroF,zeroDirection,accepted);
    const oracle::Point nearZeroExact=oracle::project(program,nearZeroF,accepted);
    validateStateAndStress(material.name+"_near_zero_C1",nearZero,nearZeroExact,accepted);
    const double c1Width=oracle::parameter(program.material,"compression_smoothing_width");
    require(std::max({std::abs(nearZero.state[6]),std::abs(nearZero.state[7]),
                      std::abs(nearZero.state[8])})<c1Width,
            material.name+" near-zero specimen did not enter all three C1 principal branches");
    validateFullTangent(material.name+"_near_zero_C1",metal,world,
                        nearZeroF,accepted,nearZeroExact);

    // Compression is a distinct branch of the convex C1 radius. This fixture
    // only checks the material point; it does not claim a physical crease or
    // an independently measured compression curve.
    for(unsigned axis=0;axis<2u;++axis) {
        const unsigned component=axis==0u?0u:4u;
        const std::string axisName=axis==0u?"MD":"CD";
        Matrix compressionF=kIdentity;
        compressionF[component]=0.965;
        compressionF[axis==0u?4u:0u]=1.004;
        compressionF[8]=1.001;
        const Response compressed=metal.evaluate(world,compressionF,zeroDirection,accepted);
        const oracle::Point compressionExact=oracle::project(program,compressionF,accepted);
        validateStateAndStress(material.name+"_"+axisName+"_compression",compressed,
                               compressionExact,accepted);
        require(compressed.state[12]-accepted[12]>1.0e-8,
                material.name+" "+axisName+" compression did not reach plastic flow");
        require(compressed.state[axis==0u?6u:7u]<-c1Width,
                material.name+" "+axisName+" compression did not select its negative-stress branch");
        validateFullTangent(material.name+"_"+axisName+"_compression",metal,world,
                            compressionF,accepted,compressionExact);
        Matrix axisDirection{};axisDirection[component]=1.0;
        const Response tangentCenter=metal.evaluate(world,compressionF,axisDirection,accepted);
        validateBranchAwareFD(material.name+"_"+axisName+"_compression",metal,world,
                              compressionF,component,accepted,tangentCenter);
        if (fixture.role=="liner"&&axis==0u) {
            constexpr unsigned capturedDirection=2u;
            Matrix shearDirection{};shearDirection[capturedDirection]=1.0;
            const Response shearCenter=metal.evaluate(world,compressionF,
                                                        shearDirection,accepted);
            validateBranchAwareFD(material.name+"_MD_compression_F13",metal,
                                  world,compressionF,capturedDirection,accepted,
                                  shearCenter);
        }
    }

    for (unsigned axis=0;axis<2u;++axis) {
        const std::string axisName=axis==0u?"MD":"CD";
        const auto& curve=curves.at({fixture.csvComponent,axisName});
        const double strain=0.5*(curve.yieldStrain+curve.endpointStrain);
        Matrix loadedF=kIdentity;
        const unsigned component=axis==0u?0u:4u;
        loadedF[component]=1.0+strain;
        const Response loaded=metal.evaluate(world,loadedF,zeroDirection,accepted);
        const oracle::Point loadedExact=oracle::project(program,loadedF,accepted);
        validateStateAndStress(material.name+"_"+axisName+"_load",loaded,loadedExact,accepted);
        require(loaded.state[12]-accepted[12]>1.0e-8,
                material.name+" "+axisName+" load did not activate plastic flow");
        const unsigned channel=axis==0u?13u:14u;
        require(loaded.state[channel]>accepted[channel]+1.0e-8,
                material.name+" "+axisName+" tensile load did not update its directional hardening history");
        validateFullTangent(material.name+"_"+axisName+"_load",metal,world,
                            loadedF,accepted,loadedExact);
        Matrix axisDirection{}; axisDirection[component]=1.0;
        const Response tangentCenter=metal.evaluate(world,loadedF,axisDirection,accepted);
        validateBranchAwareFD(material.name+"_"+axisName+"_load",metal,world,
                              loadedF,component,accepted,tangentCenter);

        const State loadedState=asDouble(loaded);
        const bool traceUnload=gProjectionIterationTrace&&axisName=="CD";
        const Response unload=metal.evaluate(world,kIdentity,zeroDirection,
                                              loadedState,false,traceUnload);
        if(traceUnload) {
            const auto& tr=unload.projectionTrace;
            const unsigned n=static_cast<unsigned>(tr[1]);
            const unsigned iterationCount=static_cast<unsigned>(tr[2]);
            const double residualScale=oracle::parameter(
                program.material,"residual_scale");
            std::cout<<std::setprecision(10)
                <<"{\"schema\":\"numi.cardboard.c1-local-projection-trace.v1\","
                <<"\"material\":\""<<material.name<<"\",\"asset_sha256\":\""
                <<digest<<"\",\"axis\":\"CD\",\"stage\":\"unload_to_identity\","
                <<"\"F_row_major\":";writeArray(std::cout,kIdentity);
            std::cout<<",\"F_low_row_major\":";
            const Matrix zeroLow{};writeArray(std::cout,zeroLow);
            std::cout<<",\"accepted_state\":";writeArray(std::cout,loadedState);
            std::cout<<",\"trace_supported\":"<<(tr[7]!=-1.0f?"true":"false")
                <<",\"state_count\":"<<n<<",\"attempted_iterations\":"
                <<iterationCount<<",\"converged\":"<<(tr[3]==1.0f?"true":"false")
                <<",\"terminal_projection_failure\":"<<tr[7]
                <<",\"projection_tolerance\":"<<tr[6]
                <<",\"timestep_s\":1e-4,\"temperature_K\":293.15"
                <<",\"residual_scale_factor\":"<<residualScale
                <<",\"residual_row_names\":[\"ep11\",\"ep22\",\"ep33\",\"ep23\",\"ep13\",\"ep12\","
                  "\"sbar11\",\"sbar22\",\"sbar33\",\"sbar23\",\"sbar13\",\"sbar12\","
                  "\"accumulated_multiplier\",\"hardening_md_accumulation\",\"hardening_cd_accumulation\"]"
                <<",\"residual_units\":\"dimensionless; rows scaled by residual_scale_factor\""
                <<",\"final_residual_inf_scaled\":"<<tr[4]
                <<",\"final_residual_inf_unscaled\":"<<tr[4]/residualScale
                <<",\"final_state_norm_inf\":"<<tr[5]<<",\"iterations\":[";
            const unsigned copied=std::min<unsigned>(iterationCount,
                static_cast<unsigned>(kProjectionTraceIterationCapacity));
            for(unsigned iteration=0u;iteration<copied;++iteration) {
                if(iteration)std::cout<<',';
                const std::size_t base=kProjectionTraceHeaderSlots+
                    iteration*kProjectionTraceRecordSlots;
                const auto span=[&](std::size_t offset,std::size_t count) {
                    return std::span<const float>(tr.data()+base+offset,count);
                };
                std::cout<<"{\"index\":"<<iteration
                    <<",\"record_flags\":"<<tr[base]
                    <<",\"accepted_backtrack_index\":"<<tr[base+1u]
                    <<",\"accepted_alpha\":"<<tr[base+2u]
                    <<",\"local_solve_succeeded\":"<<((tr[base]==3.0f||tr[base]==7.0f)?"true":"false")
                    <<",\"update_accepted\":"<<(tr[base]==7.0f?"true":"false")
                    <<",\"residual_inf_scaled\":"<<tr[base+3u]
                    <<",\"residual_inf_unscaled\":"<<tr[base+3u]/residualScale
                    <<",\"state_inf\":"<<tr[base+4u]
                    <<",\"candidate_state_high\":";
                writeArray(std::cout,span(6u,n));
                std::cout<<",\"candidate_state_low\":";
                writeArray(std::cout,span(6u+kProjectionTraceStateCapacity,n));
                std::cout<<",\"residual_rows_scaled\":";
                writeArray(std::cout,span(6u+2u*kProjectionTraceStateCapacity,n));
                std::cout<<",\"residual_row_scale\":";
                std::array<double,oracle::kStateCount> rowScale{};
                rowScale.fill(residualScale);
                writeArray(std::cout,std::span<const double>(rowScale.data(),n));
                std::cout<<",\"newton_delta\":";
                writeArray(std::cout,span(6u+3u*kProjectionTraceStateCapacity,n));
                std::cout<<",\"backtrack_residual_inf_scaled\":";
                writeArray(std::cout,span(6u+4u*kProjectionTraceStateCapacity,
                                          kProjectionTraceStateCapacity));
                std::cout<<",\"jacobian_row_major_scaled\":";
                writeArray(std::cout,span(6u+5u*kProjectionTraceStateCapacity,n*n));
                std::cout<<",\"candidate_after_high\":";
                writeArray(std::cout,span(6u+5u*kProjectionTraceStateCapacity+
                    kProjectionTraceStateCapacity*kProjectionTraceStateCapacity,n));
                std::cout<<",\"candidate_after_low\":";
                writeArray(std::cout,span(6u+6u*kProjectionTraceStateCapacity+
                    kProjectionTraceStateCapacity*kProjectionTraceStateCapacity,n));
                std::cout<<'}';
            }
            std::cout<<"],\"physical_steps\":0,\"production_equations_unchanged\":true}\n";
        }
        const oracle::Point unloadExact=oracle::project(program,kIdentity,loadedState);
        validateStateAndStress(material.name+"_"+axisName+"_unload",unload,unloadExact,loadedState);
        const State unloadedState=asDouble(unload);
        const Response reload=metal.evaluate(world,loadedF,zeroDirection,unloadedState);
        const oracle::Point reloadExact=oracle::project(program,loadedF,unloadedState);
        validateStateAndStress(material.name+"_"+axisName+"_reload",reload,reloadExact,unloadedState);
        for (unsigned state=12u;state<15u;++state) {
            require(unload.state[state]+2.0e-7>=loaded.state[state],
                    material.name+" unloading lost irreversible state "+std::to_string(state));
            require(reload.state[state]+2.0e-7>=unload.state[state],
                    material.name+" reloading lost irreversible state "+std::to_string(state));
        }
        std::cout<<"{\"material\":\""<<material.name<<"\",\"axis\":\""<<axisName
            <<"\",\"source_yield_strain\":"<<curve.yieldStrain
            <<",\"source_digitized_endpoint_strain\":"<<curve.endpointStrain
            <<",\"instrument_strain\":"<<strain
            <<",\"history13_load_unload_reload\":["<<loaded.state[13]<<','<<unload.state[13]<<','<<reload.state[13]
            <<"],\"history14_load_unload_reload\":["<<loaded.state[14]<<','<<unload.state[14]<<','<<reload.state[14]
            <<"],\"multiplier_load_unload_reload\":["<<loaded.state[12]<<','<<unload.state[12]<<','<<reload.state[12]
            <<"],\"physical_steps\":0}\n";
    }
}

struct CPUWord { float high=0.0f; float low=0.0f; };

CPUWord cpuWordSum(const float a,const float b) {
    const float high=a+b;
    const float virtualB=high-a;
    return {high,(a-(high-virtualB))+(b-virtualB)};
}

CPUWord cpuWordAdd(const CPUWord a,const CPUWord b) {
    const auto high=cpuWordSum(a.high,b.high);
    const auto low=cpuWordSum(a.low,b.low);
    const auto middle=cpuWordSum(high.low,low.high);
    const auto merged=cpuWordSum(high.high,middle.high);
    const auto tail=cpuWordSum(merged.low,middle.low);
    const auto finalTail=cpuWordSum(tail.high,low.low+tail.low);
    return cpuWordSum(merged.high,finalTail.high+finalTail.low);
}

CPUWord cpuWordNegate(const CPUWord value) { return {-value.high,-value.low}; }

CPUWord cpuWordMultiply(const CPUWord a,const CPUWord b) {
    const float high=a.high*b.high;
    const float low=std::fma(a.high,b.high,-high)+a.high*b.low+
        a.low*b.high+a.low*b.low;
    return cpuWordSum(high,low);
}

CPUWord cpuWordDivide(const CPUWord a,const CPUWord b) {
    const float first=a.high/b.high;
    const auto remainder=cpuWordAdd(a,cpuWordNegate(cpuWordMultiply(b,{first,0.0f})));
    return cpuWordAdd({first,0.0f},{(remainder.high+remainder.low)/b.high,0.0f});
}

CPUWord splitDoubleWord(const double value) {
    const float high=static_cast<float>(value);
    const float low=static_cast<float>(value-static_cast<double>(high));
    return {high,low};
}

bool solveTwoWordLocalSystem(std::array<CPUWord,oracle::kStateCount*
                              oracle::kStateCount>& matrix,
                             std::array<CPUWord,oracle::kStateCount>& rhs) {
    constexpr unsigned count=oracle::kStateCount;
    for (unsigned column=0u;column<count;++column) {
        unsigned pivot=column;
        float magnitude=std::abs(matrix[column*count+column].high+
                                 matrix[column*count+column].low);
        for (unsigned row=column+1u;row<count;++row) {
            const auto value=matrix[row*count+column];
            const float candidate=std::abs(value.high+value.low);
            if (candidate>magnitude) { pivot=row; magnitude=candidate; }
        }
        if (!(magnitude>1.0e-8f)||!std::isfinite(magnitude)) return false;
        if (pivot!=column) {
            for (unsigned entry=column;entry<count;++entry)
                std::swap(matrix[column*count+entry],matrix[pivot*count+entry]);
            std::swap(rhs[column],rhs[pivot]);
        }
        const CPUWord diagonal=matrix[column*count+column];
        for (unsigned row=column+1u;row<count;++row) {
            const CPUWord factor=cpuWordDivide(matrix[row*count+column],diagonal);
            matrix[row*count+column]={0.0f,0.0f};
            for (unsigned entry=column+1u;entry<count;++entry)
                matrix[row*count+entry]=cpuWordAdd(matrix[row*count+entry],
                    cpuWordNegate(cpuWordMultiply(factor,matrix[column*count+entry])));
            rhs[row]=cpuWordAdd(rhs[row],cpuWordNegate(cpuWordMultiply(factor,rhs[column])));
        }
    }
    for (int row=int(count)-1;row>=0;--row) {
        CPUWord value=rhs[unsigned(row)];
        for (unsigned column=unsigned(row)+1u;column<count;++column)
            value=cpuWordAdd(value,cpuWordNegate(cpuWordMultiply(
                matrix[unsigned(row)*count+column],rhs[column])));
        rhs[unsigned(row)]=cpuWordDivide(
            value,matrix[unsigned(row)*count+unsigned(row)]);
        if (!std::isfinite(rhs[unsigned(row)].high)||
            !std::isfinite(rhs[unsigned(row)].low)) return false;
    }
    return true;
}

bool solveFloatLocalSystem(std::array<float,oracle::kStateCount*
                            oracle::kStateCount>& matrix,
                           std::array<float,oracle::kStateCount>& rhs) {
    constexpr unsigned count=oracle::kStateCount;
    for (unsigned column=0u;column<count;++column) {
        unsigned pivot=column;
        float magnitude=std::abs(matrix[column*count+column]);
        for (unsigned row=column+1u;row<count;++row) {
            const float candidate=std::abs(matrix[row*count+column]);
            if (candidate>magnitude) { pivot=row; magnitude=candidate; }
        }
        if (!(magnitude>1.0e-8f)||!std::isfinite(magnitude)) return false;
        if (pivot!=column) {
            for (unsigned entry=column;entry<count;++entry)
                std::swap(matrix[column*count+entry],matrix[pivot*count+entry]);
            std::swap(rhs[column],rhs[pivot]);
        }
        const float diagonal=matrix[column*count+column];
        for (unsigned row=column+1u;row<count;++row) {
            const float factor=matrix[row*count+column]/diagonal;
            matrix[row*count+column]=0.0f;
            for (unsigned entry=column+1u;entry<count;++entry)
                matrix[row*count+entry]-=factor*matrix[column*count+entry];
            rhs[row]-=factor*rhs[column];
        }
    }
    for (int row=int(count)-1;row>=0;--row) {
        float value=rhs[unsigned(row)];
        for (unsigned column=unsigned(row)+1u;column<count;++column)
            value-=matrix[unsigned(row)*count+column]*rhs[column];
        rhs[unsigned(row)]=value/matrix[unsigned(row)*count+unsigned(row)];
        if (!std::isfinite(rhs[unsigned(row)])) return false;
    }
    return true;
}

void runCpuTangentCancellation(const std::filesystem::path& path) {
    const auto parsed=parseMatterFile(path.string());
    require(parsed.succeeded(),"CPU cancellation material parse failed: "+
            diagnostics(parsed.diagnostics));
    require(parsed.material.name=="hajali2009_liner_hill_asymmetric_c1_transfer",
            "CPU cancellation specimen must use the source-bound liner C1 transfer asset");
    const auto compiled=detail::compileConstitutive(parsed.material,
                                                     NM_EXPRESSION_STACK_CAPACITY);
    require(compiled.succeeded(),"CPU cancellation material compile failed: "+
            diagnostics(compiled.diagnostics));
    const auto& program=compiled.program;
    Matrix f{kIdentity};
    f[0]=0.9649999738; f[4]=1.003999949; f[8]=1.001000047;
    Matrix h{};
    h[2]=1.0; // Captured failing F13 tangent direction, not an axial proxy.
    State accepted{};
    for(unsigned i=0;i<oracle::kStateCount;++i)
        accepted[i]=program.material.internalState[i].initialValue;
    // Use the exact native candidate from capture130 so this arithmetic test
    // probes the same C1 branch and cancellation scale. This is CPU evidence,
    // not a replacement for the subsequent native checker.
    const State candidate{-0.01866336726,-0.0118428329,0.000988249667,
        0.0,0.0,0.0,-3.059290648,1.203199863,-2.080524609e-06,
        0.0,0.0,0.0,0.02333543636,0.0,0.02333543636};
    const auto linearization=oracle::linearize(program,f,h,accepted,candidate);
    const Matrix exactChain=oracle::tangent(program,f,h,accepted,candidate);
    Matrix cdLoadF=kIdentity;
    cdLoadF[4]=1.0+0.5*(0.009+0.0448717949);
    const auto cdLoadReference=oracle::project(program,cdLoadF,accepted);
    const auto cdUnloadReference=oracle::project(program,kIdentity,
                                                 cdLoadReference.state);

    std::array<float,oracle::kStateCount*oracle::kStateCount> fp32Jacobian{};
    std::array<float,oracle::kStateCount> fp32Direction{};
    std::array<CPUWord,oracle::kStateCount*oracle::kStateCount> wordJacobian{};
    std::array<CPUWord,oracle::kStateCount> wordDirection{};
    for(unsigned row=0;row<oracle::kStateCount;++row) {
        fp32Direction[row]=static_cast<float>(-
            linearization.deformationResidualDerivative[row]);
        wordDirection[row]=splitDoubleWord(
            -linearization.deformationResidualDerivative[row]);
        for(unsigned column=0;column<oracle::kStateCount;++column) {
            const unsigned index=row*oracle::kStateCount+column;
            fp32Jacobian[index]=static_cast<float>(
                linearization.jacobian[index]);
            wordJacobian[index]=splitDoubleWord(linearization.jacobian[index]);
        }
    }
    require(solveFloatLocalSystem(fp32Jacobian,fp32Direction),
            "CPU FP32 tangent solve failed");
    require(solveTwoWordLocalSystem(wordJacobian,wordDirection),
            "CPU two-word tangent solve failed");
    std::cout<<std::setprecision(12)
        <<"{\"schema\":\"numi.cardboard.asymmetric-c1-cpu-cancellation.v1\","
        <<"\"material\":\""<<program.material.name<<"\","
        <<"\"asset_sha256\":\""<<sha256(path)<<"\","
        <<"\"F_row_major\":";writeArray(std::cout,f);
    std::cout<<",\"H_row_major\":";writeArray(std::cout,h);
    std::cout<<",\"candidate_state\":";writeArray(std::cout,candidate);
    std::cout<<",\"local_linearization_residual_inf\":"
        <<oracle::normInfinity(linearization.residual)
        <<",\"cpu_CD_load_unload_reference\":{\"load_iterations\":"
        <<cdLoadReference.iterations<<",\"load_residual_inf\":"
        <<cdLoadReference.residualInfinity<<",\"unload_iterations\":"
        <<cdUnloadReference.iterations<<",\"unload_residual_inf\":"
        <<cdUnloadReference.residualInfinity<<"},\"directions\":[";
    bool first=true;
    double mdExact=0.0,mdRounded=0.0;
    double worstTwoWordError=0.0;
    for(const unsigned component : {2u,6u}) {
        CPUWord stateTerm{0.0f,0.0f};
        float roundedStateTerm=0.0f;
        for(unsigned state=0;state<oracle::kStateCount;++state) {
            const double derivative=linearization.stressStateDerivative[
                component*oracle::kStateCount+state];
            const float derivativeFloat=static_cast<float>(derivative);
            roundedStateTerm+=derivativeFloat*fp32Direction[state];
            stateTerm=cpuWordAdd(stateTerm,cpuWordMultiply(
                splitDoubleWord(derivative),wordDirection[state]));
        }
        const double fixed=linearization.fixedStateTangent[component];
        const double exactTotal=exactChain[component];
        const auto fixedWord=splitDoubleWord(fixed);
        const auto ddTotal=cpuWordAdd(fixedWord,stateTerm);
        const double compensatedTotal=static_cast<double>(ddTotal.high)+ddTotal.low;
        const float roundedFixed=static_cast<float>(fixed);
        const float fp32Total=roundedFixed+roundedStateTerm;
        const double fp32Error=std::abs(double(fp32Total)-exactTotal)/
                               std::max(1.0,std::abs(exactTotal));
        const double compensatedError=std::abs(compensatedTotal-exactTotal)/
            std::max(1.0,std::abs(exactTotal));
        worstTwoWordError=std::max(worstTwoWordError,compensatedError);
        if(component==2u){mdExact=exactTotal;mdRounded=fp32Total;}
        if(!first)std::cout<<',';
        first=false;
        std::cout<<"{\"P_component\":"<<component
            <<",\"fixed_state_term_Pa_per_F\":"<<fixed
            <<",\"fp32_implicit_state_term_Pa_per_F\":"<<roundedStateTerm
            <<",\"oracle_total_Pa_per_F\":"<<exactTotal
            <<",\"fp32_fixed_state_term_Pa_per_F\":"<<roundedFixed
            <<",\"fp32_accumulated_total_Pa_per_F\":"<<fp32Total
            <<",\"fp32_relative_cancellation_error\":"<<fp32Error
            <<",\"two_word_local_direction_Pa_per_F\":"<<compensatedTotal
            <<",\"two_word_relative_error\":"<<compensatedError<<'}';
    }
    std::cout<<"],\"native_evidence\":false,\"purpose\":\"reproduce_cancellation_sensitivity_not_qualify_Metal\"}\n";
    require(std::abs(mdExact)>0.0&&std::abs(mdRounded-mdExact)/std::abs(mdExact)>1.0e-2&&
            std::abs(mdRounded-mdExact)/std::abs(mdExact)<5.0e-2,
            "CPU C1 cancellation specimen no longer exposes the expected FP32 loss");
    require(worstTwoWordError<1.0e-6,
            "CPU two-word local solve/contraction did not recover the independent FP64 tangent");
    require(cdUnloadReference.residualInfinity<1.0e-8,
            "independent FP64 CD unload projection did not converge");
}

} // namespace

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            if(argc==3&&std::string(argv[1])=="--cpu-tangent-cancellation") {
                runCpuTangentCancellation(argv[2]);
                return 0;
            }
            if (argc==5 && std::string(argv[1])=="--cpu-source-fit-diagnose") {
                const std::filesystem::path linerPath=argv[2],mediumPath=argv[3],csvPath=argv[4];
                require(sha256(csvPath)==kSourceCsvHash,
                        "CPU fit diagnostic registered CSV SHA256 changed");
                const Curves curves=readSourceCurves(csvPath);
                for (const auto& entry : {
                    std::pair<std::filesystem::path,std::string>{linerPath,"liner"},
                    {mediumPath,"medium_NSSC"}}) {
                    const auto parsed=parseMatterFile(entry.first.string());
                    require(parsed.succeeded(),"CPU fit diagnostic parse failed: "+
                            diagnostics(parsed.diagnostics));
                    const auto compiled=detail::compileConstitutive(parsed.material,
                                                                     NM_EXPRESSION_STACK_CAPACITY);
                    require(compiled.succeeded(),"CPU fit diagnostic compile failed: "+
                            diagnostics(compiled.diagnostics));
                    for (const auto& axis : {std::string("MD"),std::string("CD")}) {
                        const auto& curve=curves.at({entry.second,axis});
                        const double span=curve.endpointStrain-curve.yieldStrain;
                        const double aStrain=curve.yieldStrain+0.30*span;
                        const double bStrain=curve.yieldStrain+0.70*span;
                        const unsigned axisIndex=axis=="MD"?0u:1u;
                        const double a=cpuStressFreeAxialMPa(compiled.program,axisIndex,aStrain);
                        const double b=cpuStressFreeAxialMPa(compiled.program,axisIndex,bStrain);
                        const double observed=(b-a)/(bStrain-aStrain);
                        std::cout<<std::setprecision(10)
                            <<"{\"material\":\""<<parsed.material.name
                            <<"\",\"axis\":\""<<axis<<"\",\"observed_slope_MPa\":"<<observed
                            <<",\"digitized_slope_MPa\":"<<curve.tangentMPa
                            <<",\"digitized_interval_MPa\":["<<curve.tangentLowMPa<<','
                            <<curve.tangentHighMPa<<"],\"inside_interval\":"
                            <<(observed>=curve.tangentLowMPa&&observed<=curve.tangentHighMPa?"true":"false")
                            <<",\"diagnostic_only\":true}\n";
                    }
                }
                return 0;
            }
            bool diagnosticCapture=false;
            bool materialIterationsSpecified=false;
            for(int index=5;index<argc;++index) {
                const std::string option=argv[index];
                if(option=="--diagnostic-tangent-capture") {
                    require(!diagnosticCapture,"duplicate --diagnostic-tangent-capture option");
                    diagnosticCapture=true;
                } else if(option=="--diagnostic-cd-unload-trace") {
                    require(!gProjectionIterationTrace,"duplicate --diagnostic-cd-unload-trace option");
                    gProjectionIterationTrace=true;
                } else if(option=="--compensated-implicit") {
                    require(!gCompensatedProjection,"duplicate --compensated-implicit option");
                    gCompensatedProjection=true;
                } else if(option=="--material-iterations") {
                    require(!materialIterationsSpecified,
                        "duplicate --material-iterations option");
                    require(index+1<argc,"--material-iterations requires 16, 24, or 32");
                    const std::string value=argv[++index];
                    unsigned parsed=0u;
                    const auto [end,error]=std::from_chars(
                        value.data(),value.data()+value.size(),parsed);
                    require(error==std::errc{} && end==value.data()+value.size() &&
                            (parsed==16u||parsed==24u||parsed==32u),
                        "--material-iterations must be exactly 16, 24, or 32");
                    gMaterialIterationBudget=parsed;
                    materialIterationsSpecified=true;
                } else {
                    throw std::runtime_error("unknown option: "+option);
                }
            }
            require(argc>=5&&argc<=10,
                "usage: asymmetric-c1-paper-gpu-check LINER_ASSET MEDIUM_ASSET SOURCE_FIT_CSV C1_TRANSFER_CSV [--diagnostic-tangent-capture] [--diagnostic-cd-unload-trace] [--compensated-implicit] [--material-iterations 16|24|32]");
            require(!(diagnosticCapture&&gProjectionIterationTrace),
                "--diagnostic-tangent-capture and --diagnostic-cd-unload-trace select separate diagnostic paths");
            gDetailedTangentDiagnostic=diagnosticCapture;
            const std::filesystem::path linerPath=argv[1],mediumPath=argv[2],csvPath=argv[3],transferPath=argv[4];
            const std::string csvDigest=sha256(csvPath);
            require(csvDigest==kSourceCsvHash,"registered Haj-Ali digitized source CSV SHA256 changed");
            const std::string transferDigest=sha256(transferPath);
            require(transferDigest==kTransferCsvHash,"registered C1 transfer CSV SHA256 changed");
            const Curves curves=readSourceCurves(csvPath);
            std::cout<<std::setprecision(10)
                <<"{\"schema\":\"numi.cardboard.asymmetric-c1-native-material-point.v1\","
                <<"\"source_csv_sha256\":\""<<csvDigest
                <<"\",\"transfer_csv_sha256\":\""<<transferDigest
                <<"\",\"evidence_scope\":\"production_Metal_generic_projection_stress_tangent_only\","
                <<"\"physical_steps\":0,\"physical_validation\":false,"
                <<"\"compiled_material_local_newton_iterations\":"<<gMaterialIterationBudget<<','
                <<"\"projection_trace_capacity_iterations\":"<<kProjectionTraceIterationCapacity<<','
                <<"\"projection_trace_truncated\":"
                <<(gMaterialIterationBudget>kProjectionTraceIterationCapacity?"true":"false")<<','
                <<"\"diagnostic_tangent_capture_requested\":"<<(diagnosticCapture?"true":"false")<<','
                <<"\"diagnostic_cd_unload_trace_requested\":"<<(gProjectionIterationTrace?"true":"false")<<','
                <<"\"compiled_generic_compensated_implicit_requested\":"
                <<(gCompensatedProjection?"true":"false")<<','
                <<"\"native_nonzero_gate\":"<<kRelativeGate
                <<",\"zero_response_policy\":\"inconclusive_unless_measured_noise_floor_is_recorded\"}\n";
            MetalInstrument metal;
            runMaterial(metal,{"liner","hajali2009_liner_hill_asymmetric_c1_transfer",kLinerHash,
                               "liner",linerPath},curves);
            runMaterial(metal,{"medium","hajali2009_medium_hill_asymmetric_c1_transfer",kMediumHash,
                               "medium_NSSC",mediumPath},curves);
            if(gProjectionIterationTrace) {
                if(gCompletedProjectionTraceMaterials!=2u) {
                    std::cout<<"{\"schema\":\"numi.cardboard.asymmetric-c1-diagnostic-result.v1\","
                        <<"\"status\":\"incomplete_projection_trace\",\"completed_materials\":"
                        <<gCompletedProjectionTraceMaterials<<",\"required_materials\":2,"
                        <<"\"physical_steps\":0,\"physical_validation\":false}\n";
                    return 2;
                }
                std::cout<<"TRACE_CAPTURED diagnostic_only=true physical_steps=0 physical_validation=false"
                    <<" compensated_implicit="<<(gCompensatedProjection?"true":"false")<<'\n';
                return 0;
            }
            if(diagnosticCapture)
                require(gObservedTangentCapture,
                    "diagnostic request did not reach the selected liner MD compression tangent");
            if(gObservedTangentMismatch) {
                std::cout<<"{\"schema\":\"numi.cardboard.asymmetric-c1-diagnostic-result.v1\","
                    <<"\"status\":\"captured_tangent_gate_failure\",\"capture_emitted\":"
                    <<(gObservedTangentCapture?"true":"false")<<",\"physical_steps\":0}\n";
                return 2;
            }
            std::cout<<"PASS native asymmetric C1 15-state generic material-point checks; physical_validation=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr<<"FAIL C1 native material-point check: "<<error.what()<<'\n';
            return 1;
        }
    }
}
