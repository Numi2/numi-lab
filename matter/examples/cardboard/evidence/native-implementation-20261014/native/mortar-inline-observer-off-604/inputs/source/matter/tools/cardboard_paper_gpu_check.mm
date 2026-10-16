#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "numi/matter/matter.hpp"
#include "cardboard_paper_reference.hpp"
#include "cardboard_solver_reference.hpp"
#include "cardboard_paper_projection_scaling.hpp"
#include "numi/matter/fiber_exp_linear.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cctype>
#include <limits>
#include <cstring>

namespace {
using namespace numi::matter;
using Matrix=std::array<double,9>;
constexpr Matrix identity{1,0,0,0,1,0,0,0,1};
void need(bool v,const std::string& s){if(!v)throw std::runtime_error(s);}
std::string diagnostics(const std::vector<Diagnostic>& ds){std::string s;for(const auto& d:ds)s+=d.message+"; ";return s;}
double parameter(const MaterialProgram& m,const std::string& n){for(const auto& p:m.parameters)if(p.name==n)return p.defaultValue;throw std::runtime_error("missing "+n);}
template<class T> id<MTLBuffer> buffer(id<MTLDevice> d,const std::vector<T>& x){T z{};return [d newBufferWithBytes:x.empty()?&z:x.data() length:sizeof(T)*std::max(std::size_t(1),x.size()) options:MTLResourceStorageModeShared];}
struct Response{bool valid=false;bool projectionValid=false;Matrix p{},dp{};std::vector<float> state,stateLow;
    float residual=0,stateNorm=0;unsigned iterations=0,flags=0,failure=0;};
constexpr unsigned kReplayStateCount=13u;
constexpr unsigned kReplayCandidateCount=4u;
constexpr std::array<const char*,kReplayCandidateCount> kReplayCandidateNames{
    "captured_production","ordinary_replay","accurate_candidate","fp64_root"};
struct StateWords {
    std::array<float,NM_MAX_MATERIAL_STATE> high{};
    std::array<float,NM_MAX_MATERIAL_STATE> low{};
    bool available=false;
};
struct CompiledResidualRows {
    bool valid=false;
    bool nonfinite=false;
    std::array<float,NM_MAX_MATERIAL_STATE> high{};
    std::array<float,NM_MAX_MATERIAL_STATE> low{};
};
using ResidualRowCandidates=std::array<CompiledResidualRows,kReplayCandidateCount>;
numi_cardboard_paper::Parameters capturedFP64Parameters(
    const std::vector<float>& values,numi_cardboard_paper::MaterialKind kind);

struct F64BytecodeResult { bool valid=false; std::string failure; double value=0.0; };

// Independent binary64 interpreter over the exact compiled instruction
// stream. Constants and parameters retain their compiled FP32 bit patterns;
// only arithmetic is promoted to double. This is a diagnostic of lowering
// and constant selection, not a bitwise Metal emulator.
F64BytecodeResult evaluateCompiledProgramFP64(
    const CompiledWorld& world,const NMMaterialGPU& material,
    const std::uint32_t programIndex,const std::vector<float>& accepted,
    const StateWords& candidate,
    const Matrix& deformation,const Matrix& deformationLow,const Matrix& deformationRate,
    const double timestep=1.0e-4,const double temperature=293.15) {
    F64BytecodeResult result;
    if(programIndex==NM_INVALID_INDEX||programIndex>=world.scalarPrograms.size()){
        result.failure="invalid_program_index";return result;
    }
    const NMScalarProgramGPU& program=world.scalarPrograms[programIndex];
    const bool useLowWords=(program.flags&2u)!=0u;
    if(program.maximumStack>NM_EXPRESSION_STACK_CAPACITY||
       program.firstInstruction>world.instructions.size()||
       program.instructionCount>world.instructions.size()-program.firstInstruction||
       accepted.size()<material.stateCount||material.stateCount>NM_MAX_MATERIAL_STATE||
       !(timestep>=0.0)||!std::isfinite(timestep)||!std::isfinite(temperature)){
        result.failure="invalid_program_or_context";return result;
    }
    std::array<double,NM_EXPRESSION_STACK_CAPACITY> stack{};
    std::array<double,NM_EXPRESSION_CACHE_CAPACITY> cache{};
    std::array<bool,NM_EXPRESSION_CACHE_CAPACITY> cacheWritten{};
    unsigned depth=0u;
    bool valid=true;
    const auto push=[&](double value)->bool{
        if(depth>=NM_EXPRESSION_STACK_CAPACITY||!std::isfinite(value)){
            valid=false;return false;
        }
        stack[depth++]=value;return true;
    };
    const auto pop=[&]()->double{
        if(depth==0u){valid=false;return 0.0;}
        return stack[--depth];
    };
    const double domainFloor=double(1.0e-12f);
    for(unsigned local=0u;local<program.instructionCount&&valid;++local){
        const NMExpressionInstructionGPU instruction=
            world.instructions[program.firstInstruction+local];
        switch(instruction.opcode){
        case NM_EXPR_CONSTANT:push(double(instruction.immediate.x));break;
        case NM_EXPR_PARAMETER:{
            const std::size_t index=std::size_t(material.parameterOffset)+instruction.index;
            if(instruction.index>=material.parameterCount||index>=world.parameters.size())
                valid=false;
            else push(double(world.parameters[index].valueAndBounds.x));
            break;}
        case NM_EXPR_F:
            if(instruction.index>=9u)valid=false;
            else push(deformation[instruction.index]+
                (useLowWords?deformationLow[instruction.index]:0.0));
            break;
        case NM_EXPR_DF:
            if(instruction.index>=9u)valid=false;else push(0.0);break;
        case NM_EXPR_RATE:
            if(instruction.index>=9u)valid=false;
            else push(deformationRate[instruction.index]);
            break;
        case NM_EXPR_DT:push(timestep);break;
        case NM_EXPR_TEMPERATURE:push(temperature);break;
        case NM_EXPR_STATE:
            if(instruction.index>=material.stateCount||instruction.index>=accepted.size())
                valid=false;
            else push(double(accepted[instruction.index]));
            break;
        case NM_EXPR_NEXT_STATE:
            if(instruction.index>=material.stateCount)valid=false;
            else push(double(candidate.high[instruction.index])+
                (useLowWords?double(candidate.low[instruction.index]):0.0));
            break;
        case NM_EXPR_ADD:{const double b=pop(),a=pop();push(a+b);break;}
        case NM_EXPR_SUBTRACT:{const double b=pop(),a=pop();push(a-b);break;}
        case NM_EXPR_MULTIPLY:{const double b=pop(),a=pop();push(a*b);break;}
        case NM_EXPR_DIVIDE:{const double b=pop(),a=pop();
            if(!(std::abs(b)>domainFloor))valid=false;else push(a/b);break;}
        case NM_EXPR_NEGATE:push(-pop());break;
        case NM_EXPR_LOG:{const double a=pop();if(!(a>0.0))valid=false;else push(std::log(a));break;}
        case NM_EXPR_EXP:push(std::exp(pop()));break;
        case NM_EXPR_EXPM1_MINUS_X:{const double a=pop();push(std::exp(a)-1.0-a);break;}
        case NM_EXPR_SQRT:{const double a=pop();if(a<0.0)valid=false;else push(std::sqrt(a));break;}
        case NM_EXPR_ABS:push(std::abs(pop()));break;
        case NM_EXPR_MIN:{const double b=pop(),a=pop();push(std::min(a,b));break;}
        case NM_EXPR_MAX:{const double b=pop(),a=pop();push(std::max(a,b));break;}
        case NM_EXPR_POW_INTEGER:{
            const double base=pop();const int exponent=instruction.integer;
            if(exponent<0&&!(std::abs(base)>domainFloor)){valid=false;break;}
            unsigned power=static_cast<unsigned>(exponent<0?-exponent:exponent);
            double factor=base,value=1.0;
            while(power!=0u){if((power&1u)!=0u)value*=factor;factor*=factor;power>>=1u;}
            push(exponent<0?1.0/value:value);break;}
        case NM_EXPR_CLAMP:{const double upper=pop(),lower=pop(),value=pop();
            if(upper<lower)valid=false;else push(std::clamp(value,lower,upper));break;}
        case NM_EXPR_SELECT_NONNEGATIVE:{const double negative=pop(),nonnegative=pop(),condition=pop();
            push(condition>=0.0?nonnegative:negative);break;}
        case NM_EXPR_CACHE_STORE:
            if((program.flags&1u)==0u||instruction.index>=cache.size()||depth==0u)valid=false;
            else{cache[instruction.index]=stack[depth-1u];cacheWritten[instruction.index]=true;}
            break;
        case NM_EXPR_CACHE_LOAD:
            if((program.flags&1u)==0u||instruction.index>=cache.size()||
               !cacheWritten[instruction.index])valid=false;
            else push(cache[instruction.index]);break;
        case NM_EXPR_FIBER_EXP_LINEAR:{
            const double limit=pop(),c5=pop(),c4=pop(),c3=pop(),stretch=pop();
            bool fiberValid=true;
            const double value=numi_matter_fiber::evaluate<double>(stretch,c3,c4,c5,
                limit,static_cast<unsigned>(instruction.integer),fiberValid);
            if(!fiberValid)valid=false;else push(value);
            break;}
        default:result.failure="unsupported_opcode_"+std::to_string(instruction.opcode);valid=false;break;
        }
    }
    if(valid&&depth==1u&&std::isfinite(stack[0])){
        result.valid=true;result.value=stack[0];
    }else if(result.failure.empty())result.failure="bytecode_domain_or_stack_failure";
    return result;
}

struct CompiledResidualFP64 {
    bool valid=false;
    std::array<double,NM_MAX_MATERIAL_STATE> rows{};
    std::string failure;
};
CompiledResidualFP64 evaluateCompiledResidualsFP64(
    const CompiledWorld& world,const std::array<float,NM_MAX_MATERIAL_STATE>& accepted,
    const StateWords& state,const Matrix& f,const Matrix& fLow,const Matrix& rate,
    const float timestep) {
    CompiledResidualFP64 out;
    if(!state.available){out.failure="candidate_unavailable";return out;}
    if(world.materials.empty()||world.materials[0].stateCount!=kReplayStateCount){
        out.failure="unsupported_material_state_count";return out;
    }
    const NMMaterialGPU& material=world.materials[0];
    std::vector<float> acceptedVector(accepted.begin(),accepted.begin()+kReplayStateCount);
    for(unsigned row=0u;row<kReplayStateCount;++row){
        const auto value=evaluateCompiledProgramFP64(world,material,
            material.implicitResidualProgramOffset+row,acceptedVector,state,f,fLow,rate,
            static_cast<double>(timestep));
        if(!value.valid){out.failure="row_"+std::to_string(row)+":"+value.failure;return out;}
        out.rows[row]=value.value;
    }
    out.valid=true;return out;
}

std::array<double,NM_MAX_MATERIAL_STATE> sourceResidualRowsFP64(
    const std::vector<float>& capturedParameters,
    numi_cardboard_paper::MaterialKind kind,const Matrix& f,const Matrix& fLow,
    const std::array<float,NM_MAX_MATERIAL_STATE>& accepted,const StateWords& candidate,
    const bool useLowWords) {
    Matrix exactF{};
    for(unsigned i=0u;i<9u;++i)exactF[i]=double(f[i])+
        (useLowWords?double(fLow[i]):0.0);
    numi_cardboard_paper::State accepted64{},candidate64{};
    for(unsigned i=0u;i<kReplayStateCount;++i){
        accepted64[i]=double(accepted[i]);
        candidate64[i]=double(candidate.high[i])+
            (useLowWords?double(candidate.low[i]):0.0);
    }
    const auto params=capturedFP64Parameters(capturedParameters,kind);
    const auto rows=numi_cardboard_paper::residual(params,exactF,accepted64,candidate64);
    std::array<double,NM_MAX_MATERIAL_STATE> out{};
    std::copy(rows.begin(),rows.end(),out.begin());return out;
}

std::string floatBitsHex(float value){
    std::uint32_t bits=0u;std::memcpy(&bits,&value,sizeof(bits));
    std::ostringstream out;out<<"0x"<<std::hex<<std::setw(8)<<std::setfill('0')<<bits;
    return out.str();
}
bool compiledMaterialUsesOpcode(const CompiledWorld& world,const std::uint32_t opcode){
    for(const auto& program:world.scalarPrograms){
        const std::size_t first=program.firstInstruction;
        const std::size_t count=program.instructionCount;
        if(first>world.instructions.size()||count>world.instructions.size()-first)continue;
        for(std::size_t local=0u;local<count;++local)
            if(world.instructions[first+local].opcode==opcode)return true;
    }
    return false;
}
struct Instrument {
    id<MTLDevice> device;id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline,residualRowsPipeline;
    Instrument(){device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal unavailable");
        const std::string name=[[device name] UTF8String];need(name.find("Apple")!=std::string::npos&&name.find("Paravirtual")==std::string::npos,"physical Apple GPU required");
        queue=[device newCommandQueue];NSError* error=nil;
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&error];need(lib!=nil,"paper instrument library missing");
        pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cardboard_paper_check"] error:&error];need(pipeline!=nil,"paper instrument kernel missing");
        residualRowsPipeline=[device newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"cardboard_paper_compiled_residual_rows"] error:&error];
        need(residualRowsPipeline!=nil,"compiled residual-row instrument kernel missing");
        std::cout<<"device="<<name<<" production_projection_and_algorithmic_tangent=true physical_steps=0\n";
    }
    Response evaluate(const CompiledWorld& w,const Matrix& f,const Matrix& fLow,
                      const Matrix& h,const std::vector<float>& state,bool accurateProjection,
                      const float timestep=1.0e-4f,const Matrix& deformationRate=Matrix{}){
        std::vector<float> inputs;for(double v:f)inputs.push_back(float(v));
        for(double v:h)inputs.push_back(float(v));
        for(double v:fLow)inputs.push_back(float(v));
        for(double v:deformationRate)inputs.push_back(float(v));
        inputs.push_back(timestep);
        inputs.push_back(accurateProjection?1.0f:0.0f);
        std::vector<float> parameters;for(const auto& v:w.parameters)parameters.push_back(v.valueAndBounds.x);
        std::array<id<MTLBuffer>,7> b{buffer(device,std::vector<NMMaterialGPU>{w.materials.at(0)}),buffer(device,w.scalarPrograms),buffer(device,w.instructions),buffer(device,parameters),buffer(device,state),buffer(device,inputs),buffer(device,std::vector<float>(25+2*NM_MAX_MATERIAL_STATE,0))};
        auto cb=[queue commandBuffer];auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:pipeline];
        for(NSUInteger i=0;i<b.size();++i)[encoder setBuffer:b[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[encoder endEncoding];[cb commit];[cb waitUntilCompleted];
        need(cb.status==MTLCommandBufferStatusCompleted,"Metal paper evaluation failed");
        const auto* out=static_cast<const float*>(b[6].contents);Response r;r.valid=out[0]==1;r.projectionValid=out[19+NM_MAX_MATERIAL_STATE]==1;
        r.residual=out[20+NM_MAX_MATERIAL_STATE];r.stateNorm=out[21+NM_MAX_MATERIAL_STATE];
        r.iterations=static_cast<unsigned>(out[22+NM_MAX_MATERIAL_STATE]);
        r.flags=static_cast<unsigned>(out[23+NM_MAX_MATERIAL_STATE]);
        r.failure=static_cast<unsigned>(out[24+NM_MAX_MATERIAL_STATE]);
        for(unsigned i=0;i<9;++i){r.p[i]=out[1+i];r.dp[i]=out[10+i];}
        r.state.assign(out+19,out+19+w.materials.at(0).stateCount);
        r.stateLow.assign(out+25+NM_MAX_MATERIAL_STATE,
                          out+25+NM_MAX_MATERIAL_STATE+w.materials.at(0).stateCount);
        return r;
    }
    ResidualRowCandidates evaluateCompiledResidualRows(
        const CompiledWorld& w,const Matrix& f,const Matrix& fLow,
        const std::array<float,NM_MAX_MATERIAL_STATE>& accepted,
        const std::array<StateWords,kReplayCandidateCount>& candidates,
        const Matrix& deformationRate,const float timestep) {
        std::vector<float> inputs;inputs.reserve(40u+kReplayCandidateCount*65u+1u);
        for(double value:f)inputs.push_back(static_cast<float>(value));
        for(double value:fLow)inputs.push_back(static_cast<float>(value));
        for(double value:deformationRate)inputs.push_back(static_cast<float>(value));
        for(unsigned i=0u;i<kReplayStateCount;++i)inputs.push_back(accepted[i]);
        for(const StateWords& candidate:candidates){
            inputs.push_back(candidate.available?1.0f:0.0f);
            inputs.insert(inputs.end(),candidate.high.begin(),candidate.high.end());
            inputs.insert(inputs.end(),candidate.low.begin(),candidate.low.end());
        }
        inputs.push_back(timestep);
        std::vector<float> parameters;parameters.reserve(w.parameters.size());
        for(const auto& value:w.parameters)parameters.push_back(value.valueAndBounds.x);
        const std::size_t stride=2u+2u*NM_MAX_MATERIAL_STATE;
        std::vector<float> resultValues(kReplayCandidateCount*stride,0.0f);
        std::array<id<MTLBuffer>,7> b{
            buffer(device,std::vector<NMMaterialGPU>{w.materials.at(0)}),
            buffer(device,w.scalarPrograms),buffer(device,w.instructions),
            buffer(device,parameters),buffer(device,std::vector<float>(accepted.begin(),accepted.end())),
            buffer(device,inputs),buffer(device,resultValues)};
        auto cb=[queue commandBuffer];auto encoder=[cb computeCommandEncoder];
        [encoder setComputePipelineState:residualRowsPipeline];
        for(NSUInteger i=0;i<b.size();++i)[encoder setBuffer:b[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(kReplayCandidateCount,1,1)
            threadsPerThreadgroup:MTLSizeMake(kReplayCandidateCount,1,1)];
        [encoder endEncoding];[cb commit];[cb waitUntilCompleted];
        need(cb.status==MTLCommandBufferStatusCompleted,
             "compiled residual-row Metal evaluation failed");
        const auto* out=static_cast<const float*>(b[6].contents);
        ResidualRowCandidates rows{};
        for(unsigned candidate=0u;candidate<kReplayCandidateCount;++candidate){
            const std::size_t base=candidate*stride;
            rows[candidate].valid=out[base]==1.0f;
            rows[candidate].nonfinite=out[base+1u]==1.0f;
            for(unsigned row=0u;row<NM_MAX_MATERIAL_STATE;++row){
                rows[candidate].high[row]=out[base+2u+row];
                rows[candidate].low[row]=out[base+2u+NM_MAX_MATERIAL_STATE+row];
            }
        }
        return rows;
    }
    Response evaluate(const CompiledWorld& w,const Matrix& f,const Matrix& h,
                      const std::vector<float>& state){
        return evaluate(w,f,Matrix{},h,state,false);
    }
};
CompiledWorld cook(const MaterialProgram& m){
    WorldSource w;w.materials={m};w.gravity={0,0,0};w.environmentCount=1;w.frameTimestep=1e-4;
    ObjectSource o;o.name="paper_constitutive_instrument";o.representation=Representation::fem;o.deformableContact=false;o.deformableSelfContact=false;o.mixedFEM=false;
    o.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}}};o.tetrahedra={{{0,1,2,3}}};o.characteristicLength=.01;w.objects={o};
    const auto result=compileWorld(w,{.maximumRateExponent=0,.emitSpecializedMetal=false,.localMaterialNewtonIterations=16u});need(result.succeeded(),diagnostics(result.diagnostics));return result.world;
}
double maxabs(const Matrix& a){double v=0;for(double x:a)v=std::max(v,std::abs(x));return v;}
double plastic(const Response& r){double x=0;for(unsigned i=0;i<6;++i)x=std::max(x,std::abs(double(r.state.at(i))));return x;}
std::array<double,6> finalNormalizedStress(const MaterialProgram& m,const Matrix& f,const Response& r){
    const double det=f[0]*(f[4]*f[8]-f[5]*f[7])-f[1]*(f[3]*f[8]-f[5]*f[6])+f[2]*(f[3]*f[7]-f[4]*f[6]);
    need(det>0,"invalid instrument deformation");
    Matrix inv{f[4]*f[8]-f[5]*f[7],f[2]*f[7]-f[1]*f[8],f[1]*f[5]-f[2]*f[4],
        f[5]*f[6]-f[3]*f[8],f[0]*f[8]-f[2]*f[6],f[2]*f[3]-f[0]*f[5],
        f[3]*f[7]-f[4]*f[6],f[1]*f[6]-f[0]*f[7],f[0]*f[4]-f[1]*f[3]};
    Matrix stress{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)for(unsigned k=0;k<3;++k)stress[3*i+j]+=inv[3*i+k]*r.p[3*k+j]/det/parameter(m,"sigma_ref");
    return {stress[0],stress[4],stress[8],.5*(stress[5]+stress[7]),.5*(stress[2]+stress[6]),.5*(stress[1]+stress[3])};
}
double yieldMeasure(const MaterialProgram& m,const std::array<double,6>& s){const auto sq=[](double x){return x*x;};return std::sqrt(parameter(m,"hill_F")*sq(s[1]-s[2])+parameter(m,"hill_G")*sq(s[2]-s[0])+parameter(m,"hill_H")*sq(s[0]-s[1])+2*parameter(m,"hill_L")*sq(s[3])+2*parameter(m,"hill_M")*sq(s[4])+2*parameter(m,"hill_N")*sq(s[5]));}

struct ExactMetrics {
    bool branchMatches=false,tangentBranchPreserved=false;
    bool oracleProjectionValid=false,tangentReferenceProjectionValid=false;
    std::string oracleProjectionFailure,tangentReferenceProjectionFailure;
    unsigned oracleProjectionIteration=0u,oracleProjectionWorstRow=0u;
    unsigned oracleProjectionLineSearchTrials=0u;
    double oracleProjectionResidual=0.0,oracleProjectionStateNorm=0.0;
    double oracleProjectionWorstResidual=0.0,oracleProjectionLastAlpha=0.0;
    double oracleProjectionLastTrialResidual=0.0,oracleProjectionBestTrialResidual=0.0;
    bool tangentReferenceProjectionFailed=false;
    unsigned tangentProjectionIteration=0u,tangentProjectionWorstRow=0u;
    unsigned tangentProjectionLineSearchTrials=0u,tangentProjectionSample=0u;
    double tangentProjectionResidual=0.0,tangentProjectionWorstResidual=0.0;
    double tangentProjectionLastAlpha=0.0,tangentProjectionLastTrialResidual=0.0;
    double tangentProjectionBestTrialResidual=0.0;
    std::string metalBranch,oracleBranch;
    bool tangentGate=false,zeroDirection=false;
    double firstPiolaRelative=0.0,stateRelative=0.0,tangentRelative=0.0;
    double zeroDirectionTangentError=0.0;
    unsigned resolvedFirstPiola=0u,resolvedState=0u,resolvedTangent=0u;
    unsigned nearZeroFirstPiola=0u,nearZeroState=0u,nearZeroTangent=0u;
    double maxNearZeroFirstPiolaAbsoluteError=0.0;
    double maxNearZeroStateAbsoluteError=0.0;
    double maxNearZeroTangentAbsoluteError=0.0;
    double tangentFiniteDifferenceNoiseRelative=0.0;
    double maxTangentFiniteDifferenceNoiseAbsolute=0.0;
    bool tangentFiniteDifferenceNoiseResolved=false;
    double firstPiolaResolutionCutoff=0.0,stateResolutionCutoff=0.0;
    double tangentResolutionCutoff=0.0;
    Matrix firstPiolaAbsoluteError{},tangentAbsoluteError{};
    std::array<double,13> stateAbsoluteError{};
    numi_cardboard_paper::State oracleState{};
    Matrix oracleFirstPiola{},oracleTangent{};
    Matrix oracleImplicitTangentPaPerS{};
    bool oracleImplicitTangentValid=false;
    std::string oracleImplicitTangentFailure;
    double implicitTangentNormPaPerS=0.0;
    double implicitTangentMaxAbsoluteErrorPaPerS=0.0;
    double implicitTangentNormRelativeError=0.0;
    double fp64ResidualAtMetalState=0.0,twoWordStateResidualFloor=0.0;
    double residualAbsoluteDifference=0.0,residualRelativeToFloor=0.0;
};

struct ExactProjectionAttempt {
    bool valid=false;
    std::string failure="not_started";
    unsigned iteration=0u,worstRow=0u,lineSearchTrials=0u;
    double residual=0.0,stateNorm=0.0,worstResidual=0.0;
    double lastAlpha=0.0,lastTrialResidual=0.0,bestTrialResidual=0.0;
    numi_cardboard_paper::Result result{};
};

numi_cardboard_paper::Parameters capturedFP64Parameters(
    const std::vector<float>& values,numi_cardboard_paper::MaterialKind kind){
    need(values.size()==20,"exact capture requires the 20-entry paper parameter table");
    auto p=numi_cardboard_paper::parameters(kind);
    p.density=values[0];
    for(unsigned i=0;i<6;++i)p.c[i]=double(values[1+i])/1.0e6;
    for(unsigned i=0;i<3;++i)p.shear[i]=double(values[7+i])/1.0e6;
    p.sigmaReference=double(values[10])/1.0e6;
    for(unsigned i=0;i<6;++i)p.hill[i]=values[11+i];
    p.qFloor=values[17];
    p.fischerBurmeisterEpsilon=values[18];
    return p;
}

ExactProjectionAttempt projectExactParameters(
    const numi_cardboard_paper::Parameters& p,const Matrix& f,
    const numi_cardboard_paper::State& accepted){
    using namespace numi_cardboard_paper;
    ExactProjectionAttempt attempt;
    State x=accepted;
    const Vector6 trial=normalizedSecondPiola(p,f,plasticStrain(accepted));
    for(unsigned i=0;i<6;++i)x[6+i]=trial[i];
    unsigned iterations=0;
    constexpr unsigned maxIterations=64;
    constexpr double tolerance=1.0e-12;
    for(;iterations<maxIterations;++iterations){
        const auto r=residual(p,f,accepted,x);
        const double norm=residualNorm(r);
        double stateScale=1.0;
        for(double value:x)stateScale=std::max(stateScale,1.0+std::abs(value));
        attempt.iteration=iterations;
        attempt.residual=norm;
        attempt.stateNorm=stateScale;
        attempt.worstRow=0u;
        for(unsigned row=1u;row<13u;++row)
            if(std::abs(r[row])>std::abs(r[attempt.worstRow]))attempt.worstRow=row;
        attempt.worstResidual=r[attempt.worstRow];
        if(norm<=tolerance*stateScale){
            attempt.valid=true;
            attempt.failure.clear();
            break;
        }
        const auto jac=numi_cardboard_solver::implicitJacobian(p,accepted,x);
        std::array<double,13> rhs{},step{};
        for(unsigned i=0;i<13;++i)rhs[i]=-r[i];
        if(!solveLinear(jac,rhs,step)){
            attempt.failure="singular_analytic_implicit_jacobian";
            break;
        }
        double alpha=1.0;
        bool acceptedStep=false;
        attempt.bestTrialResidual=std::numeric_limits<double>::infinity();
        attempt.lineSearchTrials=0u;
        for(unsigned backtrack=0;backtrack<24;++backtrack){
            State candidate=x;
            for(unsigned i=0;i<13;++i)candidate[i]+=alpha*step[i];
            const double trialNorm=residualNorm(residual(p,f,accepted,candidate));
            attempt.lineSearchTrials=backtrack+1u;
            attempt.lastAlpha=alpha;
            attempt.lastTrialResidual=trialNorm;
            if(trialNorm<attempt.bestTrialResidual)
                attempt.bestTrialResidual=trialNorm;
            if(std::isfinite(trialNorm)&&trialNorm<=
               (1.0-1.0e-4*alpha)*norm){
                x=candidate;acceptedStep=true;break;
            }
            alpha*=0.5;
        }
        if(!acceptedStep){
            attempt.failure="armijo_line_search_failed";
            break;
        }
    }
    const double finalNorm=residualNorm(residual(p,f,accepted,x));
    double stateScale=1.0;
    for(double value:x)stateScale=std::max(stateScale,1.0+std::abs(value));
    attempt.iteration=iterations;
    attempt.residual=finalNorm;
    attempt.stateNorm=stateScale;
    if(!attempt.valid&&finalNorm<=tolerance*stateScale){
        attempt.valid=true;
        attempt.failure.clear();
    }else if(!attempt.valid&&attempt.failure=="not_started"){
        attempt.failure="newton_iteration_limit";
    }
    if(!attempt.valid)return attempt;
    const Vector6 ep=plasticStrain(x),s=returnedNormalizedStress(p,f,accepted,x);
    Vector6 dep{};
    for(unsigned i=0;i<6;++i)dep[i]=x[i]-accepted[i];
    const Matrix second=secondPiola(p,f,ep),first=multiply(f,second);
    const double work=second[0]*dep[0]+second[4]*dep[1]+second[8]*dep[2]+
        2.0*(second[5]*dep[3]+second[2]*dep[4]+second[1]*dep[5]);
    attempt.result={x,second,first,hillEquivalent(p,s),finalNorm,work,iterations};
    return attempt;
}

ExactMetrics compareWithExactInput(const Matrix& f,const Matrix& fLow,
                                   const Matrix& h,const std::vector<float>& accepted,
                                   const Response& metal,const MaterialProgram& material,
                                   const std::vector<float>& capturedParameters,
                                   numi_cardboard_paper::MaterialKind kind){
    using namespace numi_cardboard_paper;
    Matrix exactF=f;
    for(unsigned i=0;i<9;++i)exactF[i]+=fLow[i];
    State accepted64{};
    for(unsigned i=0;i<13;++i)accepted64[i]=accepted.at(i);
    const auto oracleParameters=capturedFP64Parameters(capturedParameters,kind);
    const auto oracleAttempt=projectExactParameters(oracleParameters,exactF,accepted64);
    State metalState{};
    for(unsigned i=0;i<material.internalState.size();++i)
        metalState[i]=double(metal.state.at(i))+double(metal.stateLow.at(i));

    ExactMetrics result;
    result.oracleProjectionValid=oracleAttempt.valid;
    result.oracleProjectionFailure=oracleAttempt.failure;
    result.oracleProjectionIteration=oracleAttempt.iteration;
    result.oracleProjectionWorstRow=oracleAttempt.worstRow;
    result.oracleProjectionLineSearchTrials=oracleAttempt.lineSearchTrials;
    result.oracleProjectionResidual=oracleAttempt.residual;
    result.oracleProjectionStateNorm=oracleAttempt.stateNorm;
    result.oracleProjectionWorstResidual=oracleAttempt.worstResidual;
    result.oracleProjectionLastAlpha=oracleAttempt.lastAlpha;
    result.oracleProjectionLastTrialResidual=oracleAttempt.lastTrialResidual;
    result.oracleProjectionBestTrialResidual=oracleAttempt.bestTrialResidual;
    if(!oracleAttempt.valid){
        result.oracleBranch="reference_projection_failed";
        result.metalBranch="not_compared_reference_projection_failed";
        result.tangentReferenceProjectionFailure="center_projection_failed";
        return result;
    }
    const auto& oracle=oracleAttempt.result;
    double pScale=1.0,stateScale=1.0;
    for(double x:oracle.firstPiola)pScale=std::max(pScale,std::abs(x*1.0e6));
    for(double x:oracle.state)stateScale=std::max(stateScale,std::abs(x));
    const double pFloor=64.0*std::numeric_limits<float>::epsilon()*pScale;
    const double stateFloor=64.0*std::numeric_limits<float>::epsilon()*stateScale;
    result.firstPiolaResolutionCutoff=pFloor;
    result.stateResolutionCutoff=stateFloor;
    result.oracleState=oracle.state;
    result.oracleFirstPiola=oracle.firstPiola;
    if(metal.valid){
        for(unsigned i=0;i<9;++i){
            const double expected=oracle.firstPiola[i]*1.0e6;
            const double error=std::abs(double(metal.p[i])-expected);
            result.firstPiolaAbsoluteError[i]=error;
            if(std::abs(expected)>pFloor){
                result.firstPiolaRelative=std::max(result.firstPiolaRelative,error/std::abs(expected));
                ++result.resolvedFirstPiola;
            }else{
                ++result.nearZeroFirstPiola;
                result.maxNearZeroFirstPiolaAbsoluteError=
                    std::max(result.maxNearZeroFirstPiolaAbsoluteError,error);
            }
        }
    }
    for(unsigned i=0;i<material.internalState.size();++i){
        const double expected=oracle.state[i];
        const double error=std::abs(metalState[i]-expected);
        result.stateAbsoluteError[i]=error;
        if(std::abs(expected)>stateFloor){
            result.stateRelative=std::max(result.stateRelative,error/std::abs(expected));
            ++result.resolvedState;
        }else{
            ++result.nearZeroState;
            result.maxNearZeroStateAbsoluteError=
                std::max(result.maxNearZeroStateAbsoluteError,error);
        }
    }

    const auto branch=[](double dl,double margin){
        if(margin<1.0e-4)return std::string("yield_boundary_unresolved");
        return dl>1.0e-7?std::string("plastic"):std::string("elastic");
    };
    const double oracleDelta=oracle.state[12]-accepted64[12];
    const double qMargin=std::abs(1.0-oracle.equivalentStress);
    result.oracleBranch=branch(oracleDelta,qMargin);

    double directionScale=0.0;
    for(double value:h)directionScale=std::max(directionScale,std::abs(value));
    result.zeroDirection=directionScale==0.0;
    // Keep the complete implicit derivative separate from a centered ray,
    // which may cross the narrow complementarity transition.
    try {
        const Matrix analytic=numi_cardboard_solver::directionalTangent(
            oracleParameters,exactF,h,accepted64,oracle.state);
        result.oracleImplicitTangentValid=true;
        for(unsigned i=0u;i<9u;++i){
            const double value=analytic[i]*1.0e6;
            need(std::isfinite(value),"nonfinite independent implicit tangent");
            result.oracleImplicitTangentPaPerS[i]=value;
            result.implicitTangentNormPaPerS=std::max(
                result.implicitTangentNormPaPerS,std::abs(value));
            result.implicitTangentMaxAbsoluteErrorPaPerS=std::max(
                result.implicitTangentMaxAbsoluteErrorPaPerS,
                std::abs(double(metal.dp[i])-value));
        }
        if(result.implicitTangentNormPaPerS>0.0)
            result.implicitTangentNormRelativeError=
                result.implicitTangentMaxAbsoluteErrorPaPerS/
                result.implicitTangentNormPaPerS;
    }catch(const std::exception& error){
        result.oracleImplicitTangentValid=false;
        result.oracleImplicitTangentFailure=error.what();
    }
    if(result.zeroDirection){
        result.zeroDirectionTangentError=maxabs(metal.dp);
        result.tangentBranchPreserved=true;
        // A zero perturbation cannot validate a directional derivative, even
        // when both the reported tangent and its error are exactly zero.
        result.tangentGate=false;
    }else{
        constexpr std::array<double,2> deformationSteps{2.0e-6,1.0e-6};
        std::array<Matrix,2> tangentByStep{};
        result.tangentBranchPreserved=result.oracleBranch!="yield_boundary_unresolved";
        result.tangentReferenceProjectionValid=true;
        for(unsigned sample=0;sample<deformationSteps.size();++sample){
            const double step=deformationSteps[sample]/directionScale;
            Matrix plus=exactF,minus=exactF;
            for(unsigned i=0;i<9;++i){plus[i]+=step*h[i];minus[i]-=step*h[i];}
            const auto plusAttempt=projectExactParameters(oracleParameters,plus,accepted64);
            const auto minusAttempt=projectExactParameters(oracleParameters,minus,accepted64);
            if(!plusAttempt.valid||!minusAttempt.valid){
                const auto& failed=!plusAttempt.valid?plusAttempt:minusAttempt;
                result.tangentReferenceProjectionValid=false;
                result.tangentReferenceProjectionFailed=true;
                result.tangentProjectionSample=sample;
                result.tangentProjectionIteration=failed.iteration;
                result.tangentProjectionWorstRow=failed.worstRow;
                result.tangentProjectionLineSearchTrials=failed.lineSearchTrials;
                result.tangentProjectionResidual=failed.residual;
                result.tangentProjectionWorstResidual=failed.worstResidual;
                result.tangentProjectionLastAlpha=failed.lastAlpha;
                result.tangentProjectionLastTrialResidual=failed.lastTrialResidual;
                result.tangentProjectionBestTrialResidual=failed.bestTrialResidual;
                result.tangentReferenceProjectionFailure=
                    std::string(!plusAttempt.valid?"plus":"minus")+"_sample_"+
                    std::to_string(sample)+":"+failed.failure;
                result.tangentBranchPreserved=false;
                break;
            }
            const auto& oraclePlus=plusAttempt.result;
            const auto& oracleMinus=minusAttempt.result;
            const auto plusBranch=branch(oraclePlus.state[12]-accepted64[12],
                                         std::abs(1.0-oraclePlus.equivalentStress));
            const auto minusBranch=branch(oracleMinus.state[12]-accepted64[12],
                                          std::abs(1.0-oracleMinus.equivalentStress));
            result.tangentBranchPreserved=result.tangentBranchPreserved&&
                plusBranch==result.oracleBranch&&minusBranch==result.oracleBranch;
            for(unsigned i=0;i<9;++i)
                tangentByStep[sample][i]=(oraclePlus.firstPiola[i]-oracleMinus.firstPiola[i])*
                    1.0e6/(2.0*step);
        }
        // Centered-difference step halving provides a directly observed noise
        // estimate. Richardson extrapolation is only used after both scales
        // remain on the same smooth constitutive branch.
        if(!result.tangentReferenceProjectionValid)return result;
        double tangentScale=1.0;
        Matrix tangent{};
        for(unsigned i=0;i<9;++i){
            tangent[i]=(4.0*tangentByStep[1][i]-tangentByStep[0][i])/3.0;
            const double noise=std::abs(tangentByStep[1][i]-tangentByStep[0][i]);
            result.maxTangentFiniteDifferenceNoiseAbsolute=std::max(
                result.maxTangentFiniteDifferenceNoiseAbsolute,noise);
            tangentScale=std::max(tangentScale,std::abs(tangent[i]));
        }
        const double tangentFloor=64.0*std::numeric_limits<float>::epsilon()*tangentScale;
        result.tangentResolutionCutoff=tangentFloor;
        for(unsigned i=0;i<9;++i)if(std::abs(tangent[i])>tangentFloor){
            const double noise=std::abs(tangentByStep[1][i]-tangentByStep[0][i]);
            result.tangentRelative=std::max(result.tangentRelative,
                std::abs(double(metal.dp[i])-tangent[i])/std::abs(tangent[i]));
            result.tangentFiniteDifferenceNoiseRelative=std::max(
                result.tangentFiniteDifferenceNoiseRelative,noise/std::abs(tangent[i]));
            ++result.resolvedTangent;
            result.tangentAbsoluteError[i]=std::abs(double(metal.dp[i])-tangent[i]);
            result.oracleTangent[i]=tangent[i];
        }else{
            ++result.nearZeroTangent;
            result.tangentAbsoluteError[i]=std::abs(double(metal.dp[i])-tangent[i]);
            result.oracleTangent[i]=tangent[i];
            result.maxNearZeroTangentAbsoluteError=std::max(
                result.maxNearZeroTangentAbsoluteError,result.tangentAbsoluteError[i]);
        }
        result.tangentFiniteDifferenceNoiseResolved=
            result.resolvedTangent>0u&&result.tangentFiniteDifferenceNoiseRelative<=1.0e-3;
        result.tangentGate=result.tangentBranchPreserved&&result.resolvedTangent>0u&&
            result.tangentFiniteDifferenceNoiseResolved&&result.tangentRelative<=1.0e-3;
    }

    const double metalDelta=metalState[12]-accepted64[12];
    // Recompute q from the Metal candidate using the same documented FP64
    // Hill equation; this only classifies the branch, not the Metal solve.
    Vector6 metalStress{};
    for(unsigned i=0;i<6;++i)metalStress[i]=metalState[6+i];
    const double metalQ=hillEquivalent(oracleParameters,metalStress);
    result.metalBranch=branch(metalDelta,std::abs(1.0-metalQ));
    result.branchMatches=result.oracleBranch==result.metalBranch &&
        result.oracleBranch!="yield_boundary_unresolved";

    const double residualScale=capturedParameters.at(19);
    result.fp64ResidualAtMetalState=residualScale*residualNorm(
        residual(oracleParameters,exactF,accepted64,metalState));
    State roundedOracle{};
    for(unsigned i=0;i<13;++i){
        const float high=static_cast<float>(oracle.state[i]);
        const float low=static_cast<float>(oracle.state[i]-double(high));
        roundedOracle[i]=double(high)+double(low);
    }
    result.twoWordStateResidualFloor=residualScale*residualNorm(
        residual(oracleParameters,exactF,accepted64,roundedOracle));
    result.residualAbsoluteDifference=std::abs(
        double(metal.residual)-result.fp64ResidualAtMetalState);
    const double residualScaleFloor=std::max(
        result.twoWordStateResidualFloor,
        16.0*std::numeric_limits<double>::epsilon()*residualScale);
    result.residualRelativeToFloor=result.residualAbsoluteDifference/residualScaleFloor;
    return result;
}

void check(Instrument& gpu,const std::string& path){
    const auto parsed=parseMatterFile(path);need(parsed.succeeded(),diagnostics(parsed.diagnostics));const auto& m=parsed.material;const auto w=cook(m);
    const auto kind=m.name.find("liner")!=std::string::npos?numi_cardboard_paper::MaterialKind::liner:numi_cardboard_paper::MaterialKind::medium;
    need(w.materials[0].stateCount==13,"expected paper state layout");const auto initial=w.stateInitials;
    Matrix direction{.013,.007,.004,-.002,-.009,.003,.002,-.004,.006};
    auto rest=gpu.evaluate(w,identity,direction,initial);need(rest.valid,std::string("initial material validity failed; projection=")+(rest.projectionValid?"true":"false"));need(plastic(rest)<1e-7&&maxabs(rest.p)<100,"initial paper stress/history not zero");
    struct Case{const char* name;Matrix f;};std::vector<Case> cases;
    Matrix f=identity;f[0]=std::sqrt(1-.002);cases.push_back({"md_elastic",f});f[0]=m.name.find("liner")!=std::string::npos?.994:.9975;cases.push_back({"md_plastic",f});
    f=identity;f[1]=.004;cases.push_back({"shear12_plastic",f});f=identity;f[2]=.0003;cases.push_back({"shear13_plastic",f});f=identity;f[5]=.0003;cases.push_back({"shear23_plastic",f});
    for(const auto& c:cases){auto r=gpu.evaluate(w,c.f,direction,initial);
        std::cout<<m.name<<" case="<<c.name<<" valid="<<r.valid; if(!r.valid){std::cout<<'\n';throw std::runtime_error(std::string("local return failed at ")+c.name);}
        const Matrix rotation{0,-1,0,1,0,0,0,0,1};
        const auto rotatedF=numi_cardboard_paper::multiply(rotation,c.f);
        const auto rotatedH=numi_cardboard_paper::multiply(rotation,direction);
        const auto rotated=gpu.evaluate(w,rotatedF,rotatedH,initial);
        need(rotated.valid,"rotated constitutive projection rejected");
        const auto expectedP=numi_cardboard_paper::multiply(rotation,r.p);
        double rotationError=0,historyError=0;
        for(unsigned i=0;i<9;++i)rotationError=std::max(rotationError,std::abs(rotated.p[i]-expectedP[i]));
        for(unsigned i=0;i<6;++i)historyError=std::max(historyError,std::abs(double(rotated.state[i])-r.state[i]));
        need(rotationError/std::max(1.0,maxabs(r.p))<1e-3&&historyError<1e-6,"paper response violates rotation objectivity");
        const auto oracle=numi_cardboard_paper::project(kind,c.f,{});
        double stressError=0,stressScale=1,stateError=0;
        for(unsigned i=0;i<9;++i){stressError=std::max(stressError,std::abs(r.p[i]-oracle.firstPiola[i]*1e6));stressScale=std::max(stressScale,std::abs(oracle.firstPiola[i]*1e6));}
        for(unsigned i=0;i<6;++i)stateError=std::max(stateError,std::abs(double(r.state[i])-oracle.state[i]));
        std::cout<<" oracle_stress_relative="<<stressError/stressScale<<" oracle_Ep_absolute="<<stateError;
        need(stressError/stressScale<.01&&stateError<2e-5,"GPU disagrees with independent FP64 constitutive oracle");
        const auto finalStress=finalNormalizedStress(m,c.f,r);const double q=yieldMeasure(m,finalStress);double work=0;for(unsigned i=0;i<6;++i)work+=(i<3?1:2)*finalStress[i]*parameter(m,"sigma_ref")*r.state[i];
        Matrix plus=c.f,minus=c.f;constexpr double eps=1e-3;for(unsigned i=0;i<9;++i){plus[i]+=eps*direction[i];minus[i]-=eps*direction[i];}
        const auto rp=gpu.evaluate(w,plus,direction,initial),rm=gpu.evaluate(w,minus,direction,initial);need(rp.valid&&rm.valid,"FD projection failed plus="+std::to_string(rp.valid)+" minus="+std::to_string(rm.valid)+" plus_projection="+std::to_string(rp.projectionValid)+" minus_projection="+std::to_string(rm.projectionValid)+" at "+c.name);
        double error=0,scale=1;for(unsigned i=0;i<9;++i){const double fd=(rp.p[i]-rm.p[i])/(2*eps);error=std::max(error,std::abs(fd-r.dp[i]));scale=std::max(scale,std::abs(fd));}
        std::cout<<" q="<<q<<" max_Ep="<<plastic(r)<<" plastic_work_J_m3="<<work<<" tangent_fd_relative="<<error/scale<<'\n';
        need(q<=1.003,"projected stress violates Hill surface");need(work>=-1e-3,"negative plastic dissipation");need(error/scale<.03,"algorithmic tangent fails centered FD");
        if(std::string(c.name)!="md_elastic")need(plastic(r)>1e-7,"loaded paper did not yield");
    }
    // Unload to zero elastic strain, not to an imposed flat total strain.
    // Use a pure MD sequence and verify retained plastic memory independently
    // of displayed geometry. E=Ep implies C=I+2Ep, so Cholesky gives F^T F.
    auto loaded=gpu.evaluate(w,cases[1].f,direction,initial);Matrix c=identity;
    c[0]+=2*loaded.state[0];c[4]+=2*loaded.state[1];c[8]+=2*loaded.state[2];c[5]=c[7]=2*loaded.state[3];c[2]=c[6]=2*loaded.state[4];c[1]=c[3]=2*loaded.state[5];
    Matrix lower{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<=i;++j){double sum=c[3*i+j];for(unsigned k=0;k<j;++k)sum-=lower[3*i+k]*lower[3*j+k];lower[3*i+j]=i==j?std::sqrt(sum):sum/lower[3*j+j];}
    Matrix relaxed{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)relaxed[3*i+j]=lower[3*j+i];
    const auto unload=gpu.evaluate(w,relaxed,direction,loaded.state);need(unload.valid,"unload projection failed");
    double drift=0;for(unsigned i=0;i<6;++i)drift=std::max(drift,std::abs(double(unload.state[i])-loaded.state[i]));
    std::cout<<m.name<<" unload_stress_Pa="<<maxabs(unload.p)<<" plastic_memory_drift="<<drift<<" retained_Ep="<<plastic(unload)<<'\n';
    need(plastic(unload)>1e-7&&drift<2e-5&&maxabs(unload.p)<5000,"plastic memory unload check failed");
}

int replayCaptured(Instrument& gpu,const char* inputPath,const char* outputPath,
                   const char* linerPath,const char* mediumPath,bool scaled,
                   bool initialized,bool compensated,bool accurateProjection) {
    std::ifstream input(inputPath);need(input.good(),"cannot read captured material input");
    need(!std::filesystem::exists(outputPath),"captured replay output already exists");
    const auto originalPosition=input.tellg();
    bool exactFormat=false;
    std::string line;
    while(std::getline(input,line)) {
        if(line.empty())continue;
        if(line.rfind("# numi.cardboard.paper-exact-capture.v1",0)==0)exactFormat=true;
        break;
    }
    input.clear();input.seekg(originalPosition);
    need(!exactFormat||accurateProjection,
         "paper-exact-capture.v1 input requires --accurate-projection");
    std::ofstream output(outputPath);need(output.good(),"cannot create captured material output");
    std::array<CompiledWorld,2> worlds;
    const char* paths[]{linerPath,mediumPath};
    for(unsigned i=0;i<2;++i) {
        const auto parsed=parseMatterFile(paths[i]);need(parsed.succeeded(),diagnostics(parsed.diagnostics));
        auto material=scaled?numi::cardboard::scalePaperProjectionResiduals(parsed.material):parsed.material;
        material.paperHillScalarInitializer=initialized;
        material.paperCompensatedArithmetic=compensated||accurateProjection;
        material.paperHillAccurateProjection=accurateProjection;
        worlds[i]=cook(material);
    }
    unsigned rows=0,failed=0,comparisonNonpass=0,comparisonHardFailed=0,
             comparisonInconclusive=0;
    output<<std::setprecision(17);
    while(std::getline(input,line)) {
        if(line.empty() || line[0]=='#')continue;
        std::istringstream row(line);std::string key,kind;Matrix f{},h{};
        row>>key>>kind;need(kind=="liner"||kind=="medium","unknown material kind");
        need(!key.empty()&&std::all_of(key.begin(),key.end(),[](unsigned char c){
            return std::isalnum(c)||c=='_'||c=='-'||c=='.';}),"invalid capture key");
        Matrix fLow{};
        std::vector<float> accepted(13),captured(13),capturedParameters;
        Matrix sourceP{},sourceTangent{};
        std::array<double,4> sourceProjection{};
        const auto read=[&](auto& values){for(auto& v:values){double x;need(bool(row>>x)&&std::isfinite(x),"invalid capture value");v=static_cast<typename std::remove_reference_t<decltype(values)>::value_type>(x);}};
        read(f);
        if(exactFormat)read(fLow);
        read(h);read(accepted);read(captured);
        if(exactFormat){
            capturedParameters.resize(20);read(capturedParameters);
            read(sourceP);read(sourceTangent);read(sourceProjection);
        }
        std::string extra;need(!(row>>extra),"extra capture values");
        const auto& world=worlds[kind=="liner"?0:1];
        if(exactFormat){
            need(world.parameters.size()==capturedParameters.size(),
                 "captured and instrument parameter counts differ");
            for(unsigned i=0;i<capturedParameters.size();++i){
                const float compiled=world.parameters[i].valueAndBounds.x;
                need(std::memcmp(&compiled,&capturedParameters[i],sizeof(float))==0,
                     "captured and instrument FP32 parameter bits differ");
            }
        } else if(accurateProjection){
            capturedParameters.reserve(world.parameters.size());
            for(const auto& value:world.parameters)
                capturedParameters.push_back(value.valueAndBounds.x);
        }
        const Response result=gpu.evaluate(world,f,fLow,h,accepted,accurateProjection);
        if(!result.valid)++failed;
        const auto emit=[&](const auto& values){output<<'[';for(unsigned i=0;i<values.size();++i){if(i)output<<',';output<<values[i];}output<<']';};
        output<<"{\"key\":\""<<key<<"\",\"valid\":"<<(result.valid?"true":"false")
            <<",\"projection_valid\":"<<(result.projectionValid?"true":"false")
            <<",\"first_piola_Pa\":";emit(result.p);
        output<<",\"tangent_Pa_per_s\":";emit(result.dp);
        output<<",\"projected_state\":";emit(result.state);
        if(accurateProjection){
            output<<",\"input_F_local_high\":";emit(f);
            output<<",\"input_F_local_low\":";emit(fLow);
            output<<",\"input_F_direction_local_per_s\":";emit(h);
            output<<",\"projected_state_low\":";emit(result.stateLow);
            output<<",\"captured_projected_state_high\":";emit(captured);
            if(exactFormat){
                output<<",\"captured_first_piola_Pa\":";emit(sourceP);
                output<<",\"captured_tangent_Pa_per_s\":";emit(sourceTangent);
                output<<",\"captured_projection_diagnostics\":";emit(sourceProjection);
                output<<",\"captured_parameter_bits_verified\":true";
            }
        }
        output<<",\"projection_residual\":"<<result.residual
            <<",\"projection_state_norm\":"<<result.stateNorm
            <<",\"projection_iterations\":"<<result.iterations
            <<",\"projection_flags\":"<<result.flags
            <<",\"projection_failure\":"<<result.failure;
        if(accurateProjection){
            const auto materialKind=kind=="liner"?numi_cardboard_paper::MaterialKind::liner:
                                                     numi_cardboard_paper::MaterialKind::medium;
            const auto metrics=compareWithExactInput(f,fLow,h,accepted,result,
                                                      world.constitutive.at(0).material,
                                                      capturedParameters,materialKind);
            const double projectionThreshold=2.0e-8*(1.0+double(result.stateNorm));
            const bool projectionPass=result.projectionValid&&
                result.residual>=0.0f&&double(result.residual)<=projectionThreshold;
            const bool stressResolved=metrics.resolvedFirstPiola>0u;
            const bool stateResolved=metrics.resolvedState>0u;
            const bool stressAccuracyPass=stressResolved&&
                metrics.firstPiolaRelative<=1.0e-3;
            const bool stateAccuracyPass=stateResolved&&metrics.stateRelative<=1.0e-3;
            const bool tangentAccuracyPass=!metrics.zeroDirection&&
                metrics.tangentBranchPreserved&&metrics.resolvedTangent>0u&&
                metrics.tangentFiniteDifferenceNoiseResolved&&
                metrics.tangentRelative<=1.0e-3;
            const bool branchResolved=metrics.oracleBranch!="yield_boundary_unresolved"&&
                metrics.metalBranch!="yield_boundary_unresolved";
            const bool branchMismatch=branchResolved&&!metrics.branchMatches;
            const bool hardFailure=!result.valid||!projectionPass||
                (stressResolved&&!stressAccuracyPass)||
                (stateResolved&&!stateAccuracyPass)||
                (!metrics.zeroDirection&&metrics.tangentBranchPreserved&&
                 metrics.resolvedTangent>0u&&metrics.tangentFiniteDifferenceNoiseResolved&&
                 metrics.tangentRelative>1.0e-3)||branchMismatch;
            const bool hasInconclusive=!stressResolved||!stateResolved||
                metrics.nearZeroState>0u||metrics.nearZeroTangent>0u||
                metrics.zeroDirection||!metrics.tangentBranchPreserved||!branchResolved||
                (!metrics.zeroDirection&&!metrics.tangentFiniteDifferenceNoiseResolved);
            const char* overallStatus=hardFailure?"fail":
                (hasInconclusive?"inconclusive":"pass");
            const char* stressStatus=!stressResolved?"inconclusive_no_resolved_components":
                (stressAccuracyPass?"pass":"fail");
            const char* stateStatus=!stateResolved?"inconclusive_no_resolved_components":
                (!stateAccuracyPass?"fail":
                 (metrics.nearZeroState>0u?"inconclusive_near_zero_components":"pass"));
            const char* tangentStatus=metrics.zeroDirection?"inconclusive_zero_direction":
                (!metrics.tangentBranchPreserved?"inconclusive_branch_changed":
                 (metrics.resolvedTangent==0u?"inconclusive_no_resolved_components":
                  (!metrics.tangentFiniteDifferenceNoiseResolved?"inconclusive_fd_step_noise":
                   (tangentAccuracyPass?"pass":"fail"))));
            double sourcePRelative=0.0,sourcePScale=1.0;
            for(double value:sourceP)sourcePScale=std::max(sourcePScale,std::abs(value));
            const double sourcePFloor=64.0*std::numeric_limits<float>::epsilon()*sourcePScale;
            unsigned sourcePResolved=0;
            const bool sourceProjectionValid=exactFormat&&sourceProjection[3]!=0.0;
            const bool sourcePAvailable=sourceProjectionValid&&result.valid;
            for(unsigned i=0;i<9;++i)if(sourcePAvailable&&std::abs(sourceP[i])>sourcePFloor){
                sourcePRelative=std::max(sourcePRelative,
                    std::abs(result.p[i]-sourceP[i])/std::abs(sourceP[i]));
                ++sourcePResolved;
            }
            double sourceResidualRelative=0.0;
            if(sourceProjectionValid){
                const double residualDenominator=std::max(
                    std::abs(sourceProjection[0]),metrics.twoWordStateResidualFloor);
                sourceResidualRelative=std::abs(
                    double(result.residual)-sourceProjection[0])/
                    std::max(residualDenominator,1.0e-30);
            }
            const bool gate=std::string(overallStatus)=="pass";
            if(!gate){
                ++comparisonNonpass;
                if(hardFailure)++comparisonHardFailed;
                else ++comparisonInconclusive;
            }
            std::array<double,9> stressErrorsPa{};
            std::array<double,9> tangentErrors{};
            std::array<double,9> oracleStressPa{};
            for(unsigned i=0;i<9;++i){
                stressErrorsPa[i]=metrics.firstPiolaAbsoluteError[i];
                tangentErrors[i]=metrics.tangentAbsoluteError[i];
                oracleStressPa[i]=metrics.oracleFirstPiola[i]*1.0e6;
            }
            output<<",\"fp64_oracle_comparison\":{\"first_piola_relative_error\":";
            if(metrics.resolvedFirstPiola>0u&&result.valid)output<<metrics.firstPiolaRelative;
            else output<<"null";
            output<<",\"first_piola_accuracy_status\":\""<<stressStatus
                <<"\",\"projected_state_relative_error\":"
                <<metrics.stateRelative<<",\"resolved_first_piola_components\":"
                <<metrics.resolvedFirstPiola<<",\"near_zero_first_piola_components\":"
                <<metrics.nearZeroFirstPiola<<",\"max_near_zero_first_piola_absolute_error_Pa\":"
                <<metrics.maxNearZeroFirstPiolaAbsoluteError
                <<",\"first_piola_component_resolution_cutoff_Pa\":"
                <<metrics.firstPiolaResolutionCutoff<<",\"fp64_first_piola_Pa\":";
            emit(oracleStressPa);
            output<<",\"first_piola_absolute_error_Pa\":";
            emit(stressErrorsPa);
            output<<",\"projected_state_accuracy_status\":\""<<stateStatus
                <<"\",\"resolved_state_components\":"<<metrics.resolvedState
                <<",\"near_zero_state_components\":"<<metrics.nearZeroState
                <<",\"max_near_zero_state_absolute_error\":"
                <<metrics.maxNearZeroStateAbsoluteError
                <<",\"state_component_resolution_cutoff\":"
                <<metrics.stateResolutionCutoff<<",\"fp64_projected_state\":";
            emit(metrics.oracleState);
            output<<",\"state_absolute_error\":";emit(metrics.stateAbsoluteError);
            output<<",\"branch_metal\":\""<<metrics.metalBranch
                <<"\",\"branch_fp64\":\""<<metrics.oracleBranch
                <<"\",\"branch_matches\":"<<(metrics.branchMatches?"true":"false")
                <<",\"branch_resolved\":"<<(branchResolved?"true":"false")
                <<",\"legacy_independent_residual_at_metal_state_source_norm_not_comparable\":"
                <<metrics.fp64ResidualAtMetalState
                <<",\"fp64_root_high_low_rounding_residual_estimate_not_metal_noise\":"
                <<metrics.twoWordStateResidualFloor
                <<",\"legacy_residual_absolute_difference_between_unlike_norms_not_qualification\":"
                <<metrics.residualAbsoluteDifference
                <<",\"legacy_unlike_norm_ratio_not_for_qualification\":"
                <<metrics.residualRelativeToFloor
                <<",\"legacy_residual_comparison_status\":\"invalid_incomparable_norms_not_a_qualification_metric\""
                <<",\"local_projection_threshold\":"
                <<projectionThreshold<<",\"local_projection_status\":\""
                <<(projectionPass?"pass":"fail")<<"\",\"tangent_relative_error\":";
            if(!metrics.zeroDirection&&metrics.resolvedTangent>0u)output<<metrics.tangentRelative;
            else output<<"null";
            output<<",\"tangent_accuracy_status\":\""<<tangentStatus
                <<"\",\"resolved_tangent_components\":"<<metrics.resolvedTangent
                <<",\"near_zero_tangent_components\":"<<metrics.nearZeroTangent
                <<",\"max_near_zero_tangent_absolute_error_Pa_per_s\":"
                <<metrics.maxNearZeroTangentAbsoluteError
                <<",\"tangent_component_resolution_cutoff_Pa_per_s\":"
                <<metrics.tangentResolutionCutoff
                <<",\"tangent_fd_step_noise_relative\":"
                <<metrics.tangentFiniteDifferenceNoiseRelative
                <<",\"max_tangent_fd_step_noise_absolute_Pa_per_s\":"
                <<metrics.maxTangentFiniteDifferenceNoiseAbsolute
                <<",\"tangent_fd_max_abs_deformation_steps\":[2e-6,1e-6]"
                <<",\"tangent_fd_reference_method\":\"centered_difference_with_second_order_Richardson_extrapolation\""
                <<",\"tangent_fd_noise_resolved_1e_3\":"
                <<(metrics.tangentFiniteDifferenceNoiseResolved?"true":"false")
                <<",\"tangent_absolute_error_Pa_per_s\":";
            emit(tangentErrors);
            output<<",\"tangent_branch_preserved\":"
                <<(metrics.tangentBranchPreserved?"true":"false")
                <<",\"zero_direction_tangent_absolute_error\":"<<metrics.zeroDirectionTangentError
                <<",\"tangent_gate_1e_3\":"<<(metrics.tangentGate?"true":"false")<<'}';
            if(exactFormat){
                output<<",\"native_capture_comparison\":{\"first_piola_available\":"
                    <<(sourcePAvailable?"true":"false")<<",\"first_piola_relative_error\":";
                if(sourcePAvailable&&sourcePResolved>0u)output<<sourcePRelative;
                else output<<"null";
                output<<",\"source_projection_valid\":"<<(sourceProjectionValid?"true":"false")
                    <<",\"source_capture_status\":\""
                    <<(sourceProjectionValid?"secondary_comparison_only":"unavailable_invalid_source_projection")
                    <<"\",\"legacy_incomparable_projection_residual_ratio_not_for_qualification\":";
                if(sourceProjectionValid)output<<sourceResidualRelative;
                else output<<"null";
                output<<'}';
            }
            output<<",\"overall_status\":\""<<overallStatus
                <<"\",\"gate_1e_3\":"<<(gate?"true":"false");
        }
        output<<"}\n";++rows;
        need(output.good(),"captured replay write failed");
    }
    need(rows>0,"empty captured material input");
    std::cout<<"captured_native_replay_rows="<<rows<<" invalid="<<failed
        <<" accurate_comparison_nonpass="<<comparisonNonpass
        <<" accurate_comparison_hard_failures="<<comparisonHardFailed
        <<" accurate_comparison_inconclusive="<<comparisonInconclusive
        <<" scope=production_projection_and_tangent physical_steps=0\n";
    return failed||comparisonNonpass?2:0;
}

std::string sha256File(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    need(input.good(), "cannot read hash-bound input: " + path.string());
    CC_SHA256_CTX context{};
    need(CC_SHA256_Init(&context) == 1, "CommonCrypto SHA256 init failed");
    std::array<char, 65536> bytes{};
    while (input) {
        input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        const auto count = input.gcount();
        if (count > 0) {
            need(CC_SHA256_Update(&context, bytes.data(),
                                  static_cast<CC_LONG>(count)) == 1,
                 "CommonCrypto SHA256 update failed");
        }
    }
    need(input.eof(), "error reading hash-bound input: " + path.string());
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    need(CC_SHA256_Final(digest.data(), &context) == 1,
         "CommonCrypto SHA256 final failed");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2u);
    for (const unsigned char byte : digest) {
        result.push_back(hex[byte >> 4u]);
        result.push_back(hex[byte & 0x0fu]);
    }
    return result;
}

std::string jsonEscape(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 8u);
    for (const unsigned char c : value) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20u) result += '?';
                else result += static_cast<char>(c);
        }
    }
    return result;
}

id parseJsonLine(const std::string& line, const std::string& context) {
    NSData* data = [NSData dataWithBytes:line.data() length:line.size()];
    NSError* error = nil;
    id result = [NSJSONSerialization JSONObjectWithData:data
        options:NSJSONReadingFragmentsAllowed error:&error];
    need(result != nil, context + ": invalid JSON" +
        (error == nil ? std::string{} : std::string(" (") +
         [[error.localizedDescription description] UTF8String] + ")"));
    return result;
}

NSDictionary* jsonDictionary(id value, const std::string& context) {
    need([value isKindOfClass:[NSDictionary class]], context + ": expected object");
    return (NSDictionary*)value;
}

NSArray* jsonArray(id value, const std::string& context) {
    need([value isKindOfClass:[NSArray class]], context + ": expected array");
    return (NSArray*)value;
}

double jsonNumber(id value, const std::string& context) {
    need([value isKindOfClass:[NSNumber class]], context + ": expected number");
    const double result = [(NSNumber*)value doubleValue];
    need(std::isfinite(result), context + ": nonfinite number");
    return result;
}

std::string jsonString(id value, const std::string& context) {
    need([value isKindOfClass:[NSString class]], context + ": expected string");
    return std::string([(NSString*)value UTF8String]);
}

id jsonRequired(NSDictionary* object, NSString* key,
                const std::string& context) {
    id result = [object objectForKey:key];
    need(result != nil, context + ": missing " + [key UTF8String]);
    return result;
}

double jsonFieldNumber(NSDictionary* object, NSString* key,
                       const std::string& context) {
    return jsonNumber(jsonRequired(object, key, context), context + "." +
                      [key UTF8String]);
}

std::string jsonFieldString(NSDictionary* object, NSString* key,
                            const std::string& context) {
    return jsonString(jsonRequired(object, key, context), context + "." +
                      [key UTF8String]);
}

std::vector<double> jsonNumericArray(id value, const std::size_t expected,
                                     const std::string& context) {
    NSArray* array = jsonArray(value, context);
    need([array count] == expected, context + ": wrong array length");
    std::vector<double> result;
    result.reserve(expected);
    for (NSUInteger index = 0u; index < [array count]; ++index)
        result.push_back(jsonNumber([array objectAtIndex:index], context));
    return result;
}

Matrix jsonNativeFloatMatrix3(NSDictionary* object, NSString* key,
                   const std::string& context) {
    NSArray* rows = jsonArray(jsonRequired(object, key, context), context + "." +
                              [key UTF8String]);
    need([rows count] == 3u, context + "." + [key UTF8String] +
         ": expected three rows");
    Matrix result{};
    for (NSUInteger row = 0u; row < 3u; ++row) {
        NSArray* columns = jsonArray([rows objectAtIndex:row], context + ".row");
        need([columns count] == 3u, context + ": expected a 3x3 matrix");
        for (NSUInteger column = 0u; column < 3u; ++column) {
            // Native captures use max_digits10 for FP32. Recover that word
            // before promoting it: the printed decimal is not itself a more
            // precise physical input for the independent FP64 calculation.
            const float nativeWord=static_cast<float>(jsonNumber(
                [columns objectAtIndex:column], context + ".matrix value"));
            need(std::isfinite(nativeWord),context + ": native FP32 matrix overflow");
            result[3u * row + column]=static_cast<double>(nativeWord);
        }
    }
    return result;
}

template<std::size_t N>
void emitJsonArray(std::ostream& output, const std::array<double, N>& values) {
    output << '[';
    for (std::size_t index = 0u; index < N; ++index) {
        if (index != 0u) output << ',';
        output << values[index];
    }
    output << ']';
}

void emitJsonNumber(std::ostream& output, double value) {
    if (std::isfinite(value)) output << value;
    else output << "null";
}

void emitJsonArray(std::ostream& output, const std::vector<float>& values) {
    output << '[';
    for (std::size_t index = 0u; index < values.size(); ++index) {
        if (index != 0u) output << ',';
        output << values[index];
    }
    output << ']';
}

void emitJsonArray(std::ostream& output, const std::vector<double>& values) {
    output << '[';
    for (std::size_t index = 0u; index < values.size(); ++index) {
        if (index != 0u) output << ',';
        output << values[index];
    }
    output << ']';
}

struct ReplayPolicyMetrics {
    unsigned resolvedP = 0u;
    unsigned nearZeroP = 0u;
    unsigned resolvedState = 0u;
    unsigned nearZeroState = 0u;
    unsigned resolvedTangent = 0u;
    unsigned nearZeroTangent = 0u;
    double pRelative = 0.0;
    double stateRelative = 0.0;
    double tangentRelative = 0.0;
};

ReplayPolicyMetrics compareCapturedCenter(const Response& replay,
                                           const Matrix& capturedP,
                                           const Matrix& capturedTangent,
                                           const std::vector<double>& capturedState) {
    ReplayPolicyMetrics result;
    double pScale = 1.0, stateScale = 1.0, tangentScale = 1.0;
    for (double value : capturedP) pScale = std::max(pScale, std::abs(value));
    for (double value : capturedTangent)
        tangentScale = std::max(tangentScale, std::abs(value));
    for (double value : capturedState)
        stateScale = std::max(stateScale, std::abs(value));
    const double pFloor = 64.0 * std::numeric_limits<float>::epsilon() * pScale;
    const double tangentFloor = 64.0 * std::numeric_limits<float>::epsilon() *
        tangentScale;
    const double stateFloor = 64.0 * std::numeric_limits<float>::epsilon() *
        stateScale;
    for (std::size_t index = 0u; index < capturedP.size(); ++index) {
        const double expected = capturedP[index];
        const double error = std::abs(replay.p[index] - expected);
        if (std::abs(expected) > pFloor) {
            result.pRelative = std::max(result.pRelative, error / std::abs(expected));
            ++result.resolvedP;
        } else ++result.nearZeroP;
        const double expectedTangent = capturedTangent[index];
        const double tangentError = std::abs(replay.dp[index] - expectedTangent);
        if (std::abs(expectedTangent) > tangentFloor) {
            result.tangentRelative = std::max(result.tangentRelative,
                tangentError / std::abs(expectedTangent));
            ++result.resolvedTangent;
        } else ++result.nearZeroTangent;
    }
    for (std::size_t index = 0u; index < capturedState.size(); ++index) {
        const double expected = capturedState[index];
        const double replayValue = double(replay.state.at(index)) +
            double(replay.stateLow.at(index));
        const double error = std::abs(replayValue - expected);
        if (std::abs(expected) > stateFloor) {
            result.stateRelative = std::max(result.stateRelative,
                error / std::abs(expected));
            ++result.resolvedState;
        } else ++result.nearZeroState;
    }
    return result;
}

int replayCapturedFemCenters(Instrument& gpu, const char* inputPath,
                             const char* outputPath, const char* linerPath,
                             const char* mediumPath,
                             const std::string& onlySelection) {
    const std::filesystem::path capturePath(inputPath);
    const std::filesystem::path outputFile(outputPath);
    const std::filesystem::path linerFile(linerPath);
    const std::filesystem::path mediumFile(mediumPath);
    need(std::filesystem::is_regular_file(capturePath),
         "captured FEM center input is not a regular file");
    need(!std::filesystem::exists(outputFile),
         "captured FEM center output already exists");
    const std::string captureSHA = sha256File(capturePath);
    const std::string linerSHA = sha256File(linerFile);
    const std::string mediumSHA = sha256File(mediumFile);

    std::array<MaterialProgram, 2> parsedMaterials;
    const std::array<std::filesystem::path, 2> materialPaths{linerFile, mediumFile};
    const std::array<std::string, 2> expectedNames{
        "hajali2009_liner_hill_ideal", "hajali2009_medium_hill_ideal"};
    for (std::size_t index = 0u; index < parsedMaterials.size(); ++index) {
        const auto parsed = parseMatterFile(materialPaths[index].string());
        need(parsed.succeeded(), diagnostics(parsed.diagnostics));
        parsedMaterials[index] = parsed.material;
        need(parsedMaterials[index].name == expectedNames[index],
             "captured center replay requires the exact named 13-state source descriptors");
        need(parsedMaterials[index].internalState.size() == 13u,
             "captured center replay requires the source 13-state ideal-Hill law");
    }

    std::array<CompiledWorld, 2> ordinaryWorlds;
    std::array<CompiledWorld, 2> accurateWorlds;
    std::array<std::array<double,kReplayStateCount>,2> accurateRowScales{};
    std::array<double,2> sourceResidualScales{1.0,1.0};
    for (std::size_t index = 0u; index < parsedMaterials.size(); ++index) {
        MaterialProgram ordinary = parsedMaterials[index];
        need(!ordinary.paperHillAccurateProjection &&
             !ordinary.paperCompensatedArithmetic,
             "ordinary capture parity requires unmodified source policy flags");
        ordinaryWorlds[index] = cook(ordinary);

        MaterialProgram accurate = parsedMaterials[index];
        // These flags are set before compilation so the compiled program,
        // rather than a post-compile switch, owns the second replay policy.
        accurate.paperHillScalarInitializer = false;
        accurate.paperCompensatedArithmetic = true;
        accurate.paperHillAccurateProjection = true;
        const auto rowScales=numi::cardboard::paperProjectionRowScales(accurate);
        for(unsigned row=0u;row<kReplayStateCount;++row)
            accurateRowScales[index][row]=rowScales[row];
        const auto residualScale=std::find_if(accurate.parameters.begin(),
            accurate.parameters.end(),[](const Parameter& value){
                return value.name=="residual_scale";
            });
        if(residualScale!=accurate.parameters.end())
            sourceResidualScales[index]=residualScale->defaultValue;
        accurate = numi::cardboard::scalePaperProjectionResiduals(
            std::move(accurate));
        accurateWorlds[index] = cook(accurate);
        need(ordinaryWorlds[index].materials.at(0u).stateCount == 13u &&
             accurateWorlds[index].materials.at(0u).stateCount == 13u,
             "captured source material compiled with an unexpected state count");
        need(!compiledMaterialUsesOpcode(ordinaryWorlds[index],NM_EXPR_TEMPERATURE) &&
             !compiledMaterialUsesOpcode(accurateWorlds[index],NM_EXPR_TEMPERATURE),
             "capture schema has no material-temperature input; refusing a temperature-dependent material replay");
        need(!compiledMaterialUsesOpcode(ordinaryWorlds[index],NM_EXPR_RATE) &&
             !compiledMaterialUsesOpcode(accurateWorlds[index],NM_EXPR_RATE) &&
             !compiledMaterialUsesOpcode(ordinaryWorlds[index],NM_EXPR_DT) &&
             !compiledMaterialUsesOpcode(accurateWorlds[index],NM_EXPR_DT),
             "independent paper source oracle is rate/time independent; refusing a rate- or time-dependent material replay");
    }

    std::ifstream input(capturePath);
    need(input.good(), "cannot open captured FEM center JSONL");
    std::string line;
    need(bool(std::getline(input, line)), "captured FEM center input is empty");
    NSDictionary* header = jsonDictionary(parseJsonLine(line, "capture header"),
                                          "capture header");
    need(jsonFieldString(header, @"record", "capture header") == "header" &&
         jsonFieldString(header, @"schema", "capture header") ==
             "numi.cardboard.fem-newton-capture.v1",
         "unsupported captured FEM center schema");
    need([(NSNumber*)jsonRequired(header, @"production_inputs_bitwise_unchanged",
                                  "capture header") boolValue],
         "captured production inputs were not certified bitwise unchanged");
    const unsigned environmentCount = static_cast<unsigned>(
        jsonFieldNumber(header, @"environment_count", "capture header"));
    const unsigned tetrahedraPerEnvironment = static_cast<unsigned>(
        jsonFieldNumber(header, @"tetrahedron_count_per_environment", "capture header"));
    need(environmentCount > 0u && tetrahedraPerEnvironment > 0u,
         "captured FEM center dimensions are empty");
    const double captureRoot = jsonFieldNumber(header, @"root", "capture header");
    const double captureNewton = jsonFieldNumber(header, @"newton_iteration",
                                                  "capture header");
    const std::string worldFingerprint = jsonFieldString(
        header, @"world_fingerprint_hex", "capture header");
    const std::string executionFingerprint = jsonFieldString(
        header, @"execution_fingerprint_hex", "capture header");
    const std::vector<double> captureDtValues=jsonNumericArray(
        jsonRequired(header,@"dt_seconds_by_environment","capture header"),
        environmentCount,"capture per-environment timestep");
    std::vector<float> captureTimesteps;captureTimesteps.reserve(environmentCount);
    for(double value:captureDtValues){
        need(value>0.0&&std::isfinite(value),
             "capture has a nonpositive or nonfinite timestep");
        const float converted=static_cast<float>(value);
        need(converted>0.0f&&std::isfinite(converted),
             "capture timestep is outside finite FP32 range");
        captureTimesteps.push_back(converted);
    }

    NSArray* captureMaterials = jsonArray(jsonRequired(header, @"materials",
        "capture header"), "capture materials");
    need([captureMaterials count] >= 2u,
         "capture header has no liner/medium material map");
    std::array<unsigned, 2> materialOffsets{};
    std::array<std::string, 2> captureMaterialNames{};
    std::array<unsigned, 2> captureMaterialCounts{};
    for (NSUInteger i = 0u; i < [captureMaterials count]; ++i) {
        NSDictionary* entry = jsonDictionary([captureMaterials objectAtIndex:i],
                                             "capture material map");
        const unsigned index = static_cast<unsigned>(jsonFieldNumber(
            entry, @"index", "capture material map"));
        if (index >= 2u) continue;
        captureMaterialNames[index] = jsonFieldString(entry, @"name",
                                                       "capture material map");
        materialOffsets[index] = static_cast<unsigned>(jsonFieldNumber(
            entry, @"parameter_offset", "capture material map"));
        NSArray* parameterNames = jsonArray(jsonRequired(entry, @"parameter_names",
            "capture material map"), "capture material parameter names");
        captureMaterialCounts[index] = static_cast<unsigned>([parameterNames count]);
        need(captureMaterialNames[index] == expectedNames[index] &&
             captureMaterialCounts[index] == 20u,
             "capture source material identity/parameter layout differs from the 13-state descriptors");
    }

    // Bind the named source assets to the run manifest when the capture is
    // stored beside the standard solver output directory.  Parameter-bit
    // equality below remains mandatory even when that optional manifest is
    // not present (for future self-contained captures).
    bool manifestVerified = false;
    const std::filesystem::path siblingManifest =
        capturePath.parent_path() / "output" / "manifest.json";
    if (std::filesystem::is_regular_file(siblingManifest)) {
        std::ifstream manifestInput(siblingManifest, std::ios::binary);
        const std::string manifestText((std::istreambuf_iterator<char>(manifestInput)), {});
        NSDictionary* manifest = jsonDictionary(parseJsonLine(manifestText,
            "capture run manifest"), "capture run manifest");
        NSDictionary* materialMap = jsonDictionary(jsonRequired(manifest,
            @"materials", "capture run manifest"), "manifest materials");
        const std::string recordedLinerSHA = jsonFieldString(materialMap,
            @"liner_sha256", "manifest materials");
        const std::string recordedMediumSHA = jsonFieldString(materialMap,
            @"medium_sha256", "manifest materials");
        need(recordedLinerSHA == linerSHA && recordedMediumSHA == mediumSHA,
             "source material asset hashes differ from the sibling run manifest");
        manifestVerified = true;
    }

    NSArray* environmentParameters = jsonArray(jsonRequired(header,
        @"parameters_by_environment", "capture header"),
        "capture environment parameters");
    need([environmentParameters count] == environmentCount,
         "capture environment parameter arity differs from environment count");
    std::array<std::vector<float>, 2> capturedParameterSlices;
    for (unsigned environment = 0u; environment < environmentCount; ++environment) {
        NSArray* values = jsonArray([environmentParameters objectAtIndex:environment],
                                   "capture parameter vector");
        for (unsigned materialIndex = 0u; materialIndex < 2u; ++materialIndex) {
            need([values count] >= materialOffsets[materialIndex] +
                 captureMaterialCounts[materialIndex],
                 "capture environment parameter vector is too short");
            std::vector<float> slice;
            slice.reserve(captureMaterialCounts[materialIndex]);
            for (unsigned parameterIndex = 0u;
                 parameterIndex < captureMaterialCounts[materialIndex];
                 ++parameterIndex) {
                const double value = jsonNumber([values objectAtIndex:
                    materialOffsets[materialIndex] + parameterIndex],
                    "capture material parameter");
                slice.push_back(static_cast<float>(value));
            }
            const auto& compiled = ordinaryWorlds[materialIndex].parameters;
            need(compiled.size() == slice.size(),
                 "source descriptor parameter count differs from capture");
            for (std::size_t parameterIndex = 0u; parameterIndex < slice.size();
                 ++parameterIndex) {
                const float expected = compiled[parameterIndex].valueAndBounds.x;
                need(std::memcmp(&expected, &slice[parameterIndex], sizeof(float)) == 0,
                     "source descriptor parameter bits differ from capture environment");
            }
            if (environment == 0u) capturedParameterSlices[materialIndex] = slice;
            else {
                need(slice.size() == capturedParameterSlices[materialIndex].size(),
                     "capture environment parameter slices differ in size");
                for (std::size_t parameterIndex = 0u;
                     parameterIndex < slice.size(); ++parameterIndex)
                    need(std::memcmp(&slice[parameterIndex],
                        &capturedParameterSlices[materialIndex][parameterIndex],
                        sizeof(float)) == 0,
                        "capture environment material parameters are not bit-identical");
            }
        }
    }

    std::ofstream output(outputFile, std::ios::out | std::ios::trunc);
    need(output.good(), "cannot create captured FEM center replay output");
    output << std::setprecision(17);
    const auto fingerprintHex=[](std::uint64_t value){
        std::ostringstream text;text<<std::hex<<std::setw(16)<<std::setfill('0')<<value;
        return text.str();
    };
    const auto emitCompiledBinding=[&](const char* label,const CompiledWorld& world,
                                       const MaterialProgram& source,
                                       const std::array<double,kReplayStateCount>& rowScales,
                                       bool scaledPolicy){
        const NMMaterialGPU& material=world.materials.at(0u);
        need(material.parameterCount==source.parameters.size(),
             "compiled material parameter count differs from source descriptor");
        output<<"{\"label\":\""<<label<<"\",\"source_material_name\":\""
            <<jsonEscape(source.name)<<"\",\"world_fingerprint_hex\":\""
            <<fingerprintHex(world.fingerprint)<<"\",\"material_flags\":"
            <<material.flags<<",\"state_count\":"<<material.stateCount
            <<",\"parameter_offset\":"<<material.parameterOffset
            <<",\"parameter_count\":"<<material.parameterCount
            <<",\"local_newton_iterations\":"<<material.localNewtonIterations
            <<",\"implicit_residual_program_offset\":"
            <<material.implicitResidualProgramOffset<<",\"source_parameters\":[";
        for(std::size_t i=0u;i<source.parameters.size();++i){
            if(i)output<<',';
            output<<"{\"name\":\""<<jsonEscape(source.parameters[i].name)
                <<"\",\"compiled_f32_bits\":\""
                <<floatBitsHex(world.parameters.at(material.parameterOffset+i).valueAndBounds.x)
                <<"\"}";
        }
        output<<"],\"component_scaling\":{\"enabled\":"
            <<(scaledPolicy?"true":"false")<<",\"source_double\":[";
        for(unsigned row=0u;row<kReplayStateCount;++row){
            if(row)output<<',';output<<(scaledPolicy?rowScales[row]:1.0);
        }
        output<<"],\"compiled_f32_bits\":[";
        for(unsigned row=0u;row<kReplayStateCount;++row){
            if(row)output<<',';
            output<<'\"'<<floatBitsHex(static_cast<float>(scaledPolicy?rowScales[row]:1.0))<<'\"';
        }
        output<<"]},\"residual_programs\":[";
        for(unsigned row=0u;row<kReplayStateCount;++row){
            if(row)output<<',';
            const std::uint32_t programIndex=material.implicitResidualProgramOffset+row;
            const NMScalarProgramGPU& program=world.scalarPrograms.at(programIndex);
            output<<"{\"row\":"<<row<<",\"program_index\":"<<programIndex
                <<",\"flags\":"<<program.flags
                <<",\"deformation_and_candidate_low_words_used\":"
                <<((program.flags&2u)!=0u?"true":"false")<<",\"first_instruction\":"
                <<program.firstInstruction<<",\"instruction_count\":"
                <<program.instructionCount<<",\"maximum_stack\":"
                <<program.maximumStack<<",\"instructions\":[";
            for(unsigned local=0u;local<program.instructionCount;++local){
                if(local)output<<',';
                const auto& instruction=world.instructions.at(program.firstInstruction+local);
                output<<'['<<instruction.opcode<<','<<instruction.index<<','
                    <<instruction.integer<<','<<instruction.reserved<<",\""
                    <<floatBitsHex(instruction.immediate.x)<<"\",\""
                    <<floatBitsHex(instruction.immediate.y)<<"\",\""
                    <<floatBitsHex(instruction.immediate.z)<<"\",\""
                    <<floatBitsHex(instruction.immediate.w)<<"\"]";
            }
            output<<"]}";
        }
        output<<"]}";
    };
    output << "{\"record\":\"header\",\"schema\":\"numi.cardboard.paper-captured-center-replay.v2\""
        << ",\"physical_steps\":0,\"physical_validation\":false"
        << ",\"capture_path\":\"" << jsonEscape(capturePath.string()) << "\""
        << ",\"capture_sha256\":\"" << captureSHA << "\""
        << ",\"liner_source_path\":\"" << jsonEscape(linerFile.string()) << "\""
        << ",\"liner_source_sha256\":\"" << linerSHA << "\""
        << ",\"medium_source_path\":\"" << jsonEscape(mediumFile.string()) << "\""
        << ",\"medium_source_sha256\":\"" << mediumSHA << "\""
        << ",\"sibling_manifest_verified\":"
        << (manifestVerified ? "true" : "false")
        << ",\"capture_root\":" << captureRoot
        << ",\"capture_newton_iteration\":" << captureNewton
        << ",\"world_fingerprint_hex\":\"" << worldFingerprint << "\""
        << ",\"execution_fingerprint_hex\":\"" << executionFingerprint << "\""
        << ",\"dt_seconds_by_environment_f32\":[";
    for(unsigned environment=0u;environment<environmentCount;++environment){
        if(environment)output<<',';
        output<<"{\"value\":"<<captureTimesteps[environment]
            <<",\"bits\":\""<<floatBitsHex(captureTimesteps[environment])<<"\"}";
    }
    output<<"]"
        << ",\"environment_count\":" << environmentCount
        << ",\"tetrahedron_count_per_environment\":"
        << tetrahedraPerEnvironment
        << ",\"default_scope\":\"all source-paper liner and medium center records; glue centers are counted and emitted as skipped\""
        << ",\"ordinary_policy\":{\"accurate_projection\":false,\"scaled_residuals\":false,\"compensated_arithmetic\":false,\"scalar_initializer\":false}"
        << ",\"fp64_comparison_policy\":{\"accurate_projection\":true,\"scaled_residuals\":true,\"compensated_arithmetic\":true,\"scalar_initializer\":false,\"global_tolerance_changed\":false,\"history_or_geometry_changed\":false,\"reference_solver\":\"independent analytic implicit Jacobian; max64 iterations; Armijo c1=1e-4 with <=; max24 halvings; tolerance1e-12 times max(1,1+abs(state))\"}"
        << ",\"nonzero_component_gate\":1e-3,\"near_zero_policy\":\"inconclusive without measured noise floor\""
        << ",\"native_capture_numeric_policy\":\"recover each captured FP32 matrix word from its max_digits10 decimal before promotion to FP64; native and independent references use identical F high/low, rate, direction, stress and tangent words\""
        << ",\"compiled_bytecode_fp64_policy\":\"exact flattened production instruction sequence; FP32 immediates and parameter bits promoted to FP64; per-program flag bit2 selects deformation and candidate low words as in the native VM; binary64 arithmetic oracle, not a bitwise Metal branch/domain emulator; comparisons near domain or branch transitions remain unresolved without separate one-sided checks\""
        << ",\"analytical_residual_input_policy\":\"source_residual_before_authored_scale uses exact high-plus-low words; source_residual_with_authored_scale and accurate_component_scaled_expected each use their corresponding residual program low-word flag before applying the reported scale\""
        << ",\"native_residual_instrument\":\"production evaluateImplicitMaterialSystem residual programs at each supplied state snapshot; no projection or history mutation\""
        << ",\"temperature_input\":\"not referenced by compiled material programs; fixed checker argument is irrelevant\""
        << ",\"legacy_residual_ratio_status\":\"invalid_incomparable_norms_not_a_qualification_metric\""
        << ",\"compiled_worlds\":[";
    emitCompiledBinding("liner_ordinary_replay",ordinaryWorlds[0],parsedMaterials[0],accurateRowScales[0],false);
    output<<',';
    emitCompiledBinding("liner_accurate_candidate",accurateWorlds[0],parsedMaterials[0],accurateRowScales[0],true);
    output<<',';
    emitCompiledBinding("medium_ordinary_replay",ordinaryWorlds[1],parsedMaterials[1],accurateRowScales[1],false);
    output<<',';
    emitCompiledBinding("medium_accurate_candidate",accurateWorlds[1],parsedMaterials[1],accurateRowScales[1],true);
    output<<"]}\n";

    const auto emitCaptureMatrix = [&](const Matrix& value) {
        emitJsonArray(output, value);
    };
    std::vector<std::vector<bool>> seen(environmentCount,
        std::vector<bool>(tetrahedraPerEnvironment, false));
    unsigned recordCount = 0u, paperCenters = 0u, replayed = 0u;
    unsigned skippedGlue = 0u, failedOrdinary = 0u, failedAccurate = 0u;
    unsigned ordinaryNonpass = 0u, accurateNonpass = 0u;
    unsigned exactCenterProjectionFailures = 0u, exactTangentProjectionFailures = 0u;
    bool integrityRecord = false;
    bool foundOnlySelection = onlySelection.empty();
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        NSDictionary* record = jsonDictionary(parseJsonLine(line, "capture record"),
                                               "capture record");
        const std::string recordType = jsonFieldString(record, @"record",
                                                        "capture record");
        if (recordType == "integrity") {
            integrityRecord = [(NSNumber*)jsonRequired(record,
                @"production_inputs_bitwise_unchanged", "capture integrity") boolValue];
            continue;
        }
        if (recordType != "tetrahedron") continue;
        ++recordCount;
        const unsigned environment = static_cast<unsigned>(jsonFieldNumber(
            record, @"environment", "tetrahedron record"));
        const unsigned tetId = static_cast<unsigned>(jsonFieldNumber(
            record, @"tet_id", "tetrahedron record"));
        need(environment < environmentCount && tetId < tetrahedraPerEnvironment,
             "capture tetrahedron identity is outside declared dimensions");
        need(!seen[environment][tetId], "duplicate capture environment/tetrahedron record");
        seen[environment][tetId] = true;
        const unsigned materialIndex = static_cast<unsigned>(jsonFieldNumber(
            record, @"material_index", "tetrahedron record"));
        const std::string materialName = jsonFieldString(record, @"material_name",
                                                          "tetrahedron record");
        const unsigned stateCount = static_cast<unsigned>(jsonFieldNumber(
            record, @"material_state_count", "tetrahedron record"));
        const unsigned parameterCount = static_cast<unsigned>(jsonFieldNumber(
            record, @"material_parameter_count", "tetrahedron record"));
        const std::string selectedKey = std::to_string(environment) + ":" +
            std::to_string(tetId);
        if (!onlySelection.empty() && selectedKey != onlySelection) continue;
        foundOnlySelection = true;
        if (materialIndex >= 2u) {
            need(materialName == "starch2007_finite_glue_elastic" && stateCount == 0u,
                 "unsupported non-paper material in center replay capture");
            ++skippedGlue;
            output << "{\"record\":\"skipped_center\",\"environment\":"
                   << environment << ",\"tet_id\":" << tetId
                   << ",\"material_name\":\"" << jsonEscape(materialName)
                   << "\",\"reason\":\"non-paper elastic glue has no 13-state Hill source oracle\"}\n";
            continue;
        }
        need(materialName == expectedNames[materialIndex] && stateCount == 13u &&
             parameterCount == 20u,
             "captured center does not match the exact source 13-state material layout");
        ++paperCenters;
        NSDictionary* center = jsonDictionary(jsonRequired(record, @"center",
            "tetrahedron record"), "tetrahedron center");
        const Matrix f = jsonNativeFloatMatrix3(center, @"F_local", "tetrahedron center");
        const Matrix fLow = jsonNativeFloatMatrix3(center, @"F_local_low", "tetrahedron center");
        const Matrix h = jsonNativeFloatMatrix3(center, @"F_direction_local_per_s",
                                     "tetrahedron center");
        const Matrix deformationRate=jsonNativeFloatMatrix3(center,@"Fdot_local_per_s",
                                                  "tetrahedron center");
        const float captureTimestep=captureTimesteps.at(environment);
        const std::vector<double> centerContext=jsonNumericArray(
            jsonRequired(center,@"dt_seconds_bulk_modulus_kind_frame_active",
                         "tetrahedron center"),4u,"captured center context");
        need(static_cast<float>(centerContext[0])==captureTimestep,
             "center timestep differs from the environment header timestep");
        const Matrix captureP = jsonNativeFloatMatrix3(center, @"P_local_Pa",
                                            "tetrahedron center");
        const Matrix captureTangent = jsonNativeFloatMatrix3(center,
            @"tangent_local_Pa_per_s", "tetrahedron center");
        const std::vector<double> acceptedValues = jsonNumericArray(
            jsonRequired(center, @"accepted_state", "tetrahedron center"),
            13u, "captured accepted state");
        const std::vector<double> capturedStateValues = jsonNumericArray(
            jsonRequired(center, @"projected_state", "tetrahedron center"),
            13u, "captured projected state");
        const std::vector<double> capturedStateLowValues = jsonNumericArray(
            jsonRequired(center, @"projected_state_low", "tetrahedron center"),
            13u, "captured projected state low words");
        std::vector<float> accepted(13u);
        std::vector<double> capturedState(13u);
        StateWords capturedStateWords;capturedStateWords.available=true;
        for (std::size_t state = 0u; state < 13u; ++state) {
            accepted[state] = static_cast<float>(acceptedValues[state]);
            capturedStateWords.high[state]=static_cast<float>(capturedStateValues[state]);
            capturedStateWords.low[state]=static_cast<float>(capturedStateLowValues[state]);
            capturedState[state] = double(capturedStateWords.high[state])+
                                   double(capturedStateWords.low[state]);
        }
        const std::vector<double> captureProjection = jsonNumericArray(
            jsonRequired(center, @"projection", "tetrahedron center"),
            4u, "captured projection diagnostics");
        const std::vector<double> flags = jsonNumericArray(
            jsonRequired(center, @"flags_terminal_status_parameter_count",
                         "tetrahedron center"), 4u,
            "captured projection flags/status");
        const std::vector<double> qLambda = jsonNumericArray(
            jsonRequired(center, @"hill_q_lambda_a_b", "tetrahedron center"),
            4u, "captured Hill diagnostics");

        const auto& ordinaryWorld = ordinaryWorlds[materialIndex];
        const auto& accurateWorld = accurateWorlds[materialIndex];
        const std::vector<float>& captureParameters =
            capturedParameterSlices[materialIndex];
        const Response ordinary = gpu.evaluate(ordinaryWorld, f, fLow, h,
                                                 accepted, false,captureTimestep,
                                                 deformationRate);
        const Response accurate = gpu.evaluate(accurateWorld, f, fLow, h,
                                                 accepted, true,captureTimestep,
                                                 deformationRate);
        if (!ordinary.valid) ++failedOrdinary;
        if (!accurate.valid) ++failedAccurate;
        ++replayed;

        const ReplayPolicyMetrics parity = compareCapturedCenter(
            ordinary, captureP, captureTangent, capturedState);
        const bool parityHasResolved = parity.resolvedP > 0u &&
            parity.resolvedState > 0u && parity.resolvedTangent > 0u;
        const bool parityPass = ordinary.valid && parityHasResolved &&
            parity.pRelative <= 1.0e-3 && parity.stateRelative <= 1.0e-3 &&
            parity.tangentRelative <= 1.0e-3;
        const auto kind = materialIndex == 0u
            ? numi_cardboard_paper::MaterialKind::liner
            : numi_cardboard_paper::MaterialKind::medium;
        const ExactMetrics reference = compareWithExactInput(f, fLow, h,
            accepted, accurate, accurateWorld.constitutive.at(0u).material,
            captureParameters, kind);
        const auto responseWords=[](const Response& response){
            StateWords words;words.available=response.state.size()>=kReplayStateCount;
            if(words.available)for(unsigned i=0u;i<kReplayStateCount;++i){
                words.high[i]=response.state[i];
                if(i<response.stateLow.size())words.low[i]=response.stateLow[i];
            }
            return words;
        };
        std::array<StateWords,kReplayCandidateCount> candidateStates{};
        candidateStates[0]=capturedStateWords;
        candidateStates[1]=responseWords(ordinary);
        candidateStates[2]=responseWords(accurate);
        double rootTwoWordSplitMaxError=0.0;
        if(reference.oracleProjectionValid){
            candidateStates[3].available=true;
            for(unsigned i=0u;i<kReplayStateCount;++i){
                const float high=static_cast<float>(reference.oracleState[i]);
                const float low=static_cast<float>(reference.oracleState[i]-double(high));
                candidateStates[3].high[i]=high;
                candidateStates[3].low[i]=low;
                rootTwoWordSplitMaxError=std::max(rootTwoWordSplitMaxError,
                    std::abs(double(high)+double(low)-reference.oracleState[i]));
            }
        }
        std::array<float,NM_MAX_MATERIAL_STATE> acceptedWords{};
        for(unsigned i=0u;i<kReplayStateCount;++i)acceptedWords[i]=accepted[i];
        const ResidualRowCandidates ordinaryNativeRows=
            gpu.evaluateCompiledResidualRows(ordinaryWorld,f,fLow,acceptedWords,candidateStates,
                                             deformationRate,captureTimestep);
        const ResidualRowCandidates accurateNativeRows=
            gpu.evaluateCompiledResidualRows(accurateWorld,f,fLow,acceptedWords,candidateStates,
                                             deformationRate,captureTimestep);
        std::array<CompiledResidualFP64,kReplayCandidateCount> ordinaryBytecodeRows{};
        std::array<CompiledResidualFP64,kReplayCandidateCount> accurateBytecodeRows{};
        std::array<std::array<double,NM_MAX_MATERIAL_STATE>,kReplayCandidateCount>
            analyticSourceRows{};
        std::array<std::array<double,NM_MAX_MATERIAL_STATE>,kReplayCandidateCount>
            analyticHighOnlySourceRows{};
        for(unsigned candidate=0u;candidate<kReplayCandidateCount;++candidate){
            ordinaryBytecodeRows[candidate]=evaluateCompiledResidualsFP64(
                ordinaryWorld,acceptedWords,candidateStates[candidate],f,fLow,
                deformationRate,captureTimestep);
            accurateBytecodeRows[candidate]=evaluateCompiledResidualsFP64(
                accurateWorld,acceptedWords,candidateStates[candidate],f,fLow,
                deformationRate,captureTimestep);
            if(candidateStates[candidate].available){
                analyticSourceRows[candidate]=sourceResidualRowsFP64(
                    captureParameters,kind,f,fLow,acceptedWords,candidateStates[candidate],true);
                analyticHighOnlySourceRows[candidate]=sourceResidualRowsFP64(
                    captureParameters,kind,f,fLow,acceptedWords,candidateStates[candidate],false);
            }
        }
        if(!reference.oracleProjectionValid)++exactCenterProjectionFailures;
        if(reference.tangentReferenceProjectionFailed)++exactTangentProjectionFailures;
        const double projectionThreshold = 2.0e-8 *
            (1.0 + double(accurate.stateNorm));
        const bool accurateProjectionPass = accurate.valid &&
            accurate.projectionValid && accurate.residual >= 0.0f &&
            double(accurate.residual) <= projectionThreshold;
        const bool pPass = reference.oracleProjectionValid &&
            reference.resolvedFirstPiola > 0u &&
            reference.firstPiolaRelative <= 1.0e-3;
        const bool statePass = reference.oracleProjectionValid &&
            reference.resolvedState > 0u &&
            reference.stateRelative <= 1.0e-3;
        const bool tangentPass = reference.oracleProjectionValid &&
            reference.tangentReferenceProjectionValid && !reference.zeroDirection &&
            reference.tangentBranchPreserved && reference.resolvedTangent > 0u &&
            reference.tangentFiniteDifferenceNoiseResolved &&
            reference.tangentRelative <= 1.0e-3;
        const bool referenceHasResolved = reference.oracleProjectionValid &&
            reference.resolvedFirstPiola > 0u &&
            reference.resolvedState > 0u && reference.resolvedTangent > 0u;
        const bool accuratePass = reference.oracleProjectionValid &&
            accurateProjectionPass && pPass && statePass &&
            tangentPass && reference.branchMatches;
        if (!parityPass) ++ordinaryNonpass;
        if (!accuratePass) ++accurateNonpass;
        const char* exactReplayStatus = !reference.oracleProjectionValid
            ? "reference_projection_failed"
            : (reference.tangentReferenceProjectionFailed
                ? "inconclusive_tangent_reference_projection_failed"
                : (accuratePass
                    ? "pass_resolved_components_with_nearzero_inconclusive"
                    : (referenceHasResolved ? "fail" :
                       "inconclusive_no_resolved_components")));

        output << "{\"record\":\"captured_center_replay\",\"environment\":"
            << environment << ",\"tet_id\":" << tetId
            << ",\"material_index\":" << materialIndex
            << ",\"material_name\":\"" << jsonEscape(materialName) << "\""
            << ",\"capture_sha256\":\"" << captureSHA << "\""
            << ",\"material_source_sha256\":\""
            << (materialIndex == 0u ? linerSHA : mediumSHA) << "\""
            << ",\"world_fingerprint_hex\":\"" << worldFingerprint << "\""
            << ",\"execution_fingerprint_hex\":\"" << executionFingerprint << "\""
            << ",\"F_local_high\":"; emitCaptureMatrix(f);
        output << ",\"F_local_low\":"; emitCaptureMatrix(fLow);
        output << ",\"F_direction_local_per_s\":"; emitCaptureMatrix(h);
        output << ",\"Fdot_local_per_s\":"; emitCaptureMatrix(deformationRate);
        output << ",\"captured_timestep_seconds\":" << captureTimestep
            << ",\"captured_timestep_f32_bits\":\""
            << floatBitsHex(captureTimestep) << "\"";
        output << ",\"accepted_state\":"; emitJsonArray(output, accepted);
        output << ",\"captured_projected_state\":"; emitJsonArray(output, capturedState);
        output << ",\"captured_P_local_Pa\":"; emitCaptureMatrix(captureP);
        output << ",\"captured_tangent_local_Pa_per_s\":";
        emitCaptureMatrix(captureTangent);
        output << ",\"captured_projection_diagnostics\":";
        emitJsonArray(output, captureProjection);
        output << ",\"captured_flags_terminal_status_parameter_count\":";
        emitJsonArray(output, flags);
        output << ",\"captured_hill_q_lambda_a_b\":";
        emitJsonArray(output, qLambda);
        output<<",\"candidate_state_snapshots\":{";
        const auto emitCandidateState=[&](unsigned candidate){
            const auto& state=candidateStates[candidate];
            output<<"\""<<kReplayCandidateNames[candidate]<<"\":{";
            output<<"\"available\":"<<(state.available?"true":"false")
                <<",\"high\":";
            if(state.available){
                std::vector<float> high(state.high.begin(),state.high.begin()+kReplayStateCount);
                std::vector<float> low(state.low.begin(),state.low.begin()+kReplayStateCount);
                emitJsonArray(output,high);output<<",\"low\":";emitJsonArray(output,low);
                std::vector<double> total(kReplayStateCount);
                for(unsigned i=0u;i<kReplayStateCount;++i)
                    total[i]=double(state.high[i])+double(state.low[i]);
                output<<",\"high_plus_low\":";emitJsonArray(output,total);
            }else output<<"null,\"low\":null,\"high_plus_low\":null";
            output<<"}";
        };
        for(unsigned candidate=0u;candidate<kReplayCandidateCount;++candidate){
            if(candidate)output<<',';emitCandidateState(candidate);
        }
        output<<"},\"candidate_responses\":{";
        output<<"\"captured_production\":{\"valid\":true,\"P_local_Pa\":";
        emitCaptureMatrix(captureP);output<<",\"tangent_local_Pa_per_s\":";
        emitCaptureMatrix(captureTangent);output<<"},\"ordinary_replay\":{\"valid\":"
            <<(ordinary.valid?"true":"false")<<",\"P_local_Pa\":";
        emitCaptureMatrix(ordinary.p);output<<",\"tangent_local_Pa_per_s\":";
        emitCaptureMatrix(ordinary.dp);output<<"},\"accurate_candidate\":{\"valid\":"
            <<(accurate.valid?"true":"false")<<",\"P_local_Pa\":";
        emitCaptureMatrix(accurate.p);output<<",\"tangent_local_Pa_per_s\":";
        emitCaptureMatrix(accurate.dp);output<<"},\"fp64_root\":{\"valid\":"
            <<(reference.oracleProjectionValid?"true":"false")<<",\"P_local_Pa\":";
        if(reference.oracleProjectionValid){
            Matrix rootP{};for(unsigned i=0u;i<9u;++i)rootP[i]=reference.oracleFirstPiola[i]*1.0e6;
            emitCaptureMatrix(rootP);
        }else output<<"null";
        output<<",\"tangent_local_Pa_per_s\":";
        if(reference.tangentReferenceProjectionValid){
            Matrix rootT{};for(unsigned i=0u;i<9u;++i)rootT[i]=reference.oracleTangent[i];
            emitCaptureMatrix(rootT);
        }else output<<"null";
        output<<",\"two_word_state_split_max_abs_error\":"
            <<rootTwoWordSplitMaxError<<"}},\"compiled_residual_rows\":{";
        const auto emitRows13=[&](const auto& values){
            std::vector<double> row(values.begin(),values.begin()+kReplayStateCount);
            emitJsonArray(output,row);
        };
        const auto emitNativePolicy=[&](const char* policy,const ResidualRowCandidates& values){
            output<<"\""<<policy<<"\":[";
            for(unsigned candidate=0u;candidate<kReplayCandidateCount;++candidate){
                if(candidate)output<<',';
                output<<"{\"candidate\":\""<<kReplayCandidateNames[candidate]
                    <<"\",\"snapshot_available\":"
                    <<(candidateStates[candidate].available?"true":"false")
                    <<",\"valid\":"<<(values[candidate].valid?"true":"false")
                    <<",\"nonfinite\":"<<(values[candidate].nonfinite?"true":"false")
                    <<",\"residual_high\":";
                if(values[candidate].valid){
                    std::vector<float> high(values[candidate].high.begin(),
                        values[candidate].high.begin()+kReplayStateCount);
                    std::vector<float> low(values[candidate].low.begin(),
                        values[candidate].low.begin()+kReplayStateCount);
                    emitJsonArray(output,high);output<<",\"residual_low\":";
                    emitJsonArray(output,low);
                    std::vector<double> total(kReplayStateCount);
                    for(unsigned row=0u;row<kReplayStateCount;++row)
                        total[row]=double(values[candidate].high[row])+double(values[candidate].low[row]);
                    output<<",\"residual_high_plus_low\":";emitJsonArray(output,total);
                }else output<<"null,\"residual_low\":null,\"residual_high_plus_low\":null";
                output<<'}';
            }
            output<<']';
        };
        output<<"\"native_ordinary_policy\":{";
        emitNativePolicy("ordinary_replay",ordinaryNativeRows);
        output<<"},\"native_accurate_policy\":{";
        emitNativePolicy("accurate_candidate",accurateNativeRows);
        output<<'}';
        const auto emitBytecodePolicy=[&](const char* policy,
                const std::array<CompiledResidualFP64,kReplayCandidateCount>& values){
            output<<"\""<<policy<<"\":[";
            for(unsigned candidate=0u;candidate<kReplayCandidateCount;++candidate){
                if(candidate)output<<',';
                output<<"{\"candidate\":\""<<kReplayCandidateNames[candidate]
                    <<"\",\"valid\":"<<(values[candidate].valid?"true":"false")
                    <<",\"failure\":\""<<jsonEscape(values[candidate].failure)
                    <<"\",\"residual_fp64\":";
                if(values[candidate].valid)emitRows13(values[candidate].rows);
                else output<<"null";
                output<<'}';
            }
            output<<']';
        };
        output<<",\"compiled_bytecode_fp64\":{";
        emitBytecodePolicy("ordinary_policy",ordinaryBytecodeRows);
        output<<',';emitBytecodePolicy("accurate_policy",accurateBytecodeRows);
        output<<"},\"analytical_source_reference_fp64\":[";
        for(unsigned candidate=0u;candidate<kReplayCandidateCount;++candidate){
            if(candidate)output<<',';
            output<<"{\"candidate\":\""<<kReplayCandidateNames[candidate]
                <<"\",\"valid\":"<<(candidateStates[candidate].available?"true":"false")
                <<",\"source_residual_before_authored_scale\":";
            if(candidateStates[candidate].available){
                emitRows13(analyticSourceRows[candidate]);
                output<<",\"source_residual_with_authored_scale\":";
                std::array<double,NM_MAX_MATERIAL_STATE> ordinaryExpected{};
                std::array<double,NM_MAX_MATERIAL_STATE> accurateExpected{};
                for(unsigned row=0u;row<kReplayStateCount;++row){
                    const auto& ordinaryProgram=ordinaryWorld.scalarPrograms.at(
                        ordinaryWorld.materials.at(0u).implicitResidualProgramOffset+row);
                    const auto& accurateProgram=accurateWorld.scalarPrograms.at(
                        accurateWorld.materials.at(0u).implicitResidualProgramOffset+row);
                    ordinaryExpected[row]=((ordinaryProgram.flags&2u)!=0u?
                        analyticSourceRows[candidate][row]:analyticHighOnlySourceRows[candidate][row])*
                        sourceResidualScales[materialIndex];
                    accurateExpected[row]=((accurateProgram.flags&2u)!=0u?
                        analyticSourceRows[candidate][row]:analyticHighOnlySourceRows[candidate][row])*
                        sourceResidualScales[materialIndex]*
                        static_cast<double>(static_cast<float>(accurateRowScales[materialIndex][row]));
                }
                emitRows13(ordinaryExpected);output<<",\"accurate_component_scaled_expected\":";
                emitRows13(accurateExpected);
            }else output<<"null,\"source_residual_with_authored_scale\":null,\"accurate_component_scaled_expected\":null";
            output<<'}';
        }
        output<<"],\"source_residual_scale\":"<<sourceResidualScales[materialIndex]
            <<",\"ordinary_material_flags\":"<<ordinaryWorld.materials.at(0u).flags
            <<",\"accurate_material_flags\":"<<accurateWorld.materials.at(0u).flags
            <<",\"ordinary_compiled_world_fingerprint_hex\":\""
            <<fingerprintHex(ordinaryWorld.fingerprint)
            <<"\",\"accurate_compiled_world_fingerprint_hex\":\""
            <<fingerprintHex(accurateWorld.fingerprint)<<"\"}";
        output << ",\"ordinary_replay\":{\"policy\":\"compiled source material defaults, no accurate/scaled/compensated policy\""
            << ",\"valid\":" << (ordinary.valid ? "true" : "false")
            << ",\"projection_valid\":" << (ordinary.projectionValid ? "true" : "false")
            << ",\"iterations\":" << ordinary.iterations
            << ",\"residual\":" << ordinary.residual
            << ",\"failure\":" << ordinary.failure
            << ",\"P_local_Pa\":"; emitJsonArray(output, ordinary.p);
        output << ",\"state\":"; emitJsonArray(output, ordinary.state);
        output << ",\"state_low\":"; emitJsonArray(output, ordinary.stateLow);
        output << ",\"tangent_local_Pa_per_s\":"; emitJsonArray(output, ordinary.dp);
        output << ",\"capture_parity\":{\"gate\":1e-3,\"status\":\""
            << (parityPass ? "pass_resolved_components_with_nearzero_inconclusive" :
                parityHasResolved ? "fail" : "inconclusive_no_resolved_components")
            << "\",\"first_piola_relative_error\":" << parity.pRelative
            << ",\"projected_state_relative_error\":" << parity.stateRelative
            << ",\"tangent_relative_error\":" << parity.tangentRelative
            << ",\"resolved_components\":[" << parity.resolvedP << ','
            << parity.resolvedState << ',' << parity.resolvedTangent
            << "],\"nearzero_components_inconclusive\":[" << parity.nearZeroP << ','
            << parity.nearZeroState << ',' << parity.nearZeroTangent << "]}}"
            << ",\"accurate_fp64_replay\":{\"policy\":\"source13 accurate projection + component-scaled residuals + compensated arithmetic; flags applied before compile\""
            << ",\"valid\":" << (accurate.valid ? "true" : "false")
            << ",\"projection_valid\":" << (accurate.projectionValid ? "true" : "false")
            << ",\"iterations\":" << accurate.iterations
            << ",\"residual\":" << accurate.residual
            << ",\"failure\":" << accurate.failure
            << ",\"local_projection_threshold\":" << projectionThreshold
            << ",\"local_projection_status\":\""
            << (accurateProjectionPass ? "pass" : "fail") << "\""
            << ",\"oracle_projection\":{\"method\":\"independent_analytic_implicit_jacobian\",\"valid\":"
            << (reference.oracleProjectionValid ? "true" : "false")
            << ",\"failure\":\"" << jsonEscape(reference.oracleProjectionFailure) << "\""
            << ",\"iteration\":" << reference.oracleProjectionIteration
            << ",\"residual_norm\":";
        emitJsonNumber(output, reference.oracleProjectionResidual);
        output << ",\"state_scale\":";
        emitJsonNumber(output, reference.oracleProjectionStateNorm);
        output
            << ",\"worst_row\":" << reference.oracleProjectionWorstRow
            << ",\"worst_residual\":";
        emitJsonNumber(output, reference.oracleProjectionWorstResidual);
        output
            << ",\"line_search_trials\":" << reference.oracleProjectionLineSearchTrials
            << ",\"last_alpha\":";
        emitJsonNumber(output, reference.oracleProjectionLastAlpha);
        output << ",\"last_trial_residual\":";
        emitJsonNumber(output, reference.oracleProjectionLastTrialResidual);
        output << ",\"best_trial_residual\":";
        emitJsonNumber(output, reference.oracleProjectionBestTrialResidual);
        output << "}"
            << ",\"first_piola_relative_error\":";
        if (reference.oracleProjectionValid) output << reference.firstPiolaRelative;
        else output << "null";
        output << ",\"projected_state_relative_error\":";
        if (reference.oracleProjectionValid) output << reference.stateRelative;
        else output << "null";
        output << ",\"tangent_relative_error\":";
        if (reference.tangentReferenceProjectionValid) output << reference.tangentRelative;
        else output << "null";
        output
            << ",\"branch_matches\":" << (reference.branchMatches ? "true" : "false")
            << ",\"tangent_fd_noise_relative\":"
            << reference.tangentFiniteDifferenceNoiseRelative
            << ",\"resolved_components\":[" << reference.resolvedFirstPiola << ','
            << reference.resolvedState << ',' << reference.resolvedTangent
            << "],\"nearzero_components_inconclusive\":["
            << reference.nearZeroFirstPiola << ',' << reference.nearZeroState << ','
            << reference.nearZeroTangent << "]"
            << ",\"oracle_P_local_Pa\":";
        if (reference.oracleProjectionValid) {
            Matrix stressPa=reference.oracleFirstPiola;
            for(double& value:stressPa)value*=1.0e6;
            emitJsonArray(output, stressPa);
        }
        else output << "null";
        output << ",\"independent_implicit_tangent\":{\"valid\":"
            << (reference.oracleImplicitTangentValid ? "true" : "false")
            << ",\"failure\":\"" << jsonEscape(reference.oracleImplicitTangentFailure)
            << "\",\"method\":\"analytic differentiation of complete source13 implicit equations at the converged FP64 candidate\""
            << ",\"comparison_scope\":\"descriptive full-vector infinity norm; componentwise and measured absolute noise-floor gates remain separate\""
            << ",\"zero_direction\":" << (reference.zeroDirection ? "true" : "false")
            << ",\"oracle_Pa_per_s\":";
        if(reference.oracleImplicitTangentValid)
            emitJsonArray(output, reference.oracleImplicitTangentPaPerS);
        else output << "null";
        output << ",\"oracle_norm_Pa_per_s\":";
        emitJsonNumber(output, reference.implicitTangentNormPaPerS);
        output << ",\"maximum_absolute_error_Pa_per_s\":";
        emitJsonNumber(output, reference.implicitTangentMaxAbsoluteErrorPaPerS);
        output << ",\"norm_relative_error\":";
        if(reference.oracleImplicitTangentValid && !reference.zeroDirection)
            emitJsonNumber(output, reference.implicitTangentNormRelativeError);
        else output << "null";
        output << "}";
        output << ",\"oracle_tangent_local_Pa_per_s\":";
        if (reference.tangentReferenceProjectionValid) emitJsonArray(output, reference.oracleTangent);
        else output << "null";
        output << ",\"oracle_projected_state\":";
        if (reference.oracleProjectionValid) emitJsonArray(output, reference.oracleState);
        else output << "null";
        output << ",\"tangent_reference_projection\":{\"valid\":"
            << (reference.tangentReferenceProjectionValid ? "true" : "false")
            << ",\"failed\":" << (reference.tangentReferenceProjectionFailed ? "true" : "false")
            << ",\"failure\":\"" << jsonEscape(reference.tangentReferenceProjectionFailure) << "\""
            << ",\"sample_index\":" << reference.tangentProjectionSample
            << ",\"iteration\":" << reference.tangentProjectionIteration
            << ",\"residual_norm\":";
        emitJsonNumber(output, reference.tangentProjectionResidual);
        output
            << ",\"worst_row\":" << reference.tangentProjectionWorstRow
            << ",\"worst_residual\":";
        emitJsonNumber(output, reference.tangentProjectionWorstResidual);
        output
            << ",\"line_search_trials\":" << reference.tangentProjectionLineSearchTrials
            << ",\"last_alpha\":";
        emitJsonNumber(output, reference.tangentProjectionLastAlpha);
        output << ",\"last_trial_residual\":";
        emitJsonNumber(output, reference.tangentProjectionLastTrialResidual);
        output << ",\"best_trial_residual\":";
        emitJsonNumber(output, reference.tangentProjectionBestTrialResidual);
        output << "},\"legacy_independent_residual_at_metal_state_source_norm_not_comparable\":";
        if (reference.oracleProjectionValid) output << reference.fp64ResidualAtMetalState;
        else output << "null";
        output << ",\"fp64_root_high_low_rounding_residual_estimate_not_metal_noise\":";
        if (reference.oracleProjectionValid) output << reference.twoWordStateResidualFloor;
        else output << "null";
        output << ",\"legacy_residual_absolute_difference_between_unlike_norms_not_qualification\":";
        if (reference.oracleProjectionValid) output << reference.residualAbsoluteDifference;
        else output << "null";
        output << ",\"legacy_unlike_norm_ratio_not_for_qualification\":";
        if (reference.oracleProjectionValid) output << reference.residualRelativeToFloor;
        else output << "null";
        output << ",\"legacy_residual_comparison_status\":\"invalid_incomparable_norms_not_a_qualification_metric\""
            << ",\"status\":\"" << exactReplayStatus << "\"}}\n";
        need(output.good(), "captured FEM center replay output write failed");
    }

    need(foundOnlySelection, "requested environment:tet selection was not present");
    if (onlySelection.empty()) {
        need(recordCount == environmentCount * tetrahedraPerEnvironment,
             "capture tetrahedron record count differs from the header");
        for (unsigned environment = 0u; environment < environmentCount; ++environment)
            for (unsigned tet = 0u; tet < tetrahedraPerEnvironment; ++tet)
                need(seen[environment][tet], "capture is missing a tetrahedron center record");
        need(integrityRecord,
             "capture is missing a successful production-input integrity record");
        need(paperCenters + skippedGlue == recordCount,
             "center traversal did not classify every captured tetrahedron");
    }
    output << "{\"record\":\"summary\",\"physical_steps\":0,\"physical_validation\":false"
        << ",\"capture_sha256\":\"" << captureSHA << "\""
        << ",\"tetrahedron_records_seen\":" << recordCount
        << ",\"paper_centers\":" << paperCenters
        << ",\"paper_centers_replayed\":" << replayed
        << ",\"glue_centers_skipped\":" << skippedGlue
        << ",\"ordinary_invalid\":" << failedOrdinary
        << ",\"accurate_invalid\":" << failedAccurate
        << ",\"exact_reference_center_projection_failures\":"
        << exactCenterProjectionFailures
        << ",\"exact_reference_tangent_projection_failures\":"
        << exactTangentProjectionFailures
        << ",\"ordinary_nonpass\":" << ordinaryNonpass
        << ",\"accurate_fp64_nonpass\":" << accurateNonpass
        << ",\"selection\":\"" << (onlySelection.empty() ? "all" : onlySelection)
        << "\",\"capture_integrity_verified\":"
        << (integrityRecord ? "true" : "false") << "}\n";
    need(output.good(), "captured FEM center summary write failed");
    output.close();
    need(!output.fail(), "captured FEM center output close failed");
    // Validate the emitted instrument contract, including the nested policy
    // arrays, before publishing completion. A valid input capture and a
    // completed GPU replay do not establish that our output is readable.
    std::ifstream emitted(outputFile);
    need(emitted.good(), "cannot reopen captured FEM center replay output");
    unsigned emittedHeaders = 0u, emittedCenters = 0u;
    unsigned emittedGlue = 0u, emittedSummaries = 0u;
    std::string emittedLine;
    while (std::getline(emitted, emittedLine)) {
        @autoreleasepool {
            NSDictionary* row = jsonDictionary(
                parseJsonLine(emittedLine, "emitted replay record"),
                "emitted replay record");
            const std::string type = jsonFieldString(
                row, @"record", "emitted replay record");
            need(emittedSummaries == 0u,
                 "emitted replay has records after its terminal summary");
            if (type == "header") {
                need(emittedHeaders == 0u && emittedCenters == 0u &&
                         emittedGlue == 0u,
                     "emitted replay header is duplicated or out of order");
                ++emittedHeaders;
            } else if (type == "captured_center_replay") {
                need(emittedHeaders == 1u,
                     "emitted replay center precedes its header");
                NSDictionary* rows = jsonDictionary(
                    jsonRequired(row, @"compiled_residual_rows", "emitted center"),
                    "emitted residual rows");
                for (const auto& policy : {
                         std::pair<NSString*, NSString*>{
                             @"native_ordinary_policy", @"ordinary_replay"},
                         std::pair<NSString*, NSString*>{
                             @"native_accurate_policy", @"accurate_candidate"}}) {
                    NSDictionary* object = jsonDictionary(
                        jsonRequired(rows, policy.first, "emitted residual rows"),
                        "emitted native residual policy");
                    NSArray* candidates = jsonArray(
                        jsonRequired(object, policy.second, "emitted native residual policy"),
                        "emitted native candidate rows");
                    need(candidates.count == kReplayCandidateCount,
                         "emitted native residual candidate count differs");
                }
                ++emittedCenters;
            } else if (type == "skipped_center") {
                need(emittedHeaders == 1u,
                     "emitted skipped center precedes its header");
                ++emittedGlue;
            } else if (type == "summary") {
                need(emittedHeaders == 1u,
                     "emitted replay summary precedes its header");
                ++emittedSummaries;
            } else {
                throw std::runtime_error("unexpected emitted replay record type: " + type);
            }
        }
    }
    need(emitted.eof() && emittedHeaders == 1u && emittedSummaries == 1u &&
             emittedCenters == replayed && emittedGlue == skippedGlue,
         "emitted replay JSONL coverage or completion differs from its execution");
    std::cout << "captured_fem_center_records=" << recordCount
        << " paper_centers=" << paperCenters
        << " replayed=" << replayed
        << " glue_skipped=" << skippedGlue
        << " ordinary_invalid=" << failedOrdinary
        << " accurate_invalid=" << failedAccurate
        << " exact_center_reference_projection_failures=" << exactCenterProjectionFailures
        << " exact_tangent_reference_projection_failures=" << exactTangentProjectionFailures
        << " ordinary_nonpass=" << ordinaryNonpass
        << " accurate_fp64_nonpass=" << accurateNonpass
        << " physical_steps=0 emitted_json_schema_validated=true\n";
    return failedOrdinary || failedAccurate || ordinaryNonpass || accurateNonpass ? 2 : 0;
}
}
int main(int argc,char** argv){@autoreleasepool{try{
    if(argc>=6&&std::string(argv[1])=="--replay-fem-centers") {
        std::string selection;
        if(argc==8&&std::string(argv[6])=="--only") {
            selection=argv[7];
            const auto separator=selection.find(':');
            need(separator!=std::string::npos&&separator>0u&&
                 separator+1u<selection.size()&&
                 selection.find(':',separator+1u)==std::string::npos&&
                 std::all_of(selection.begin(),selection.end(),[&](unsigned char c){
                     return c==':'||std::isdigit(c)!=0;
                 }),"--only expects one ENV:TET integer selector");
        } else need(argc==6,
            "usage: --replay-fem-centers INPUT OUTPUT LINER MEDIUM [--only ENV:TET]");
        Instrument gpu;
        return replayCapturedFemCenters(gpu,argv[2],argv[3],argv[4],argv[5],selection);
    }
    if(argc>=6&&std::string(argv[1])=="--replay-captured") {
        bool scaled=false,initialized=false,compensated=false,accurateProjection=false;
        for(int i=6;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="--scaled-residuals"&&!scaled)scaled=true;
            else if(option=="--safeguarded-initializer"&&!initialized)initialized=true;
            else if(option=="--compensated-arithmetic"&&!compensated)compensated=true;
            else if(option=="--accurate-projection"&&!accurateProjection)accurateProjection=true;
            else throw std::runtime_error("unknown or duplicate replay option: "+option);
        }
        Instrument gpu;return replayCaptured(gpu,argv[2],argv[3],argv[4],argv[5],scaled,initialized,compensated,accurateProjection);
    }
    need(argc==3,"expected liner and medium material paths, --replay-captured INPUT OUTPUT LINER MEDIUM [--accurate-projection], or --replay-fem-centers INPUT OUTPUT LINER MEDIUM [--only ENV:TET]");
    Instrument gpu;check(gpu,argv[1]);check(gpu,argv[2]);std::cout<<"PASS paper production material projection; physical_validation=false\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}}
