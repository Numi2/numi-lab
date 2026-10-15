#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "cardboard_paper_reference.hpp"
#include "cardboard_paper_projection_scaling.hpp"
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
struct Instrument {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLComputePipelineState> pipeline;
    Instrument(){device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal unavailable");
        const std::string name=[[device name] UTF8String];need(name.find("Apple")!=std::string::npos&&name.find("Paravirtual")==std::string::npos,"physical Apple GPU required");
        queue=[device newCommandQueue];NSError* error=nil;
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&error];need(lib!=nil,"paper instrument library missing");
        pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cardboard_paper_check"] error:&error];need(pipeline!=nil,"paper instrument kernel missing");
        std::cout<<"device="<<name<<" production_projection_and_algorithmic_tangent=true physical_steps=0\n";
    }
    Response evaluate(const CompiledWorld& w,const Matrix& f,const Matrix& fLow,
                      const Matrix& h,const std::vector<float>& state,bool accurateProjection){
        std::vector<float> inputs;for(double v:f)inputs.push_back(float(v));
        for(double v:h)inputs.push_back(float(v));
        for(double v:fLow)inputs.push_back(float(v));
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
    double fp64ResidualAtMetalState=0.0,twoWordStateResidualFloor=0.0;
    double residualAbsoluteDifference=0.0,residualRelativeToFloor=0.0;
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

numi_cardboard_paper::Result projectExactParameters(
    const numi_cardboard_paper::Parameters& p,const Matrix& f,
    const numi_cardboard_paper::State& accepted){
    using namespace numi_cardboard_paper;
    State x=accepted;
    const Vector6 trial=normalizedSecondPiola(p,f,plasticStrain(accepted));
    for(unsigned i=0;i<6;++i)x[6+i]=trial[i];
    unsigned iterations=0;
    constexpr unsigned maxIterations=32;
    constexpr double tolerance=1.0e-12;
    for(;iterations<maxIterations;++iterations){
        const auto r=residual(p,f,accepted,x);
        const double norm=residualNorm(r);
        double stateNorm=0.0;
        for(double value:x)stateNorm=std::max(stateNorm,std::abs(value));
        if(norm<=tolerance*(1.0+stateNorm))break;
        std::array<double,169> jac{};
        for(unsigned column=0;column<13;++column){
            State plus=x,minus=x;
            const double step=1.0e-7*(1.0+std::abs(x[column]));
            plus[column]+=step;minus[column]-=step;
            const auto rp=residual(p,f,accepted,plus);
            const auto rm=residual(p,f,accepted,minus);
            for(unsigned row=0;row<13;++row)
                jac[13*row+column]=(rp[row]-rm[row])/(2.0*step);
        }
        std::array<double,13> rhs{},step{};
        for(unsigned i=0;i<13;++i)rhs[i]=-r[i];
        need(solveLinear(jac,rhs,step),"exact-parameter FP64 paper projection is singular");
        double alpha=1.0;
        bool acceptedStep=false;
        for(unsigned backtrack=0;backtrack<20;++backtrack){
            State candidate=x;
            for(unsigned i=0;i<13;++i)candidate[i]+=alpha*step[i];
            if(residualNorm(residual(p,f,accepted,candidate))<
               (1.0-1.0e-4*alpha)*norm){
                x=candidate;acceptedStep=true;break;
            }
            alpha*=0.5;
        }
        need(acceptedStep,"exact-parameter FP64 paper projection line search failed");
    }
    const double finalNorm=residualNorm(residual(p,f,accepted,x));
    double stateNorm=0.0;
    for(double value:x)stateNorm=std::max(stateNorm,std::abs(value));
    need(finalNorm<=tolerance*(1.0+stateNorm),
         "exact-parameter FP64 paper projection did not converge");
    const Vector6 ep=plasticStrain(x),s=returnedNormalizedStress(p,f,accepted,x);
    Vector6 dep{};
    for(unsigned i=0;i<6;++i)dep[i]=x[i]-accepted[i];
    const Matrix second=secondPiola(p,f,ep),first=multiply(f,second);
    const double work=second[0]*dep[0]+second[4]*dep[1]+second[8]*dep[2]+
        2.0*(second[5]*dep[3]+second[2]*dep[4]+second[1]*dep[5]);
    return {x,second,first,hillEquivalent(p,s),finalNorm,work,iterations};
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
    const auto oracle=projectExactParameters(oracleParameters,exactF,accepted64);
    State metalState{};
    for(unsigned i=0;i<material.internalState.size();++i)
        metalState[i]=double(metal.state.at(i))+double(metal.stateLow.at(i));

    ExactMetrics result;
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
        for(unsigned sample=0;sample<deformationSteps.size();++sample){
            const double step=deformationSteps[sample]/directionScale;
            Matrix plus=exactF,minus=exactF;
            for(unsigned i=0;i<9;++i){plus[i]+=step*h[i];minus[i]-=step*h[i];}
            const auto oraclePlus=projectExactParameters(oracleParameters,plus,accepted64);
            const auto oracleMinus=projectExactParameters(oracleParameters,minus,accepted64);
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
            const bool residualWithinMeasuredFloor=
                metrics.residualRelativeToFloor<=1.0;
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
                (!metrics.zeroDirection&&!metrics.tangentFiniteDifferenceNoiseResolved)||
                !residualWithinMeasuredFloor;
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
            const char* residualStatus=residualWithinMeasuredFloor?"within_measured_state_floor":
                "inconclusive_outside_measured_state_floor";
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
                <<",\"independent_residual_at_metal_state\":"
                <<metrics.fp64ResidualAtMetalState<<",\"two_word_state_residual_floor\":"
                <<metrics.twoWordStateResidualFloor<<",\"metal_vs_fp64_residual_absolute_error\":"
                <<metrics.residualAbsoluteDifference<<",\"residual_disagreement_over_two_word_representation_floor\":"
                <<metrics.residualRelativeToFloor<<",\"residual_comparison_status\":\""
                <<residualStatus<<"\",\"local_projection_threshold\":"
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
                    <<"\",\"projection_residual_relative_error\":";
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
}
int main(int argc,char** argv){@autoreleasepool{try{
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
    need(argc==3,"expected liner and medium material paths, or --replay-captured INPUT OUTPUT LINER MEDIUM [--accurate-projection]");
    Instrument gpu;check(gpu,argv[1]);check(gpu,argv[2]);std::cout<<"PASS paper production material projection; physical_validation=false\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}}
