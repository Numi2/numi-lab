#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "cardboard_paper_reference.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

namespace {
using namespace numi::matter;
using Matrix=std::array<double,9>;
constexpr Matrix identity{1,0,0,0,1,0,0,0,1};
void need(bool v,const std::string& s){if(!v)throw std::runtime_error(s);}
std::string diagnostics(const std::vector<Diagnostic>& ds){std::string s;for(const auto& d:ds)s+=d.message+"; ";return s;}
double parameter(const MaterialProgram& m,const std::string& n){for(const auto& p:m.parameters)if(p.name==n)return p.defaultValue;throw std::runtime_error("missing "+n);}
template<class T> id<MTLBuffer> buffer(id<MTLDevice> d,const std::vector<T>& x){T z{};return [d newBufferWithBytes:x.empty()?&z:x.data() length:sizeof(T)*std::max(std::size_t(1),x.size()) options:MTLResourceStorageModeShared];}
struct Response{bool valid=false;bool projectionValid=false;Matrix p{},dp{};std::vector<float> state;};
struct Instrument {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLComputePipelineState> pipeline;
    Instrument(){device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal unavailable");
        const std::string name=[[device name] UTF8String];need(name.find("Apple")!=std::string::npos&&name.find("Paravirtual")==std::string::npos,"physical Apple GPU required");
        queue=[device newCommandQueue];NSError* error=nil;
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&error];need(lib!=nil,"paper instrument library missing");
        pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cardboard_paper_check"] error:&error];need(pipeline!=nil,"paper instrument kernel missing");
        std::cout<<"device="<<name<<" production_projection_and_algorithmic_tangent=true physical_steps=0\n";
    }
    Response evaluate(const CompiledWorld& w,const Matrix& f,const Matrix& h,const std::vector<float>& state){
        std::vector<float> inputs;for(double v:f)inputs.push_back(float(v));for(double v:h)inputs.push_back(float(v));
        std::vector<float> parameters;for(const auto& v:w.parameters)parameters.push_back(v.valueAndBounds.x);
        std::array<id<MTLBuffer>,7> b{buffer(device,std::vector<NMMaterialGPU>{w.materials.at(0)}),buffer(device,w.scalarPrograms),buffer(device,w.instructions),buffer(device,parameters),buffer(device,state),buffer(device,inputs),buffer(device,std::vector<float>(20+NM_MAX_MATERIAL_STATE,0))};
        auto cb=[queue commandBuffer];auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:pipeline];
        for(NSUInteger i=0;i<b.size();++i)[encoder setBuffer:b[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[encoder endEncoding];[cb commit];[cb waitUntilCompleted];
        need(cb.status==MTLCommandBufferStatusCompleted,"Metal paper evaluation failed");
        const auto* out=static_cast<const float*>(b[6].contents);Response r;r.valid=out[0]==1;r.projectionValid=out[35]==1;
        for(unsigned i=0;i<9;++i){r.p[i]=out[1+i];r.dp[i]=out[10+i];}
        r.state.assign(out+19,out+19+w.materials.at(0).stateCount);return r;
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
}
int main(int argc,char** argv){@autoreleasepool{try{need(argc==3,"expected liner and medium material paths");Instrument gpu;check(gpu,argv[1]);check(gpu,argv[2]);std::cout<<"PASS paper production material projection; physical_validation=false\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}}
