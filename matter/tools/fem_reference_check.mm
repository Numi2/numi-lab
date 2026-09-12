#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include "cardiac_material_reference.hpp"
#include "../src/fem_reference_geometry.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

// Synthetic conforming two-tet explicit-reference constitutive/transaction fixture.
// The Guccione and neo-Hookean energies are sourced; these densities, geometry,
// loading, state/rate laws and learned networks are numerical controls. This
// does not supply anatomical density or a source-calibrated cardiac wall.
namespace {
using namespace numi::matter;
namespace ref=numi_cardiac_reference;
using ref::Matrix;
using Vec=std::array<double,3>;
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::string number(double x){std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();}
std::string diagnostic(const std::vector<Diagnostic>& values){std::string out;for(const auto& v:values)out+=v.message+"; ";return out;}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
template<class T> std::vector<T> repeat(const std::vector<T>& a,unsigned count=2){std::vector<T> out;for(unsigned i=0;i<count;++i)out.insert(out.end(),a.begin(),a.end());return out;}
template<class T> void paired(const std::vector<T>& a,const char* label){need(a.size()%2==0,std::string(label)+" arity");const auto n=a.size()/2;need(!n||std::memcmp(a.data(),a.data()+n,n*sizeof(T))==0,std::string(label)+" paired environments differ");}
template<class T> void envSame(const std::vector<T>& a,const std::vector<T>& b,unsigned e,const char* label){need(a.size()==b.size()&&a.size()%2==0,std::string(label)+" arity");const auto n=a.size()/2;need(!n||std::memcmp(a.data()+e*n,b.data()+e*n,n*sizeof(T))==0,std::string(label)+" isolated transaction changed state");}
template<class T> id<MTLBuffer> buffer(id<MTLDevice> d,const std::vector<T>& input){const T zero{};return [d newBufferWithBytes:(input.empty()?&zero:input.data()) length:std::max(std::size_t(1),input.size())*sizeof(T) options:MTLResourceStorageModeShared];}
Vec xyz(const nm_float4& a){return {a.x,a.y,a.z};}
Vec sub(Vec a,const Vec& b){for(unsigned i=0;i<3;++i)a[i]-=b[i];return a;}
Vec transform(const Matrix& a,const Vec& b){Vec c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[i]+=a[3*i+j]*b[j];return c;}
double dot(const Vec& a,const Vec& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
Matrix outer(const Vec& a,const Vec& b){Matrix c{};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)c[3*i+j]=a[i]*b[j];return c;}
Matrix columns(const Vec& a,const Vec& b,const Vec& c){return {a[0],b[0],c[0],a[1],b[1],c[1],a[2],b[2],c[2]};}
Matrix inverseRest(const NMTetrahedronGPU& t){return {t.inverseRestRow0.x,t.inverseRestRow0.y,t.inverseRestRow0.z,t.inverseRestRow1.x,t.inverseRestRow1.y,t.inverseRestRow1.z,t.inverseRestRow2.x,t.inverseRestRow2.y,t.inverseRestRow2.z};}
Matrix frame(const nm_float4& raw){
    const double inv=1/std::sqrt(double(raw.x)*raw.x+double(raw.y)*raw.y+double(raw.z)*raw.z+double(raw.w)*raw.w);
    const double x=raw.x*inv,y=raw.y*inv,z=raw.z*inv,w=raw.w*inv;
    return {1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
}
void set(MaterialProgram& m,const std::string& name,double value){for(auto& p:m.parameters)if(p.name==name){p.defaultValue=value;return;}throw std::runtime_error("missing "+name);}
WorldSource source(const MaterialProgram& first,const MaterialProgram& second,bool combined=false){
    WorldSource s;s.environmentCount=2;s.gravity={0,0,0};s.frameTimestep=1e-4;
    s.materials={first,second};s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=12;s.mixedSolver.fgmresIterations=64;
    ObjectSource o;o.name="synthetic_independently_authored_reference";o.representation=Representation::fem;o.mixedFEM=false;
    o.deformableContact=false;o.deformableSelfContact=false;o.characteristicLength=.015625;
    constexpr double l=.015625;
    o.femReferenceNodes={{{0,0,0}},{{l,0,0}},{{0,l,0}},{{0,0,l}},{{0,0,-l}}};
    // Exact binary affine preload, independent of the material parser and
    // cooker. It is supplied data, not an inverse unloading algorithm.
    const Matrix f{1.015625,.0078125,0,0,1,0,0,0,1};const Vec translation{.03125,-.015625,.0078125};
    for(const auto& x:o.femReferenceNodes){auto p=transform(f,x);for(unsigned j=0;j<3;++j)p[j]+=translation[j];o.femNodes.push_back(p);}
    o.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};o.femFixedNodes={0,1,2};
    o.femReferenceSourceIdentity={0x29595b22aedcde73ull,0x6f1d688db49d2714ull,0x6eed2d5d77f3dbd6ull,0x6f9ec94305b73627ull};
    if(combined){o.femMaterialIndices={0,1};o.femMaterialSourceIdentity={1,2,3,4};
        o.femMaterialFrameRotations={{{0,0,std::sin(.305),std::cos(.305)}},{{0,std::sin(-.245),0,std::cos(-.245)}}};
        o.femMaterialFrameSourceIdentity={5,6,7,8};}
    s.objects={o};return s;
}
CompiledWorld cook(const WorldSource& s){auto c=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});need(c.succeeded(),"reference fixture compile: "+diagnostic(c.diagnostics));return std::move(c.world);}
struct Device {
    id<MTLDevice> device;id<MTLCommandQueue> queue;id<MTLLibrary> library;
    id<MTLComputePipelineState> forcePipeline,operatorPipeline;
    Device(){device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal unavailable");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
        queue=[device newCommandQueue];NSError* error=nil;
        library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:NUMI_MATTER_METALLIB]] error:&error];need(library!=nil,"production Matter metallib unavailable");
        forcePipeline=make(@"numi_matter_metal::nm_fem_internal_forces");operatorPipeline=make(@"numi_matter_metal::nm_fem_apply_operator_elements");}
    id<MTLComputePipelineState> make(NSString* name){NSError* error=nil;auto f=[library newFunctionWithName:name];need(f!=nil,"production FEM function missing");auto p=[device newComputePipelineStateWithFunction:f error:&error];need(p!=nil,"production FEM pipeline failed");return p;}
};
struct Inputs {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMFEMFieldStateGPU> fields;
    std::vector<nm_float4> direction;
    std::vector<float> state;
};
Inputs manufactured(const CompiledWorld& w){
    Inputs v;v.nodes=repeat(w.fem.nodes);v.fields=repeat(w.fem.fields);v.direction.resize(4*w.dispatch.femNodeCount);
    const Matrix h{.19,-.13,.07,.11,-.17,.04,-.05,.09,.14};
    for(unsigned env=0;env<2;++env)for(unsigned n=0;n<w.dispatch.femNodeCount;++n){const unsigned i=env*w.dispatch.femNodeCount+n;
        const Vec x=xyz(w.fem.nodes[n].restAndFixed),d=transform(h,x);
        v.direction[i]={float(d[0]),float(d[1]),float(d[2]),0};}
    v.state.resize(2*w.dispatch.tetrahedronCount*w.dispatch.materialStateStride);
    for(unsigned e=0;e<2;++e)for(unsigned ti=0;ti<w.dispatch.tetrahedronCount;++ti){
        const auto& m=w.materials[w.fem.tetrahedra[ti].identity.x];
        for(unsigned i=0;i<m.stateCount;++i)v.state[(e*w.dispatch.tetrahedronCount+ti)*w.dispatch.materialStateStride+i]=w.stateInitials[m.stateInitialOffset+i];}
    return v;
}
std::vector<NMFEMElementVectorGPU> dispatch(Device& d,const CompiledWorld& w,const Inputs& input,bool tangent){
    NMMicrostepGPU micro{};micro.flags=NM_MICROSTEP_FGMRES_OPERATOR;micro.time={w.dispatch.gravityAndTimestep.w,1/w.dispatch.gravityAndTimestep.w,0,0};
    std::vector<float> parameters;for(unsigned e=0;e<2;++e)for(const auto& p:w.parameters)parameters.push_back(p.valueAndBounds.x);
    const auto objects=buffer(d.device,w.objects),materials=buffer(d.device,w.materials),programs=buffer(d.device,w.scalarPrograms),instructions=buffer(d.device,w.instructions);
    const auto params=buffer(d.device,parameters),nodes=buffer(d.device,input.nodes),tets=buffer(d.device,repeat(w.fem.tetrahedra)),state=buffer(d.device,input.state);
    const auto schedulers=buffer(d.device,repeat(w.schedulers)),adaptive=buffer(d.device,repeat(w.adaptive)),mixed=buffer(d.device,w.mixedMaterials),fields=buffer(d.device,input.fields);
    const auto learned=buffer(d.device,w.learnedMaterials),layers=buffer(d.device,w.learnedLayers),weights=buffer(d.device,w.learnedWeights),direction=buffer(d.device,input.direction);
    const auto solver=buffer(d.device,std::vector<NMFGMRESStateGPU>(2)),status=buffer(d.device,std::vector<NMMatterStatusGPU>(2));
    const unsigned count=2*w.dispatch.tetrahedronCount;const auto output=buffer(d.device,std::vector<NMFEMElementVectorGPU>(count));
    auto command=[d.queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:tangent?d.operatorPipeline:d.forcePipeline];
    [encoder setBytes:&w.dispatch length:sizeof(w.dispatch) atIndex:0];[encoder setBytes:&micro length:sizeof(micro) atIndex:1];
    [encoder setBuffer:objects offset:0 atIndex:2];[encoder setBuffer:materials offset:0 atIndex:3];[encoder setBuffer:programs offset:0 atIndex:4];[encoder setBuffer:instructions offset:0 atIndex:5];
    [encoder setBuffer:params offset:0 atIndex:6];[encoder setBuffer:nodes offset:0 atIndex:7];[encoder setBuffer:tets offset:0 atIndex:8];[encoder setBuffer:state offset:0 atIndex:9];
    if(tangent){[encoder setBuffer:direction offset:0 atIndex:10];[encoder setBuffer:schedulers offset:0 atIndex:11];[encoder setBuffer:adaptive offset:0 atIndex:12];[encoder setBuffer:output offset:0 atIndex:13];[encoder setBuffer:status offset:0 atIndex:14];
        [encoder setBuffer:mixed offset:0 atIndex:16];[encoder setBuffer:fields offset:0 atIndex:17];[encoder setBuffer:learned offset:0 atIndex:18];[encoder setBuffer:layers offset:0 atIndex:19];[encoder setBuffer:weights offset:0 atIndex:20];
        [encoder setBytes:&w.mixedSolver length:sizeof(w.mixedSolver) atIndex:21];[encoder setBuffer:solver offset:0 atIndex:22];}
    else{[encoder setBuffer:schedulers offset:0 atIndex:10];[encoder setBuffer:adaptive offset:0 atIndex:11];[encoder setBuffer:output offset:0 atIndex:12];[encoder setBuffer:status offset:0 atIndex:13];
        [encoder setBuffer:mixed offset:0 atIndex:14];[encoder setBuffer:fields offset:0 atIndex:15];[encoder setBuffer:learned offset:0 atIndex:16];[encoder setBuffer:layers offset:0 atIndex:17];[encoder setBuffer:weights offset:0 atIndex:18];}
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];
    need(command.status==MTLCommandBufferStatusCompleted,"direct production FEM command failed");
    const auto* statuses=static_cast<const NMMatterStatusGPU*>(status.contents);for(unsigned e=0;e<2;++e)need(statuses[e].code==NM_STATUS_SUCCESS,"direct production FEM rejected state code="+std::to_string(statuses[e].code));
    const auto* values=static_cast<const NMFEMElementVectorGPU*>(output.contents);return {values,values+count};
}
Matrix gradient(const CompiledWorld& w,const Inputs& v,unsigned e,unsigned ti,bool rate=false,bool direction=false){
    const auto& t=w.fem.tetrahedra[ti];const unsigned base=e*w.dispatch.femNodeCount;
    const auto at=[&](unsigned n){return direction?xyz(v.direction[base+n]):rate?xyz(v.nodes[base+n].velocityAndInverseMass):xyz(v.nodes[base+n].positionAndMass);};
    const Vec x=at(t.nodes.x);return ref::multiply(columns(sub(at(t.nodes.y),x),sub(at(t.nodes.z),x),sub(at(t.nodes.w),x)),inverseRest(t));
}
std::array<Vec,4> forces(const NMTetrahedronGPU& t,const Matrix& p){
    const Matrix a=ref::scale(ref::multiply(p,ref::transpose(inverseRest(t))),-double(t.inverseRestRow0.w));
    std::array<Vec,4> out{};for(unsigned n=1;n<4;++n)for(unsigned i=0;i<3;++i){out[n][i]=a[3*i+n-1];out[0][i]-=out[n][i];}return out;
}
std::array<Vec,4> forces(const NMFEMElementVectorGPU& v){return {xyz(v.node0),xyz(v.node1),xyz(v.node2),xyz(v.node3)};}
using Oracle=std::function<ref::Result(unsigned,const Matrix&,const Matrix&,const Matrix&,const Matrix&)>;
double compare(const CompiledWorld& w,const Inputs& v,const std::vector<NMFEMElementVectorGPU>& actual,bool tangent,const Oracle& oracle){
    double error=0,scale=0;
    for(unsigned e=0;e<2;++e)for(unsigned ti=0;ti<w.dispatch.tetrahedronCount;++ti){const auto& t=w.fem.tetrahedra[ti];
        const Matrix q=(w.objects[t.identity.y].flags&NM_OBJECT_FEM_MATERIAL_FRAME)?frame(t.materialFrameRotation):ref::identity;
        const auto result=oracle(ti,gradient(w,v,e,ti),gradient(w,v,e,ti,false,true),gradient(w,v,e,ti,true),q);
        const auto expected=forces(t,tangent?result.tangent:result.stress),observed=forces(actual[e*w.dispatch.tetrahedronCount+ti]);
        for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i){need(std::isfinite(observed[n][i]),"nonfinite production force/tangent");error=std::max(error,std::abs(observed[n][i]-expected[n][i]));scale=std::max(scale,std::abs(expected[n][i]));}}
    return error/std::max(1e-10,scale);
}
double finiteDifference(Device& d,const CompiledWorld& w,const Inputs& v,const std::vector<NMFEMElementVectorGPU>& tangent){
    constexpr double epsilon=1./2048;Inputs plus=v,minus=v;const double dt=w.dispatch.gravityAndTimestep.w;const unsigned count=2*w.dispatch.femNodeCount;
    for(unsigned i=0;i<count;++i){const auto p=v.direction[i];
        plus.nodes[i].positionAndMass.x+=epsilon*p.x;minus.nodes[i].positionAndMass.x-=epsilon*p.x;
        plus.nodes[i].positionAndMass.y+=epsilon*p.y;minus.nodes[i].positionAndMass.y-=epsilon*p.y;
        plus.nodes[i].positionAndMass.z+=epsilon*p.z;minus.nodes[i].positionAndMass.z-=epsilon*p.z;
        plus.nodes[i].velocityAndInverseMass.x+=epsilon*p.x/dt;minus.nodes[i].velocityAndInverseMass.x-=epsilon*p.x/dt;
        plus.nodes[i].velocityAndInverseMass.y+=epsilon*p.y/dt;minus.nodes[i].velocityAndInverseMass.y-=epsilon*p.y/dt;
        plus.nodes[i].velocityAndInverseMass.z+=epsilon*p.z/dt;minus.nodes[i].velocityAndInverseMass.z-=epsilon*p.z/dt;
        plus.fields[i].secondary.x+=epsilon*v.direction[count+i].w/dt;minus.fields[i].secondary.x-=epsilon*v.direction[count+i].w/dt;}
    const auto fp=dispatch(d,w,plus,false),fm=dispatch(d,w,minus,false);double error=0,scale=0;
    for(unsigned e=0;e<tangent.size();++e){const auto a=forces(fp[e]),b=forces(fm[e]),t=forces(tangent[e]);for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i){const double fd=(a[n][i]-b[n][i])/(2*epsilon);error=std::max(error,std::abs(fd-t[n][i]));scale=std::max(scale,std::abs(t[n][i]));}}
    return error/std::max(1e-10,scale);
}
struct Run {
    Device& device;CompiledWorld world;Runtime runtime;id<MTLBuffer> statuses;
    Run(Device& d,const WorldSource& s):device(d),world(cook(s)){
        const auto path=std::filesystem::temp_directory_path()/(std::string("numi-reference-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        CompileResult package;package.world=world;std::string error;need(writePackage(package,path,&error),error);CompiledWorld loaded;
        const bool ok=readPackage(path,loaded,nullptr,&error);std::error_code remove;std::filesystem::remove(path,remove);need(ok&&!remove&&loaded.fingerprint==world.fingerprint,"reference package roundtrip: "+error);world=std::move(loaded);
        RuntimeConfiguration c;c.metallib=NUMI_MATTER_METALLIB;c.environmentCount=2;c.adaptiveTransfer=false;c.captureEvents=false;c.captureDiagnostics=true;
        const auto initialized=runtime.initialize(world,c);need(initialized.encoded,initialized.message);statuses=buffer(device.device,std::vector<MRMetalWorldStatusGPU>(2));}
    RuntimeStateSnapshot state(){auto s=runtime.snapshot();need(s.available,s.message);return s;}
    void restore(const RuntimeStateSnapshot& s){const auto r=runtime.restore(s);need(r.encoded,r.message);}
    void step(unsigned index,int reject=-1,int reset=-1){
        auto* incoming=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);for(unsigned e=0;e<2;++e){incoming[e]={};incoming[e].environment=e;}
        auto command=[device.queue commandBuffer];EncodeRequest request;request.commandBuffer=(__bridge void*)command;request.environmentStatuses=(__bridge void*)statuses;
        request.controlStep=index;request.physicsSubsteps=1;request.timestepSeconds=runtime.timestepSeconds();request.runAdaptiveTransfer=false;
        id<MTLBuffer> resets=nil,failure=nil;
        if(reset>=0){std::vector<unsigned> mask((index+1)*2);mask[index*2+unsigned(reset)]=1;resets=buffer(device.device,mask);request.resetMasks=(__bridge void*)resets;request.resetMaskStepStride=2;}
        request.phase=EncodePhase::preDynamics;auto result=runtime.encode(request);need(result.encoded,result.message);
        if(reject>=0){MRMetalWorldStatusGPU bad{};bad.environment=unsigned(reject);bad.code=MR_STEP_DID_NOT_CONVERGE;failure=buffer(device.device,std::vector<MRMetalWorldStatusGPU>{bad});auto blit=[command blitCommandEncoder];[blit copyFromBuffer:failure sourceOffset:0 toBuffer:statuses destinationOffset:unsigned(reject)*sizeof(bad) size:sizeof(bad)];[blit endEncoding];}
        request.phase=EncodePhase::postCommit;result=runtime.encode(request);need(result.encoded,result.message);[command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"integrated reference command failed");
        const auto after=state();for(unsigned e=0;e<2;++e)if(int(e)!=reject)need(after.statuses[e].code==NM_STATUS_SUCCESS,"integrated reference step="+std::to_string(index)+" env="+std::to_string(e)+" status="+std::to_string(after.statuses[e].code)+" residual="+number(after.statuses[e].diagnostics.z));
        if(reject>=0)need(after.statuses[unsigned(reject)].code!=NM_STATUS_SUCCESS,"injected reference rejection lost");(void)resets;(void)failure;
    }
};
bool physicalSame(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b){return same(a.femNodes,b.femNodes)&&same(a.femFields,b.femFields)&&same(a.femMaterialState,b.femMaterialState)&&same(a.femTopologyTetrahedra,b.femTopologyTetrahedra)&&same(a.femTopologyNodes,b.femTopologyNodes)&&same(a.cohesiveFaces,b.cohesiveFaces)&&same(a.punctureChannels,b.punctureChannels)&&same(a.topologyStates,b.topologyStates)&&same(a.environmentParameters,b.environmentParameters)&&same(a.identification,b.identification);}

unsigned cpuCheckCount=0;
void cpuChecks(const WorldSource& authored){
    const auto w=cook(authored);need(w.fem.nodes.size()==5&&w.fem.tetrahedra.size()==2,"reference fixture duplicated shared nodes");
    need((w.objects[0].flags&NM_OBJECT_FEM_REFERENCE_CONFIGURATION)!=0,"reference ownership absent");
    std::vector<double> mass(5);double elementMass=0;
    constexpr double volume=.015625*.015625*.015625/6;
    for(const auto& t:w.fem.tetrahedra){need(t.inverseRestRow0.w==float(volume),"reference volume came from loaded geometry");
        const double m=double(w.materials[t.identity.x].bulk.x)*double(t.inverseRestRow0.w);elementMass+=m;
        for(unsigned n:{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w})mass[n]+=.25*m;}
    Vec referenceMoment{},currentMoment{};double nodalMass=0;
    for(unsigned n=0;n<5;++n){const auto& node=w.fem.nodes[n];const float m=float(mass[n]),inv=n<3?0.f:float(1/mass[n]);
        need(node.positionAndMass.w==m&&node.velocityAndInverseMass.w==inv,"reference mass/inverse not the canonical reference-density sum");
        need(xyz(node.positionAndMass)==authored.objects[0].femNodes[n],"initial current coordinates lost");
        need(xyz(node.restAndFixed)==authored.objects[0].femReferenceNodes[n],"reference coordinates lost");
        nodalMass+=m;for(unsigned j=0;j<3;++j){referenceMoment[j]+=m*xyz(node.restAndFixed)[j];currentMoment[j]+=m*xyz(node.positionAndMass)[j];}}
    need(std::abs(nodalMass-elementMass)<1e-7*elementMass,"mechanical reference mass not conserved within FP32 rounding");
    Vec referenceCenter{},currentCenter{};for(unsigned j=0;j<3;++j){referenceCenter[j]=referenceMoment[j]/nodalMass;currentCenter[j]=currentMoment[j]/nodalMass;
        need(xyz(w.adaptive[0].referenceCenter)[j]==float(referenceCenter[j]),"reference COM uses loaded positions");
        need(xyz(w.adaptive[0].centerAndRadius)[j]==float(currentCenter[j]),"initial COM uses unloaded reference positions");}
    Matrix nodeInertia{},elementInertia{};
    for(unsigned n=0;n<5;++n){const Vec r=sub(xyz(w.fem.nodes[n].positionAndMass),currentCenter);const auto xx=outer(r,r);for(unsigned i=0;i<9;++i)nodeInertia[i]+=float(mass[n])*((i%4==0?dot(r,r):0)-xx[i]);}
    for(const auto& t:w.fem.tetrahedra){const double quarter=.25*double(w.materials[t.identity.x].bulk.x)*double(t.inverseRestRow0.w);
        for(unsigned n:{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w}){const Vec r=sub(authored.objects[0].femNodes[n],currentCenter);const auto xx=outer(r,r);for(unsigned i=0;i<9;++i)elementInertia[i]+=quarter*((i%4==0?dot(r,r):0)-xx[i]);}}
    double inertiaError=0,inertiaScale=0;for(unsigned i=0;i<9;++i){inertiaError=std::max(inertiaError,std::abs(nodeInertia[i]-elementInertia[i]));inertiaScale=std::max(inertiaScale,std::abs(elementInertia[i]));}
    need(inertiaError<1e-7*inertiaScale,"initial lumped inertia not reference-mass weighted");++cpuCheckCount;
    const Matrix expectedF{1.015625,.0078125,0,0,1,0,0,0,1};const auto input=manufactured(w);
    for(unsigned ti=0;ti<2;++ti)need(gradient(w,input,0,ti)==expectedF,"known affine initial F is not initial edges times reference inverse");++cpuCheckCount;
    auto changed=authored;changed.objects[0].femNodes=changed.objects[0].femReferenceNodes;const auto relaxed=cook(changed);
    for(unsigned n=0;n<5;++n)need(relaxed.fem.nodes[n].positionAndMass.w==w.fem.nodes[n].positionAndMass.w&&relaxed.fem.nodes[n].velocityAndInverseMass.w==w.fem.nodes[n].velocityAndInverseMass.w,"current deformation changed reference mass");
    need(same(relaxed.fem.tetrahedra,w.fem.tetrahedra)&&relaxed.fingerprint!=w.fingerprint,"initial state/reference fingerprint ownership invalid");++cpuCheckCount;
    changed=authored;changed.objects[0].femReferenceSourceIdentity[0]^=1;need(cook(changed).fingerprint!=w.fingerprint,"reference source identity not fingerprinted");++cpuCheckCount;
    changed=authored;changed.objects[0].femReferenceNodes[3][2]*=1.125;need(cook(changed).fingerprint!=w.fingerprint,"reference geometry not fingerprinted");++cpuCheckCount;
    auto legacy=authored;legacy.objects[0].femReferenceNodes.clear();legacy.objects[0].femReferenceSourceIdentity={};const auto old=cook(legacy);
    need(!(old.objects[0].flags&NM_OBJECT_FEM_REFERENCE_CONFIGURATION),"empty reference changed legacy flag");
    for(const auto& n:old.fem.nodes)need(xyz(n.positionAndMass)==xyz(n.restAndFixed),"empty reference changed legacy rest initialization");
    for(const auto& t:old.fem.tetrahedra)need(t.inverseRestRow0.w>float(volume),"legacy reference no longer uses initial geometry");++cpuCheckCount;
    const auto reject=[&](const char* label,const std::function<void(WorldSource&)>& change){auto s=authored;change(s);const auto c=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});need(!c.succeeded(),std::string("reference source accepted ")+label);++cpuCheckCount;};
    reject("missing identity",[](auto& s){s.objects[0].femReferenceSourceIdentity={};});
    reject("orphan identity",[](auto& s){s.objects[0].femReferenceNodes.clear();});
    reject("short reference",[](auto& s){s.objects[0].femReferenceNodes.pop_back();});
    reject("long reference",[](auto& s){s.objects[0].femReferenceNodes.push_back({1,1,1});});
    reject("nonfinite reference",[](auto& s){s.objects[0].femReferenceNodes[0][0]=std::numeric_limits<double>::quiet_NaN();});
    reject("FP32 overflowing reference",[](auto& s){s.objects[0].femReferenceNodes[0][0]=1e100;});
    reject("collapsed reference",[](auto& s){s.objects[0].femReferenceNodes[3]=s.objects[0].femReferenceNodes[0];});
    reject("inverted reference",[](auto& s){s.objects[0].femReferenceNodes[3][2]*=-1;});
    reject("collapsed initial",[](auto& s){s.objects[0].femNodes[3]=s.objects[0].femNodes[0];});
    reject("inverted initial",[](auto& s){s.objects[0].femNodes[3][2]=s.objects[0].femNodes[4][2];});
    reject("nonfinite initial",[](auto& s){s.objects[0].femNodes[0][0]=std::numeric_limits<double>::infinity();});
    reject("out of material domain",[](auto& s){for(auto& x:s.objects[0].femNodes)for(double& a:x)a*=100;});
    reject("automatic",[](auto& s){s.objects[0].automaticRepresentation=true;});
    reject("MPM",[](auto& s){s.objects[0].representation=Representation::mpm;});
    reject("mixed",[](auto& s){s.objects[0].mixedFEM=true;});
    reject("multiphysics",[](auto& s){s.objects[0].multiphysics.enabled=true;});
    reject("field boundary",[](auto& s){FieldBoundarySource b;b.node=0;b.flags=NM_FIELD_DIRICHLET_TEMPERATURE;b.stableIdentifier=1;s.objects[0].fieldBoundaries={b};});
    reject("identification",[](auto& s){s.objects[0].identifiable=true;});
    reject("identified selected material",[](auto& s){s.materials[0].parameters[0].identifiable=true;});
    reject("missing density",[](auto& s){set(s.materials[0],"density",0);});
    reject("adaptive",[](auto& s){s.objects[0].adaptive=true;});
    reject("mutable topology",[](auto& s){s.objects[0].mutationPolicy.enabled=true;});
    reject("mutation command",[](auto& s){MutationCommandSource c;c.stableIdentifier=1;c.target=0;s.objects[0].mutationCommands={c};});
    reject("other mutable object",[](auto& s){auto other=s.objects[0];other.name="mutable_other";other.femReferenceNodes.clear();other.femReferenceSourceIdentity={};other.mutationPolicy.enabled=true;s.objects.push_back(other);});
    const auto rejectCooked=[&](const char* label,const std::function<void(CompiledWorld&)>& change){auto c=w;change(c);c.fingerprint=compiledWorldFingerprint(c);std::string error;need(!validateCompiledWorldLayout(c,&error),std::string("resealed cooked reference accepted ")+label);++cpuCheckCount;};
    rejectCooked("reference identity",[](auto& c){c.objects[0].referenceSourceIdentity[0]=c.objects[0].referenceSourceIdentity[1]=c.objects[0].referenceSourceIdentity[2]=c.objects[0].referenceSourceIdentity[3]=0;});
    rejectCooked("reference node",[](auto& c){c.fem.nodes[3].restAndFixed.x=.001f;});
    rejectCooked("inverse reference",[](auto& c){c.fem.tetrahedra[0].inverseRestRow0.x=std::nextafter(c.fem.tetrahedra[0].inverseRestRow0.x,0.f);});
    rejectCooked("reference volume",[](auto& c){c.fem.tetrahedra[0].inverseRestRow0.w*=2;});
    rejectCooked("reference center",[](auto& c){c.adaptive[0].referenceCenter.x=.5f;});
    rejectCooked("initial center",[](auto& c){c.adaptive[0].centerAndRadius.x=.5f;});
    rejectCooked("object reference mass",[](auto& c){c.adaptive[0].massAndError.x*=2;});
    rejectCooked("reference row padding",[](auto& c){c.fem.tetrahedra[0].inverseRestRow1.w=1;});
    rejectCooked("inverted initial",[](auto& c){c.fem.nodes[3].positionAndMass.z=c.fem.nodes[4].positionAndMass.z;});
    rejectCooked("nodal mass",[](auto& c){c.fem.nodes[3].positionAndMass.w=std::nextafter(c.fem.nodes[3].positionAndMass.w,1.f);});
    rejectCooked("noncanonical fixed tag",[](auto& c){c.fem.nodes[3].restAndFixed.w=.75f;});
    const auto path=std::filesystem::temp_directory_path()/(std::string("numi-reference-cpu-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
    CompileResult package;package.world=w;std::string error;need(writePackage(package,path,&error),error);CompiledWorld loaded;
    const bool ok=readPackage(path,loaded,nullptr,&error);std::error_code removed;std::filesystem::remove(path,removed);
    need(ok&&!removed&&loaded.fingerprint==w.fingerprint&&same(loaded.objects,w.objects)&&same(loaded.fem.nodes,w.fem.nodes)&&same(loaded.fem.tetrahedra,w.fem.tetrahedra),"reference package roundtrip "+error);++cpuCheckCount;
    std::cout<<"reference_compiler_checks="<<cpuCheckCount<<" reference_volume_m3="<<volume<<" reference_mass_kg="<<elementMass<<" initial_J="<<ref::determinant(expectedF)<<" inertia_relative_error="<<inertiaError/inertiaScale<<'\n';
}
void generalReferenceChecks(const WorldSource& authored){
    auto s=authored;auto& o=s.objects[0];
    // Rotation/shear with every matrix entry nonzero, at a translated
    // reference origin. This exercises the general adjugate and FP32 cook.
    const Matrix a{1,.25,.0625,-.125,1.125,.1875,.125,-.0625,.875};
    const Matrix f{1.015625,.0078125,0,0,1,0,0,0,1};
    const Vec origin{.3125,-.4375,.625},translation{.03125,-.015625,.0078125};
    for(unsigned n=0;n<o.femReferenceNodes.size();++n){auto x=transform(a,o.femReferenceNodes[n]);for(unsigned j=0;j<3;++j)x[j]+=origin[j];
        auto p=transform(f,x);for(unsigned j=0;j<3;++j)p[j]+=translation[j];o.femReferenceNodes[n]=x;o.femNodes[n]=p;}
    const auto w=cook(s);const auto input=manufactured(w);double inverseError=0,deformationError=0;
    for(unsigned ti=0;ti<2;++ti){const auto& t=w.fem.tetrahedra[ti];const auto x0=xyz(w.fem.nodes[t.nodes.x].restAndFixed);
        const Matrix edges=columns(sub(xyz(w.fem.nodes[t.nodes.y].restAndFixed),x0),sub(xyz(w.fem.nodes[t.nodes.z].restAndFixed),x0),sub(xyz(w.fem.nodes[t.nodes.w].restAndFixed),x0));
        const auto independentInverse=ref::transpose(ref::inverseTranspose(edges)),stored=inverseRest(t),actualF=gradient(w,input,0,ti);
        for(unsigned j=0;j<9;++j){inverseError=std::max(inverseError,std::abs(stored[j]-independentInverse[j])/std::max(1.,std::abs(independentInverse[j])));deformationError=std::max(deformationError,std::abs(actualF[j]-f[j]));}}
    need(inverseError<1e-7&&deformationError<1e-6,"general affine reference did not preserve independent inverse/F");++cpuCheckCount;
    const auto path=std::filesystem::temp_directory_path()/(std::string("numi-reference-general-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
    CompileResult package;package.world=w;std::string error;need(writePackage(package,path,&error),error);CompiledWorld loaded;
    const bool ok=readPackage(path,loaded,nullptr,&error);std::error_code removed;std::filesystem::remove(path,removed);
    need(ok&&!removed&&loaded.fingerprint==w.fingerprint&&same(loaded.fem.tetrahedra,w.fem.tetrahedra)&&same(loaded.fem.nodes,w.fem.nodes),"non-axis-aligned reference package rejected "+error);++cpuCheckCount;
    std::cout<<"general_reference_compiler_checks="<<cpuCheckCount<<" inverse_relative_error="<<inverseError<<" affine_F_max_error="<<deformationError<<" origin_translated=true\n";
}
WorldSource boundarySource(const WorldSource& authored,float stretch){
    auto s=authored;auto& o=s.objects[0];o.femMaterialIndices.clear();o.femMaterialSourceIdentity={};o.femMaterialFrameRotations.clear();o.femMaterialFrameSourceIdentity={};
    const float a=.00037f;
    o.femReferenceNodes={{{0,0,0}},{{a,0,0}},{{0,a,0}},{{0,0,a}},{{0,0,-a}}};o.femNodes=o.femReferenceNodes;
    for(auto& x:o.femNodes)x[2]=float(float(x[2])*stretch);
    for(auto& m:s.materials)m.minimumDeterminant=.2;
    return s;
}
void executableBoundaryChecks(const WorldSource& authored){
    const auto good=boundarySource(authored,.201f);const auto w=cook(good);const float a=.00037f,z=a*.2f;
    auto boundary=w;boundary.fem.nodes[3].positionAndMass.z=z;boundary.fem.nodes[4].positionAndMass.z=-z;
    const auto interval=detail::femReferenceDeterminantInterval(boundary.fem.tetrahedra[0],boundary.fem.nodes.data());
    const auto limits=w.materials[0].validity;const double oldRatio=double(z)/a;
    need(oldRatio>limits.x&&interval.nominal<limits.x,"FP32 inverse determinant boundary repro changed");
    need(!interval.strictlyAdmitted(limits)&&interval.possiblyAdmissible(limits),"FP32 rounding interval did not classify ambiguous boundary");++cpuCheckCount;
    const auto rejected=compileWorld(boundarySource(authored,.2f),{.maximumRateExponent=0,.emitSpecializedMetal=false});
    need(!rejected.succeeded()&&diagnostic(rejected.diagnostics).find("executable FP32 determinant interior")!=std::string::npos,"source admitted ratio-valid but executable-ambiguous boundary");++cpuCheckCount;
    boundary.fingerprint=compiledWorldFingerprint(boundary);std::string error;
    need(!validateCompiledWorldLayout(boundary,&error)&&error.find("reference coordinates")!=std::string::npos,"resealed cooked geometry admitted executable-ambiguous boundary");++cpuCheckCount;
    const auto interior=detail::femReferenceDeterminantInterval(w.fem.tetrahedra[0],w.fem.nodes.data());need(interior.strictlyAdmitted(limits),"safe interior was rejected");++cpuCheckCount;
    auto outside=boundary;outside.fem.nodes[3].positionAndMass.z=a*.19f;outside.fem.nodes[4].positionAndMass.z=-a*.19f;
    need(!detail::femReferenceDeterminantInterval(outside.fem.tetrahedra[0],outside.fem.nodes.data()).possiblyAdmissible(limits),"provably outside interval was admitted");++cpuCheckCount;
    auto overflow=w.fem.tetrahedra[0];overflow.inverseRestRow0.x=std::numeric_limits<float>::max();overflow.inverseRestRow1.y=std::numeric_limits<float>::max();
    const auto unsafe=detail::femReferenceDeterminantInterval(overflow,w.fem.nodes.data());need(!unsafe.strictlyAdmitted(limits)&&!unsafe.possiblyAdmissible(limits),"unbounded FP32 intermediate overflow admitted");++cpuCheckCount;
    auto subnormal=w.fem.tetrahedra[0];subnormal.inverseRestRow0.x=std::numeric_limits<float>::denorm_min();
    const auto subnormalInverse=detail::femReferenceDeterminantInterval(subnormal,w.fem.nodes.data());
    need(!subnormalInverse.strictlyAdmitted(limits)&&!subnormalInverse.possiblyAdmissible(limits),"subnormal inverse input admitted despite FTZ ambiguity");++cpuCheckCount;
    auto subnormalNodes=w.fem.nodes;subnormalNodes[0].positionAndMass.x=std::numeric_limits<float>::denorm_min();
    const auto subnormalPosition=detail::femReferenceDeterminantInterval(w.fem.tetrahedra[0],subnormalNodes.data());
    need(!subnormalPosition.strictlyAdmitted(limits)&&!subnormalPosition.possiblyAdmissible(limits),"subnormal current coordinate admitted despite FTZ ambiguity");++cpuCheckCount;
    std::cout<<"reference_executable_boundary_checks="<<cpuCheckCount<<" old_ratio="<<oldRatio<<" host_nominal_FP32_J="<<interval.nominal<<" minimum_J="<<limits.x
        <<" determinant_interval_center="<<interval.center<<" determinant_interval_radius="<<interval.radius<<" source_cooked_boundary=rejected snapshot_boundary=ambiguous_not_certified\n";
}
void boundaryRestoreChecks(Device& d,const WorldSource& authored){
    Run run(d,boundarySource(authored,.201f));const auto initial=run.state();auto boundary=initial;const float a=.00037f,z=a*.2f;
    for(unsigned e=0;e<2;++e){boundary.femNodes[e*5+3].positionAndMass.z=z;boundary.femNodes[e*5+4].positionAndMass.z=-z;}
    run.restore(boundary);need(physicalSame(boundary,run.state()),"rounding-ambiguous boundary restore changed state");
    auto outside=boundary;outside.femNodes[3].positionAndMass.z=a*.19f;outside.femNodes[4].positionAndMass.z=-a*.19f;
    const auto result=run.runtime.restore(outside);need(!result.encoded&&physicalSame(boundary,run.state()),"provably outside reference snapshot accepted or mutated state");
    run.restore(initial);need(physicalSame(initial,run.state()),"safe initial state could not be restored after boundary control");
    std::cout<<"reference_boundary_restore=overlap_accepted_without_strict_GPU_certification provably_outside=denied restoration=unchanged\n";
}
void directChecks(Device& d,const WorldSource& authored,const char* label){
    const auto w=cook(authored);const auto input=manufactured(w);
    const Oracle oracle=[&](unsigned ti,const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
        if(w.fem.tetrahedra[ti].identity.x==0)return ref::framedGuccione(f,h,q);
        auto r=ref::neoHookean(ref::multiply(f,q),ref::multiply(h,q));r.stress=ref::multiply(r.stress,ref::transpose(q));r.tangent=ref::multiply(r.tangent,ref::transpose(q));return r;};
    const auto f=dispatch(d,w,input,false),j=dispatch(d,w,input,true);
    const double fe=compare(w,input,f,false,oracle),je=compare(w,input,j,true,oracle),fd=finiteDifference(d,w,input,j);
    need(fe<3e-4&&je<3e-4&&fd<3e-3,"production reference prestress/tangent failed force="+number(fe)+" tangent="+number(je)+" FD="+number(fd));
    double forceScale=0,net=0;for(const auto& raw:f){Vec sum{};for(const auto& v:forces(raw))for(unsigned i=0;i<3;++i){sum[i]+=v[i];forceScale=std::max(forceScale,std::abs(v[i]));}for(double x:sum)net=std::max(net,std::abs(x));}
    need(forceScale>1e-3&&net<1e-6*forceScale,"prestress is missing or internal element forces violate translation invariance");
    auto relaxed=authored;relaxed.objects[0].femNodes=relaxed.objects[0].femReferenceNodes;const auto rw=cook(relaxed);const auto rf=dispatch(d,rw,manufactured(rw),false);
    double zeroScale=0;for(const auto& raw:rf)for(const auto& v:forces(raw))for(double x:v)zeroScale=std::max(zeroScale,std::abs(x));
    need(zeroScale<1e-4*forceScale,"supplied reference zero-load control has unexpected stress");
    std::cout<<"reference_direct_case="<<label<<" force_relative_error="<<fe<<" tangent_relative_error="<<je<<" finite_difference_relative_error="<<fd<<" initial_prestress_nodal_force_N="<<forceScale<<" internal_force_balance_relative="<<net/forceScale<<" unloaded_force_N="<<zeroScale<<'\n';
}
void integratedChecks(Device& device,const WorldSource& authored,unsigned steps,const char* label){
    Run run(device,authored);const auto initial=run.state();RuntimeStateSnapshot one;double motion=0;
    for(unsigned i=0;i<steps;++i){run.step(i);const auto s=run.state();if(i==0)one=s;paired(s.femNodes,"accepted reference nodes");
        for(unsigned n=0;n<s.femNodes.size();++n){need(s.femNodes[n].positionAndMass.w==initial.femNodes[n].positionAndMass.w&&s.femNodes[n].velocityAndInverseMass.w==initial.femNodes[n].velocityAndInverseMass.w,"reference trajectory changed mechanical mass");
            if(n%5<3)need(xyz(s.femNodes[n].positionAndMass)==xyz(initial.femNodes[n].positionAndMass),"fixed initial loaded node snapped to reference");
            const auto delta=sub(xyz(s.femNodes[n].positionAndMass),xyz(initial.femNodes[n].positionAndMass));motion=std::max(motion,std::sqrt(dot(delta,delta)));}}
    const auto final=run.state();need(motion>1e-7,"initial prestress did not cause mechanical relaxation");run.restore(initial);for(unsigned i=0;i<steps;++i)run.step(i);need(physicalSame(final,run.state()),"reference replay not bitwise");
    const auto before=run.state();run.step(steps,1);const auto rejected=run.state();envSame(before.femNodes,rejected.femNodes,1,"rejected reference nodes");envSame(before.femMaterialState,rejected.femMaterialState,1,"rejected reference state");
    const auto deny=[&](const char* name,const std::function<void(RuntimeStateSnapshot&)>& change){auto c=rejected;change(c);const auto result=run.runtime.restore(c);need(!result.encoded&&physicalSame(rejected,run.state()),std::string("reference snapshot accepted or changed state for ")+name);};
    deny("reference coordinates",[](auto& s){s.femNodes[3].restAndFixed.x=.001f;});
    deny("reference center",[](auto& s){s.adaptive[0].referenceCenter.x=.5f;});
    deny("reference operator",[](auto& s){s.femTopologyTetrahedra[0].inverseRestRow0.x=std::nextafter(s.femTopologyTetrahedra[0].inverseRestRow0.x,0.f);});
    deny("reference volume",[](auto& s){s.femTopologyTetrahedra[0].inverseRestRow0.w*=2;});
    deny("connectivity",[](auto& s){std::swap(s.femTopologyTetrahedra[0].nodes.x,s.femTopologyTetrahedra[0].nodes.y);});
    deny("material owner",[](auto& s){++s.femTopologyTetrahedra[0].identity.x;});
    deny("topology lineage",[](auto& s){s.femTopologyNodes[0].identity.x=s.femTopologyNodes[1].identity.x;});
    deny("topology accounting",[](auto& s){s.topologyStates[0].accounting.x=std::nextafter(s.topologyStates[0].accounting.x,1.f);});
    deny("mass",[](auto& s){s.femNodes[3].positionAndMass.w=std::nextafter(s.femNodes[3].positionAndMass.w,1.f);});
    deny("inverse mass",[](auto& s){s.femNodes[3].velocityAndInverseMass.w=std::nextafter(s.femNodes[3].velocityAndInverseMass.w,0.f);});
    deny("fixed constraint",[](auto& s){s.femNodes[3].restAndFixed.w=1;});
    deny("inverted current geometry",[](auto& s){s.femNodes[3].positionAndMass.z=s.femNodes[4].positionAndMass.z;});
    deny("collapsed current geometry",[](auto& s){const auto p=s.femNodes[0].positionAndMass;s.femNodes[3].positionAndMass.x=p.x;s.femNodes[3].positionAndMass.y=p.y;s.femNodes[3].positionAndMass.z=p.z;});
    deny("nonfinite current geometry",[](auto& s){s.femNodes[3].positionAndMass.x=std::numeric_limits<float>::infinity();});
    deny("fixed density parameter",[](auto& s){s.environmentParameters[0]=std::nextafter(s.environmentParameters[0],0.f);});
    deny("scheduler floor",[](auto& s){++s.schedulers[1].baseExponent;});
    deny("generation",[](auto& s){++s.allocationGeneration;});
    deny("representation",[](auto& s){s.adaptive[0].activeRepresentation=NM_REPRESENTATION_RIGID;});
    if(!authored.objects[0].femMaterialFrameRotations.empty())deny("reference material frame",[](auto& s){s.femTopologyTetrahedra[0].materialFrameRotation={0,0,0,1};});
    run.step(steps+1,-1,1);const auto reset=run.state();envSame(one.femNodes,reset.femNodes,1,"reset to supplied initial loaded nodes");envSame(one.femMaterialState,reset.femMaterialState,1,"reset reference state");
    std::cout<<"reference_runtime_case="<<label<<" accepted_steps="<<steps<<" environments=2 motion_m="<<motion<<" replay=bitwise rollback=isolated reset=loaded_initial_bitwise snapshot_rebinding=denied reference_mass=unchanged\n";
}
} // namespace
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3||(argc==4&&std::string(argv[3])=="--cpu-only"),"usage: fem-reference-check GUCCIONE.nmatter NEO.nmatter [--cpu-only]");
    auto a=parseMatterFile(argv[1]),b=parseMatterFile(argv[2]);need(a.succeeded()&&b.succeeded(),diagnostic(a.diagnostics)+diagnostic(b.diagnostics));
    set(a.material,"density",900);set(b.material,"density",1200); // synthetic fixture only
    const auto simple=source(a.material,b.material),combined=source(a.material,b.material,true);
    std::cout<<std::setprecision(17)<<"reference_abi="<<NM_MATTER_ABI_VERSION<<'\n';cpuChecks(simple);cpuChecks(combined);generalReferenceChecks(simple);generalReferenceChecks(combined);executableBoundaryChecks(simple);
    if(argc==4){std::cout<<"fem_reference_compiler_check=pass unloaded_reference_recovered=false anatomical_wall_qualified=false\n";return 0;}
    Device device;std::cout<<"reference_device="<<[device.device name].UTF8String<<'\n';
    boundaryRestoreChecks(device,simple);directChecks(device,simple,"reference_only");directChecks(device,combined,"reference_regional_frames");
    integratedChecks(device,simple,16,"reference_only");integratedChecks(device,combined,16,"reference_regional_frames");
    std::cout<<"fem_reference_check=pass unloaded_reference_recovered=false anatomical_wall_qualified=false source_density_supplied=false active_cardiac_source_reproduced=false\n";return 0;
}catch(const std::exception& e){std::cerr<<"fem reference check failed: "<<e.what()<<'\n';return 1;}}}
