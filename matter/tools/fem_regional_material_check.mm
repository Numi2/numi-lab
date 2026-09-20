#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include "cardiac_material_reference.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

// Synthetic conforming two-tet REGIONAL constitutive/transaction fixture.
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
WorldSource source(const MaterialProgram& first,const MaterialProgram& second,bool framed=true){
    WorldSource s;s.environmentCount=2;s.gravity={0,0,0};s.frameTimestep=1e-4;
    // The default is a deliberately different third constitutive owner. It
    // owns the common external interface; no cell is assigned to it.
    auto fallback=second;fallback.name="synthetic_unselected_interface_owner";set(fallback,"density",1500);
    s.materials={fallback,first,second};
    s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=12;s.mixedSolver.fgmresIterations=64;
    ObjectSource o;o.name="synthetic_conforming_regional_materials";o.representation=Representation::fem;o.mixedFEM=false;
    o.deformableContact=false;o.deformableSelfContact=false;o.characteristicLength=.015625;
    constexpr double l=.015625;
    o.femNodes={{{0,0,0}},{{l,0,0}},{{0,l,0}},{{0,0,l}},{{0,0,-l}}};
    o.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};o.femFixedNodes={0,1,2};o.femInitialVelocity={.025,.017,.013};
    o.femMaterialIndices={1,2};o.femMaterialSourceIdentity={0x9e373d3bb8876e8full,0x73b6ca446bf2ca80ull,0xd52a057214116d32ull,0xa92e27a449c57444ull};
    if(framed){o.femMaterialFrameRotations={{{0,0,std::sin(.305),std::cos(.305)}},{{0,std::sin(-.245),0,std::cos(-.245)}}};
        o.femMaterialFrameSourceIdentity={0xf731c96c934e625eull,0x836db2ec25021611ull,0x71a72c65dc7b2f5aull,0x36ab5d1b57e2bb96ull};}
    s.objects={o};return s;
}
CompiledWorld cook(const WorldSource& s){auto c=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});need(c.succeeded(),"regional fixture compile: "+diagnostic(c.diagnostics));return std::move(c.world);}

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
    const Matrix f{1.13,.12,.04,-.025,.94,.06,.015,-.035,1.025};
    const Matrix h{.19,-.13,.07,.11,-.17,.04,-.05,.09,.14};
    const Matrix rate{.3,-.2,.1,.12,.22,-.18,-.1,.08,.25};
    for(unsigned env=0;env<2;++env)for(unsigned n=0;n<w.dispatch.femNodeCount;++n){const unsigned i=env*w.dispatch.femNodeCount+n;
        const Vec x=xyz(w.fem.nodes[n].positionAndMass),p=transform(f,x),r=transform(rate,x),d=transform(h,x);
        v.nodes[i].positionAndMass.x=p[0];v.nodes[i].positionAndMass.y=p[1];v.nodes[i].positionAndMass.z=p[2];
        v.nodes[i].velocityAndInverseMass.x=r[0];v.nodes[i].velocityAndInverseMass.y=r[1];v.nodes[i].velocityAndInverseMass.z=r[2];
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
double difference(const std::vector<NMFEMElementVectorGPU>& a,const std::vector<NMFEMElementVectorGPU>& b){double error=0,scale=0;need(a.size()==b.size(),"element output arity");for(unsigned e=0;e<a.size();++e){const auto x=forces(a[e]),y=forces(b[e]);for(unsigned n=0;n<4;++n)for(unsigned i=0;i<3;++i){error=std::max(error,std::abs(x[n][i]-y[n][i]));scale=std::max(scale,std::abs(y[n][i]));}}return error/std::max(1e-10,scale);}
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
unsigned cpuCheckCount=0;
void cpuChecks(const WorldSource& authored){
    const auto world=cook(authored);need(world.fem.nodes.size()==5&&world.fem.tetrahedra.size()==2,"conforming fixture duplicated shared nodes");
    need((world.objects[0].flags&NM_OBJECT_FEM_REGIONAL_MATERIAL)!=0,"regional ownership flag absent");
    std::vector<double> mass(5);double elementMass=0;
    for(unsigned ti=0;ti<2;++ti){const auto& t=world.fem.tetrahedra[ti];need(t.identity.x==ti+1&&t.identity.y==0,"tet material/object ownership changed");
        const double m=double(world.materials[ti+1].bulk.x)*double(t.inverseRestRow0.w);elementMass+=m;
        for(unsigned n:{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w})mass[n]+=.25*m;}
    Vec moment{};double nodalMass=0;std::vector<double> roundedMass(5);
    for(unsigned n=0;n<5;++n){const auto& node=world.fem.nodes[n];const float expected=float(mass[n]);
        need(std::bit_cast<unsigned>(node.positionAndMass.w)==std::bit_cast<unsigned>(expected),"regional nodal mass is not one rounded FP64 incident sum at node="+std::to_string(n));
        const float inv=n<3?0.f:float(1/mass[n]);need(std::bit_cast<unsigned>(node.velocityAndInverseMass.w)==std::bit_cast<unsigned>(inv),"canonical inverse mass mismatch");
        roundedMass[n]=expected;nodalMass+=expected;for(unsigned j=0;j<3;++j)moment[j]+=expected*authored.objects[0].femNodes[n][j];}
    need(std::abs(nodalMass-elementMass)<1e-7*elementMass,"regional total mass not conservative within FP32 rounding");
    Vec center=moment;for(double& x:center)x/=nodalMass;
    Matrix nodalInertia{},elementLumpedInertia{};
    for(unsigned n=0;n<5;++n){const Vec x=sub(xyz(world.fem.nodes[n].positionAndMass),center);const auto xx=outer(x,x);const double r2=dot(x,x);
        for(unsigned i=0;i<9;++i)nodalInertia[i]+=roundedMass[n]*((i%4==0?r2:0)-xx[i]);}
    // Compare the executable lumped inertia to independent element-node
    // quadrature, not continuum tetrahedral inertia (which is different).
    for(const auto& t:world.fem.tetrahedra){const double quarter=.25*double(world.materials[t.identity.x].bulk.x)*double(t.inverseRestRow0.w);
        for(unsigned n:{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w}){const Vec x=sub(authored.objects[0].femNodes[n],center);const auto xx=outer(x,x);const double r2=dot(x,x);
            for(unsigned i=0;i<9;++i)elementLumpedInertia[i]+=quarter*((i%4==0?r2:0)-xx[i]);}}
    double inertiaError=0,inertiaScale=0;for(unsigned i=0;i<9;++i){inertiaError=std::max(inertiaError,std::abs(nodalInertia[i]-elementLumpedInertia[i]));inertiaScale=std::max(inertiaScale,std::abs(elementLumpedInertia[i]));}
    need(inertiaError<1e-7*inertiaScale,"regional lumped inertia disagrees with element density ownership");
    need(world.adaptive[0].massAndError.x==float(nodalMass),"object mass ignores regional cells");
    for(unsigned j=0;j<3;++j)need(xyz(world.adaptive[0].referenceCenter)[j]==float(center[j]),"object center ignores regional masses");
    ++cpuCheckCount;
    auto legacy=authored;legacy.objects[0].femMaterialIndices.clear();legacy.objects[0].femMaterialSourceIdentity={};const auto homogeneous=cook(legacy);
    for(const auto& t:homogeneous.fem.tetrahedra)need(t.identity.x==0,"empty map changed legacy material");
    need((homogeneous.objects[0].flags&NM_OBJECT_FEM_REGIONAL_MATERIAL)==0,"empty map enabled regional ownership");++cpuCheckCount;
    auto changed=authored;std::swap(changed.objects[0].femMaterialIndices[0],changed.objects[0].femMaterialIndices[1]);need(cook(changed).fingerprint!=world.fingerprint,"material assignment not fingerprinted");
    changed=authored;changed.objects[0].femMaterialSourceIdentity[0]^=1;need(cook(changed).fingerprint!=world.fingerprint,"regional source identity not fingerprinted");cpuCheckCount+=2;
    // An unselected object material cannot set cadence for a much faster
    // regional law. The authored candidate gives dt/stable in (16,32), hence
    // exponent 5 independently of the compiler's rate-selection helper.
    auto multirate=authored;multirate.objects[0].femMaterialFrameRotations.clear();multirate.objects[0].femMaterialFrameSourceIdentity={};
    set(multirate.materials[2],"density",9);
    bool foundKappa=false;for(auto& p:multirate.materials[2].parameters)if(p.name=="kappa"){p.upper=16e6;foundKappa=true;}
    need(foundKappa,"source kappa parameter missing from scheduler fixture");
    const double cflRatio=multirate.frameTimestep*std::sqrt(16e6/9.)/(.35*multirate.objects[0].characteristicLength);
    need(cflRatio>16&&cflRatio<32,"scheduler fixture no longer has independently bounded exponent 5");
    const auto selectedRate=compileWorld(multirate,{.maximumRateExponent=8,.emitSpecializedMetal=false});
    need(selectedRate.succeeded(),"selected rate fixture: "+diagnostic(selectedRate.diagnostics));
    multirate.objects[0].femMaterialIndices.clear();multirate.objects[0].femMaterialSourceIdentity={};
    const auto defaultRate=compileWorld(multirate,{.maximumRateExponent=8,.emitSpecializedMetal=false});
    need(defaultRate.succeeded(),"default rate fixture: "+diagnostic(defaultRate.diagnostics));
    const auto& rate=selectedRate.world.schedulers[0];need(rate.baseExponent==5&&rate.activeExponent==5&&rate.requestedExponent==5&&defaultRate.world.schedulers[0].baseExponent==0,
        "regional scheduler did not select the fastest selected material");++cpuCheckCount;
    const auto reject=[&](const std::string& label,const std::function<void(WorldSource&)>& mutation){auto s=authored;s.objects[0].femMaterialFrameRotations.clear();s.objects[0].femMaterialFrameSourceIdentity={};mutation(s);const auto r=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});need(!r.succeeded(),"source admission accepted "+label);++cpuCheckCount;};
    reject("short map",[](auto& s){s.objects[0].femMaterialIndices.pop_back();});
    reject("long map",[](auto& s){s.objects[0].femMaterialIndices.push_back(1);});
    reject("out-of-range material",[](auto& s){s.objects[0].femMaterialIndices[0]=NM_INVALID_INDEX;});
    reject("missing provenance",[](auto& s){s.objects[0].femMaterialSourceIdentity={};});
    reject("orphan provenance",[](auto& s){s.objects[0].femMaterialIndices.clear();});
    reject("automatic representation",[](auto& s){s.objects[0].automaticRepresentation=true;});
    reject("MPM representation",[](auto& s){s.objects[0].representation=Representation::mpm;});
    reject("mixed FEM",[](auto& s){s.objects[0].mixedFEM=true;});
    reject("multiphysics",[](auto& s){s.objects[0].multiphysics.enabled=true;});
    reject("field boundaries",[](auto& s){FieldBoundarySource b;b.node=0;b.flags=NM_FIELD_DIRICHLET_TEMPERATURE;b.stableIdentifier=1;s.objects[0].fieldBoundaries={b};});
    reject("identification",[](auto& s){s.objects[0].identifiable=true;});
    reject("adaptive",[](auto& s){s.objects[0].adaptive=true;});
    reject("mutable topology",[](auto& s){s.objects[0].mutationPolicy.enabled=true;});
    reject("mutation command",[](auto& s){MutationCommandSource c;c.stableIdentifier=1;c.target=0;s.objects[0].mutationCommands={c};});
    reject("selected material lacks FEM",[](auto& s){s.materials[2].supportedRepresentations={Representation::mpm};});
    reject("selected identifiable density",[](auto& s){for(auto& p:s.materials[2].parameters)if(p.name=="density")p.identifiable=true;});
    reject("selected identifiable stiffness",[](auto& s){for(auto& p:s.materials[1].parameters)if(p.name=="kappa")p.identifiable=true;});
    reject("default identifiable parameter",[](auto& s){s.materials[0].parameters[0].identifiable=true;});
    reject("selected zero density",[](auto& s){set(s.materials[2],"density",0);});
    reject("selected nonfinite density",[](auto& s){set(s.materials[2],"density",std::numeric_limits<double>::infinity());});
    reject("selected FP32-underflow density",[](auto& s){set(s.materials[2],"density",1e-100);});
    for(unsigned field=0;field<4;++field)reject("nonuniform interface field "+std::to_string(field),[field](auto& s){
        auto& m=s.materials[2];double* values[]={&m.staticFriction,&m.dynamicFriction,&m.restitution,&m.adhesion};*values[field]+=.01;});
    reject("same FP32 but different source interface",[](auto& s){s.materials[2].staticFriction=std::nextafter(s.materials[0].staticFriction,1.);});
    const auto rejectCooked=[&](const std::string& label,const std::function<void(CompiledWorld&)>& mutation){auto c=world;mutation(c);c.fingerprint=compiledWorldFingerprint(c);std::string error;need(!validateCompiledWorldLayout(c,&error),"resealed cooked admission accepted "+label);++cpuCheckCount;};
    rejectCooked("material swap with stale mass",[](auto& w){w.fem.tetrahedra[0].identity.x=2;});
    rejectCooked("fractional fixed tag",[](auto& w){w.fem.nodes[3].restAndFixed.w=.75f;});
    rejectCooked("nodal mass tamper",[](auto& w){w.fem.nodes[0].positionAndMass.w=std::nextafter(w.fem.nodes[0].positionAndMass.w,1.f);});
    rejectCooked("nodal inverse mass tamper",[](auto& w){w.fem.nodes[3].velocityAndInverseMass.w=std::nextafter(w.fem.nodes[3].velocityAndInverseMass.w,0.f);});
    rejectCooked("nonuniform cooked interface",[](auto& w){w.materials[2].interfaceResponse.x=std::nextafter(w.materials[2].interfaceResponse.x,1.f);});
    const auto addIdentification=[](CompiledWorld& w,unsigned material){
        // The checked source files both put density first. This otherwise
        // valid posterior is accepted by the homogeneous control below.
        const auto offset=w.materials[material].parameterOffset;const auto p=w.parameters[offset].valueAndBounds;
        NMIdentificationDistributionGPU d{};d.identity={material,0,offset,0};d.momentsAndBounds={p.x,1,p.y,p.z};d.update={.2f,1,1e-5f,0};
        w.identification.push_back(d);w.dispatch.flags|=NM_MATTER_IDENTIFICATION;
    };
    need(authored.materials[0].parameters[0].name=="density"&&authored.materials[2].parameters[0].name=="density","density parameter order changed");
    auto posteriorControl=homogeneous;addIdentification(posteriorControl,2);posteriorControl.fingerprint=compiledWorldFingerprint(posteriorControl);
    std::string posteriorError;need(validateCompiledWorldLayout(posteriorControl,&posteriorError),"identification negative control itself invalid: "+posteriorError);++cpuCheckCount;
    rejectCooked("selected material identification distribution",[&](auto& w){addIdentification(w,2);});
    rejectCooked("default material identification distribution",[&](auto& w){addIdentification(w,0);});
    const auto path=std::filesystem::temp_directory_path()/(std::string("numi-regional-cpu-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
    CompileResult package;package.world=world;std::string error;need(writePackage(package,path,&error),error);CompiledWorld loaded;
    const bool ok=readPackage(path,loaded,nullptr,&error);std::error_code removed;std::filesystem::remove(path,removed);
    need(ok&&!removed&&loaded.fingerprint==world.fingerprint&&same(loaded.objects,world.objects)&&same(loaded.fem.tetrahedra,world.fem.tetrahedra)&&same(loaded.fem.nodes,world.fem.nodes),"regional package roundtrip "+error);++cpuCheckCount;
    std::cout<<"regional_compiler_checks="<<cpuCheckCount<<" shared_nodes=3 element_mass_kg="<<elementMass<<" nodal_mass_kg="<<nodalMass
        <<" selected_material_rate_exponent="<<rate.baseExponent<<" default_material_rate_exponent="<<defaultRate.world.schedulers[0].baseExponent<<" lumped_inertia_relative_error="<<inertiaError/inertiaScale<<" source_density_supplied=false\n";
}

std::pair<MaterialProgram,MaterialProgram> stateMaterials(){
    const auto a=parseMatter(R"(material synthetic_regional_state_one {
        parameter density : kg/m^3 = 900 in [0,2000]; parameter k : Pa = 1000; parameter eta : Pa*s = 20;
        state history : one = 0.23 transfer average;
        model generic; energy = 0.5*k*pow(F(0,0)-next(history),2);
        dissipation = 0.5*eta*pow(Fdot(0,2),2);
        update history = history + dt()*Fdot(0,1); valid = J(); supports fem;
    })");
    const auto b=parseMatter(R"(material synthetic_regional_state_two {
        parameter density : kg/m^3 = 1200 in [0,2000]; parameter eta : Pa*s = 31; parameter k : Pa = 1700;
        state memory : one = 0.41 transfer average; state second : one = 0.17 transfer average;
        model generic; energy = 0.5*k*pow(F(1,1)-next(memory),2) + 0.25*k*pow(F(2,2)-next(second),2);
        dissipation = 0.5*eta*pow(Fdot(2,0),2);
        update memory = memory + 2*dt()*Fdot(1,2); update second = second + dt()*Fdot(2,1);
        valid = J(); supports fem;
    })");
    need(a.succeeded()&&b.succeeded(),"synthetic state programs: "+diagnostic(a.diagnostics)+diagnostic(b.diagnostics));return {a.material,b.material};
}
void directChecks(Device& device,const WorldSource& authored){
    const auto world=cook(authored);const auto input=manufactured(world);
    const Oracle sourceOracle=[](unsigned ti,const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
        if(ti==0)return ref::framedGuccione(f,h,q);
        auto r=ref::neoHookean(ref::multiply(f,q),ref::multiply(h,q));r.stress=ref::multiply(r.stress,ref::transpose(q));r.tangent=ref::multiply(r.tangent,ref::transpose(q));return r;};
    const auto force=dispatch(device,world,input,false),tangent=dispatch(device,world,input,true);
    const double fe=compare(world,input,force,false,sourceOracle),te=compare(world,input,tangent,true,sourceOracle),fd=finiteDifference(device,world,input,tangent);
    need(fe<3e-4&&te<3e-4&&fd<3e-3,"regional source oracle mismatch force="+number(fe)+" tangent="+number(te)+" fd="+number(fd));
    paired(force,"regional element force");paired(tangent,"regional element tangent");
    auto fallback=authored;fallback.objects[0].femMaterialIndices={0,0};const auto control=cook(fallback);
    const double consequence=difference(force,dispatch(device,control,input,false));need(consequence>1e-3,"regional fixture cannot distinguish homogeneous fallback");
    const auto [a,b]=stateMaterials();const auto sw=cook(source(a,b));const auto si=manufactured(sw);const double dt=sw.dispatch.gravityAndTimestep.w;
    need(sw.dispatch.materialStateStride>=2&&sw.materials[1].stateCount==1&&sw.materials[2].stateCount==2,"heterogeneous state layout not exercised");
    const Oracle stateOracle=[dt](unsigned ti,const Matrix& f,const Matrix& h,const Matrix& rate,const Matrix& q){
        const Matrix fl=ref::multiply(f,q),hl=ref::multiply(h,q),rl=ref::multiply(rate,q);Matrix p{},dp{};
        if(ti==0){p[0]=1000*(fl[0]-double(float(.23))-dt*rl[1]);p[2]=20*rl[2];dp[0]=1000*(hl[0]-hl[1]);dp[2]=20*hl[2]/dt;}
        else{p[4]=1700*(fl[4]-double(float(.41))-2*dt*rl[5]);p[8]=850*(fl[8]-double(float(.17))-dt*rl[7]);p[6]=31*rl[6];dp[4]=1700*(hl[4]-2*hl[5]);dp[8]=850*(hl[8]-hl[7]);dp[6]=31*hl[6]/dt;}
        ref::Result r;r.stress=ref::multiply(p,ref::transpose(q));r.tangent=ref::multiply(dp,ref::transpose(q));return r;};
    const auto sf=dispatch(device,sw,si,false),st=dispatch(device,sw,si,true);
    const double sfe=compare(sw,si,sf,false,stateOracle),ste=compare(sw,si,st,true,stateOracle),sfd=finiteDifference(device,sw,si,st);
    need(sfe<3e-4&&ste<3e-3&&sfd<4e-3,"regional state/rate oracle mismatch force="+number(sfe)+" tangent="+number(ste)+" fd="+number(sfd));
    std::cout<<"regional_source_force_error="<<fe<<" regional_source_tangent_error="<<te<<" regional_source_fd_error="<<fd<<" homogeneous_fallback_consequence="<<consequence
        <<" regional_state_force_error="<<sfe<<" regional_state_tangent_error="<<ste<<" regional_state_fd_error="<<sfd<<" heterogeneous_state_stride="<<sw.dispatch.materialStateStride<<'\n';
}
void learnedChecks(Device& device){
    auto parsed=parseMatter(R"(material synthetic_regional_icnn { parameter density : kg/m^3 = 900 in [0,2000]; parameter scale : Pa = 1;
        model generic; energy = 0*scale; valid = J(); supports fem; })");need(parsed.succeeded(),diagnostic(parsed.diagnostics));
    std::array<MaterialProgram,2> material{parsed.material,parsed.material};
    const std::array<Vec,2> fibres{{{.6,.8,0},{0,.6,.8}}};
    const std::array<double,2> weight{.75,1.125},bias{.125,-.0625},growth{.03125,.0625};
    for(unsigned i=0;i<2;++i){material[i].name+="_"+std::to_string(i);set(material[i],"density",i?1200:900);material[i].hint=ConstitutiveHint::polyconvexICNN;material[i].mixed.fibreDirection=fibres[i];
        LearnedMaterialSource network;network.invariantCount=5;network.softplusBeta=2.5f;network.determinantFloor=.05f;network.growthCoefficient=growth[i];
        LearnedLayerSource layer;layer.inputWidth=5;layer.outputWidth=1;layer.inputWeights={0,0,0,0,float(weight[i])};layer.biases={float(bias[i])};network.layers={layer};material[i].learned=network;}
    const auto authored=source(material[0],material[1]);const auto world=cook(authored);const auto input=manufactured(world);
    const Oracle oracle=[&](unsigned ti,const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
        Vec a=xyz(world.mixedMaterials[ti+1].fibre);const double len=std::sqrt(dot(a,a));for(double& x:a)x/=len;
        const Vec fibre=transform(q,a),u=transform(f,fibre),du=transform(h,fibre);const double w=weight[ti],g=growth[ti];
        const Matrix it=ref::inverseTranspose(f);const double j=ref::determinant(f),dj=j*ref::contract(it,h);
        const double z=w*(dot(u,u)-1)+bias[ti],sig=1/(1+std::exp(-2.5*z)),dsig=2.5*sig*(1-sig)*2*w*dot(u,du);
        const Matrix cofactor=ref::scale(it,j),dit=ref::scale(ref::multiply(ref::multiply(it,ref::transpose(h)),it),-1);
        const Matrix dc=ref::add(ref::scale(it,dj),ref::scale(dit,j));ref::Result r;
        r.stress=ref::add(ref::scale(outer(u,fibre),2*w*sig),ref::scale(cofactor,g*(1-1/(j*j))));
        r.tangent=ref::add(ref::scale(ref::add(ref::scale(outer(u,fibre),dsig),ref::scale(outer(du,fibre),sig)),2*w),
            ref::scale(ref::add(ref::scale(cofactor,2*dj/(j*j*j)),ref::scale(dc,1-1/(j*j))),g));return r;};
    const auto f=dispatch(device,world,input,false),t=dispatch(device,world,input,true);
    const double fe=compare(world,input,f,false,oracle),te=compare(world,input,t,true,oracle),fd=finiteDifference(device,world,input,t);
    need(fe<3e-4&&te<3e-4&&fd<3e-3,"regional ICNN oracle mismatch force="+number(fe)+" tangent="+number(te)+" fd="+number(fd));
    // Independent frame equivalence: each distinct material has exactly one
    // cell here, so author Q*a as its unframed reference-world fibre.
    auto equivalent=source(material[0],material[1],false);
    for(unsigned ti=0;ti<2;++ti){Vec a=xyz(world.mixedMaterials[ti+1].fibre);const double len=std::sqrt(dot(a,a));for(double& x:a)x/=len;
        equivalent.materials[ti+1].mixed.fibreDirection=transform(frame(world.fem.tetrahedra[ti].materialFrameRotation),a);}
    const auto e=cook(equivalent);const double ef=difference(f,dispatch(device,e,input,false)),et=difference(t,dispatch(device,e,input,true));
    need(ef<3e-4&&et<3e-4,"regional learned frame differs from independently authored world fibre");
    auto wrong=authored;for(unsigned ti=0;ti<2;++ti)wrong.materials[ti+1].mixed.fibreDirection=wrong.materials[0].mixed.fibreDirection;
    const auto wc=cook(wrong);const double consequence=difference(f,dispatch(device,wc,input,false));need(consequence>1e-3,"learned fixture cannot detect object-level fibre fallback");
    std::cout<<"regional_icnn_force_error="<<fe<<" regional_icnn_tangent_error="<<te<<" regional_icnn_fd_error="<<fd<<" regional_icnn_world_fibre_force_error="<<ef
        <<" regional_icnn_world_fibre_tangent_error="<<et<<" wrong_fibre_consequence="<<consequence<<" trained_material_qualified=false\n";
}
std::vector<float> projectState(Device& device,const CompiledWorld& world,const Inputs& input){
    auto pipeline=device.make(@"numi_matter_metal::nm_fem_validate");NMMicrostepGPU micro{};micro.time={world.dispatch.gravityAndTimestep.w,1/world.dispatch.gravityAndTimestep.w,0,0};
    std::vector<float> p;for(unsigned e=0;e<2;++e)for(const auto& x:world.parameters)p.push_back(x.valueAndBounds.x);
    const std::array<id<MTLBuffer>,12> buffers{buffer(device.device,world.objects),buffer(device.device,world.materials),buffer(device.device,world.scalarPrograms),buffer(device.device,world.instructions),
        buffer(device.device,p),buffer(device.device,input.nodes),buffer(device.device,repeat(world.fem.tetrahedra)),buffer(device.device,repeat(world.schedulers)),buffer(device.device,repeat(world.adaptive)),
        buffer(device.device,input.state),buffer(device.device,std::vector<float>(input.state.size(),-123.f)),buffer(device.device,std::vector<NMMatterStatusGPU>(2))};
    auto cb=[device.queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipeline];[enc setBytes:&world.dispatch length:sizeof(world.dispatch) atIndex:0];[enc setBytes:&micro length:sizeof(micro) atIndex:1];
    for(unsigned i=0;i<buffers.size();++i)[enc setBuffer:buffers[i] offset:0 atIndex:i+2];
    [enc dispatchThreads:MTLSizeMake(2*world.dispatch.tetrahedronCount,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
    need(cb.status==MTLCommandBufferStatusCompleted,"regional state projection command failed");const auto* statuses=static_cast<const NMMatterStatusGPU*>(buffers[11].contents);
    for(unsigned e=0;e<2;++e)need(statuses[e].code==NM_STATUS_SUCCESS,"regional state projection failed");const auto* out=static_cast<const float*>(buffers[10].contents);return {out,out+input.state.size()};
}
void projectionChecks(Device& device){
    const auto [a,b]=stateMaterials();const auto world=cook(source(a,b));auto input=manufactured(world);
    // Nonzero padding is a sentinel: projection must copy it without invoking
    // the other material's second update program on this one-state element.
    for(unsigned env=0;env<2;++env)input.state[(env*2)*world.dispatch.materialStateStride+1]=.765625f;
    const auto result=projectState(device,world,input);double error=0;
    for(unsigned env=0;env<2;++env)for(unsigned ti=0;ti<2;++ti){const auto base=(env*2+ti)*world.dispatch.materialStateStride;
        const Matrix rate=ref::multiply(gradient(world,input,env,ti,true),frame(world.fem.tetrahedra[ti].materialFrameRotation));const double dt=world.dispatch.gravityAndTimestep.w;
        const double e0=input.state[base]+dt*(ti?2*rate[5]:rate[1]);error=std::max(error,std::abs(result[base]-e0));
        if(ti)error=std::max(error,std::abs(result[base+1]-(input.state[base+1]+dt*rate[7])));
        else need(result[base+1]==input.state[base+1],"one-state projection wrote other material's state/padding");}
    need(error<1e-7,"regional projected state owner mismatch "+number(error));paired(result,"projected material states");
    std::cout<<"regional_projected_state_absolute_error="<<error<<" unowned_padding=preserved\n";
}
struct Run {
    Device& device;CompiledWorld world;Runtime runtime;id<MTLBuffer> statuses;
    Run(Device& d,const WorldSource& s):device(d),world(cook(s)){
        const auto path=std::filesystem::temp_directory_path()/(std::string("numi-regional-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        CompileResult package;package.world=world;std::string error;need(writePackage(package,path,&error),error);CompiledWorld loaded;
        const bool ok=readPackage(path,loaded,nullptr,&error);std::error_code remove;std::filesystem::remove(path,remove);need(ok&&!remove&&loaded.fingerprint==world.fingerprint,"regional package roundtrip: "+error);world=std::move(loaded);
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
        request.phase=EncodePhase::postCommit;result=runtime.encode(request);need(result.encoded,result.message);[command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"integrated regional command failed");
        const auto after=state();for(unsigned e=0;e<2;++e)if(int(e)!=reject)need(after.statuses[e].code==NM_STATUS_SUCCESS,"integrated regional step="+std::to_string(index)+" env="+std::to_string(e)+" status="+std::to_string(after.statuses[e].code)+" residual="+number(after.statuses[e].diagnostics.z));
        if(reject>=0)need(after.statuses[unsigned(reject)].code!=NM_STATUS_SUCCESS,"injected frame rejection lost");(void)resets;(void)failure;
    }
};
bool physicalSame(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b){return same(a.femNodes,b.femNodes)&&same(a.femFields,b.femFields)&&same(a.femMaterialState,b.femMaterialState)&&same(a.femTopologyTetrahedra,b.femTopologyTetrahedra)&&same(a.femTopologyNodes,b.femTopologyNodes)&&same(a.cohesiveFaces,b.cohesiveFaces)&&same(a.punctureChannels,b.punctureChannels)&&same(a.topologyStates,b.topologyStates)&&same(a.environmentParameters,b.environmentParameters)&&same(a.identification,b.identification);}
void integratedChecks(Device& device,const WorldSource& authored,unsigned steps,const char* label){
    Run run(device,authored);const auto initial=run.state();RuntimeStateSnapshot one;double motion=0;
    const auto stride=run.world.dispatch.materialStateStride;
    for(unsigned env=0;env<2;++env)for(unsigned ti=0;ti<2;++ti){const auto& material=run.world.materials[ti+1];
        for(unsigned j=0;j<material.stateCount;++j)need(initial.femMaterialState[(env*2+ti)*stride+j]==run.world.stateInitials[material.stateInitialOffset+j],"runtime initial state used other material");}
    for(unsigned i=0;i<steps;++i){const auto before=run.state();run.step(i);const auto s=run.state();if(i==0)one=s;paired(s.femNodes,"accepted regional nodes");paired(s.femMaterialState,"accepted regional state");
        for(unsigned n=0;n<s.femNodes.size();++n){need(s.femNodes[n].positionAndMass.w==initial.femNodes[n].positionAndMass.w&&s.femNodes[n].velocityAndInverseMass.w==initial.femNodes[n].velocityAndInverseMass.w,"regional evolution changed mechanical mass or inverse mass");
            const auto delta=sub(xyz(s.femNodes[n].positionAndMass),xyz(initial.femNodes[n].positionAndMass));motion=std::max(motion,std::sqrt(dot(delta,delta)));}
        if(run.world.materials[1].stateCount){Inputs input;input.nodes=s.femNodes;
            for(unsigned env=0;env<2;++env)for(unsigned ti=0;ti<2;++ti){const auto base=(env*2+ti)*stride;const Matrix q=authored.objects[0].femMaterialFrameRotations.empty()?ref::identity:frame(run.world.fem.tetrahedra[ti].materialFrameRotation);
                const Matrix rate=ref::multiply(gradient(run.world,input,env,ti,true),q);const double dt=run.world.dispatch.gravityAndTimestep.w;
                const double expected=before.femMaterialState[base]+dt*(ti?2*rate[5]:rate[1]);need(std::abs(s.femMaterialState[base]-expected)<2e-7,"accepted state update used wrong regional program");
                if(ti)need(std::abs(s.femMaterialState[base+1]-(before.femMaterialState[base+1]+dt*rate[7]))<2e-7,"accepted second regional state update incorrect");}}
    }
    const auto final=run.state();need(motion>1e-7,"regional trajectory did not move");run.restore(initial);for(unsigned i=0;i<steps;++i)run.step(i);need(physicalSame(final,run.state()),"regional replay not bitwise");
    const auto before=run.state();run.step(steps,1);const auto rejected=run.state();envSame(before.femNodes,rejected.femNodes,1,"rejected regional nodes");envSame(before.femMaterialState,rejected.femMaterialState,1,"rejected regional state");envSame(before.femTopologyTetrahedra,rejected.femTopologyTetrahedra,1,"rejected regional cells");
    const auto deny=[&](const char* name,const std::function<void(RuntimeStateSnapshot&)>& change){auto c=rejected;change(c);const auto result=run.runtime.restore(c);need(!result.encoded&&physicalSame(rejected,run.state()),std::string("regional snapshot accepted or mutated state for ")+name);};
    deny("valid material rebind",[](auto& s){s.femTopologyTetrahedra[0].identity.x=2;});
    deny("object rebind",[](auto& s){++s.femTopologyTetrahedra[0].identity.y;});
    deny("connectivity rebind",[](auto& s){std::swap(s.femTopologyTetrahedra[0].nodes.x,s.femTopologyTetrahedra[0].nodes.y);});
    deny("rest rebind",[](auto& s){s.femTopologyTetrahedra[0].inverseRestRow0.x=std::nextafter(s.femTopologyTetrahedra[0].inverseRestRow0.x,0.f);});
    deny("active flag",[](auto& s){s.femTopologyTetrahedra[0].identity.w^=NM_OBJECT_ACTIVE;});
    deny("mass",[](auto& s){s.femNodes[0].positionAndMass.w=std::nextafter(s.femNodes[0].positionAndMass.w,1.f);});
    deny("inverse mass",[](auto& s){s.femNodes[3].velocityAndInverseMass.w=std::nextafter(s.femNodes[3].velocityAndInverseMass.w,0.f);});
    // Normal allocation generation, both framed and regional-only cases.
    // Changing source lineage can alter self-contact filtering even though
    // mechanical coordinates, rest operators and cell connectivity match.
    deny("topology node lineage",[](auto& s){s.femTopologyNodes[0].identity.x=s.femTopologyNodes[1].identity.x;});
    deny("topology node owner",[](auto& s){++s.femTopologyNodes[0].identity.y;});
    deny("topology node generation",[](auto& s){++s.femTopologyNodes[s.femTopologyNodes.size()/2].identity.z;});
    deny("topology node active flag",[](auto& s){s.femTopologyNodes[0].identity.w^=NM_TOPOLOGY_ACTIVE;});
    deny("rest position",[](auto& s){s.femNodes[3].restAndFixed.x=.001f;});
    deny("free-to-fixed tag",[](auto& s){s.femNodes[3].restAndFixed.w=1.f;});
    const auto parameterSlot=[&](unsigned material,const std::string& name){const auto& p=authored.materials[material].parameters;
        const auto at=std::find_if(p.begin(),p.end(),[&](const auto& v){return v.name==name;});need(at!=p.end(),"restore control missing parameter "+name);
        return run.world.materials[material].parameterOffset+unsigned(at-p.begin());};
    const auto densitySlot=parameterSlot(2,"density"),stiffnessSlot=parameterSlot(1,authored.materials[1].internalState.empty()?"kappa":"k"),defaultSlot=parameterSlot(0,"density");
    deny("selected density parameter",[&](auto& s){const auto n=run.world.dispatch.parameterCount+densitySlot;s.environmentParameters[n]=std::nextafter(s.environmentParameters[n],0.f);});
    deny("selected stiffness parameter",[&](auto& s){s.environmentParameters[stiffnessSlot]=std::nextafter(s.environmentParameters[stiffnessSlot],0.f);});
    deny("default parameter",[&](auto& s){s.environmentParameters[defaultSlot]=std::nextafter(s.environmentParameters[defaultSlot],0.f);});
    deny("scheduler base exponent",[](auto& s){++s.schedulers[s.schedulers.size()/2].baseExponent;});
    deny("allocation generation",[](auto& s){++s.allocationGeneration;});
    deny("representation",[](auto& s){s.adaptive[0].activeRepresentation=NM_REPRESENTATION_RIGID;});
    if(!authored.objects[0].femMaterialFrameRotations.empty())deny("frame",[](auto& s){s.femTopologyTetrahedra[0].materialFrameRotation={0,0,0,1};});
    run.step(steps+1,-1,1);const auto reset=run.state();envSame(one.femNodes,reset.femNodes,1,"reset regional nodes");envSame(one.femMaterialState,reset.femMaterialState,1,"reset regional material state");envSame(one.femTopologyTetrahedra,reset.femTopologyTetrahedra,1,"reset regional cells");
    std::cout<<"regional_runtime_case="<<label<<" accepted_steps="<<steps<<" environments=2 motion_m="<<motion
        <<" replay=bitwise rollback=isolated reset=bitwise snapshot_rebinding=denied mass=unchanged\n";
}
void immutableEventRestore(Device& device,const WorldSource& authored){
    for(bool framed:{false,true}){auto s=authored;if(!framed){s.objects[0].femMaterialFrameRotations.clear();s.objects[0].femMaterialFrameSourceIdentity={};}
        s.objects[0].femCapacity.cohesiveFaces=1;s.objects[0].femCapacity.punctureChannels=1;
        Run run(device,s);run.step(0);const auto initial=run.state();
        need(run.world.dispatch.cohesiveFaceCount==1&&run.world.dispatch.punctureChannelCount==1&&initial.cohesiveFaces.size()==2&&initial.punctureChannels.size()==2,
            "immutable event fixture did not retain dormant capacity");
        need((initial.cohesiveFaces[0].adjacency.w&NM_TOPOLOGY_ACTIVE)==0&&(initial.punctureChannels[0].identity.w&NM_TOPOLOGY_ACTIVE)==0,"reserved events are unexpectedly active");
        run.restore(initial);need(physicalSame(initial,run.state()),"valid immutable event-capacity restore changed state");
        const auto deny=[&](const char* name,const std::function<void(RuntimeStateSnapshot&)>& change){auto bad=initial;change(bad);const auto result=run.runtime.restore(bad);
            need(!result.encoded&&physicalSame(initial,run.state()),std::string("immutable event restore accepted or changed state for ")+name);};
        deny("activated puncture channel",[](auto& v){auto& c=v.punctureChannels[1];c.identity={0,0x80000000u,17,NM_TOPOLOGY_ACTIVE};c.originAndRadius={0,0,0,.001f};c.axisAndHalfLength={0,0,1,.01f};});
        deny("activated cohesive face",[](auto& v){auto& f=v.cohesiveFaces[0];f.nodesAndFirst={0,1,2,0};f.adjacency={1,0,17,NM_TOPOLOGY_ACTIVE|NM_TOPOLOGY_COHESIVE};f.geometry={0,0,1,.0001220703125f};});
        deny("topology active counts",[](auto& v){++v.topologyStates[0].counts.w;});
        deny("topology generation",[](auto& v){++v.topologyStates[1].roles.w;});
        deny("topology accounting",[](auto& v){v.topologyStates[0].accounting.x=std::nextafter(v.topologyStates[0].accounting.x,1.f);});
        // The source API can also cook an initially active internal cohesive
        // face without enabling mutation. Its record has no live update path
        // in an immutable world, and its own later snapshot remains valid.
        auto activeSource=s;activeSource.objects[0].mutationPolicy.cohesiveFracture=true;
        Run active(device,activeSource);const auto activeInitial=active.state();
        need((activeInitial.cohesiveFaces[0].adjacency.w&NM_TOPOLOGY_ACTIVE)!=0,"active cooked cohesive control missing");
        active.step(0);const auto activeAfter=active.state();
        need(same(activeInitial.cohesiveFaces,activeAfter.cohesiveFaces)&&same(activeInitial.topologyStates,activeAfter.topologyStates),"immutable cooked cohesive history evolved unexpectedly");
        active.restore(activeAfter);need(physicalSame(activeAfter,active.state()),"legitimate active-cohesive own snapshot rejected");
        std::cout<<"immutable_event_capacity_restore=pass framed="<<framed<<" dormant_cohesive_capacity=1 dormant_puncture_capacity=1 topology_accounting=bound\n";
    }
}
void unrelatedPosteriorRestore(Device& device,const WorldSource& authored){
    // No frame flag: these bindings must follow regional material ownership
    // alone. The fourth material is unassigned to the regional FEM object.
    auto s=authored;s.objects[0].femMaterialFrameRotations.clear();s.objects[0].femMaterialFrameSourceIdentity={};
    auto unrelated=s.materials[1];unrelated.name="synthetic_unrelated_identifiable_material";
    for(auto& p:unrelated.parameters)p.identifiable=(p.name=="density");
    s.materials.push_back(unrelated);Run run(device,s);const auto initial=run.state();auto changed=initial;
    need(!changed.identification.empty()&&changed.identification[0].identity.x==3,"unrelated posterior fixture not initialized");
    const auto unrelatedSlot=run.world.materials[3].parameterOffset;
    for(unsigned env=0;env<2;++env)changed.environmentParameters[env*run.world.dispatch.parameterCount+unrelatedSlot]+=1.f;
    changed.identification[0].momentsAndBounds.x+=1.f;
    const auto restored=run.runtime.restore(changed);need(restored.encoded,"unrelated posterior value update rejected: "+restored.message);
    const auto after=run.state();need(same(initial.femNodes,after.femNodes)&&same(initial.femMaterialState,after.femMaterialState)&&same(initial.femTopologyTetrahedra,after.femTopologyTetrahedra)&&same(changed.environmentParameters,after.environmentParameters)&&same(changed.identification,after.identification),
        "unrelated posterior restore changed regional physical state or lost permitted posterior values");
    for(unsigned target:{0u,1u}){auto bad=after;const auto slot=run.world.materials[target].parameterOffset;const auto p=run.world.parameters[slot].valueAndBounds;
        bad.identification[0].identity={target,0,slot,0};bad.identification[0].momentsAndBounds={p.x,1,p.y,p.z};
        const auto denied=run.runtime.restore(bad);need(!denied.encoded&&physicalSame(after,run.state()),"unrelated posterior retargeted into a fixed regional material");}
    std::cout<<"regional_unframed_unrelated_posterior_values=restored posterior_identity_retarget=denied fixed_parameters=unchanged\n";
}
void legacyObjectMaterialRestore(Device& device,const WorldSource& authored){
    auto legacy=authored;auto& object=legacy.objects[0];object.femMaterialIndices.clear();object.femMaterialSourceIdentity={};object.femMaterialFrameRotations.clear();object.femMaterialFrameSourceIdentity={};
    object.materialIndex=1;object.mutationPolicy.enabled=true;
    Run run(device,legacy);const auto before=run.state();auto rebuild=before;++rebuild.allocationGeneration;
    need(rebuild.femTopologyNodes[0].identity.y==0&&rebuild.femTopologyTetrahedra[0].identity.x==1&&rebuild.femTopologyTetrahedra[0].identity.y==0,"legacy restore fixture does not distinguish material from object");
    const auto result=run.runtime.restore(rebuild);need(result.encoded,"legacy object0 material1 incidence rebuild rejected: "+result.message);
    const auto after=run.state();need(physicalSame(before,after),"legacy incidence rebuild changed physical state");
    auto bad=after;bad.femTopologyNodes[0].identity.y=1;++bad.allocationGeneration;const auto denied=run.runtime.restore(bad);
    need(!denied.encoded&&physicalSame(after,run.state()),"legacy incidence rebuild confused matching material id for object owner");
    std::cout<<"legacy_restore_object0_material1=pass incidence_owner_mismatch=denied\n";
}
} // namespace
int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3||(argc==4&&std::string(argv[3])=="--cpu-only"),"usage: fem-regional-material-check GUCCIONE.nmatter NEO.nmatter [--cpu-only]");
    auto a=parseMatterFile(argv[1]),b=parseMatterFile(argv[2]);need(a.succeeded()&&b.succeeded(),diagnostic(a.diagnostics)+diagnostic(b.diagnostics));
    set(a.material,"density",900);set(b.material,"density",1200); // explicit synthetic fixture densities only
    const auto authored=source(a.material,b.material);std::cout<<std::setprecision(17)<<"regional_abi="<<NM_MATTER_ABI_VERSION<<'\n';cpuChecks(authored);
    if(argc==4){std::cout<<"fem_regional_material_compiler_check=pass anatomical_wall_qualified=false\n";return 0;}
    Device device;std::cout<<"regional_device="<<[device.device name].UTF8String<<'\n';directChecks(device,authored);learnedChecks(device);projectionChecks(device);
    integratedChecks(device,authored,16,"source_laws_framed");integratedChecks(device,source(a.material,b.material,false),2,"source_laws_unframed");
    const auto [sa,sb]=stateMaterials();integratedChecks(device,source(sa,sb),2,"heterogeneous_state_programs");legacyObjectMaterialRestore(device,authored);unrelatedPosteriorRestore(device,authored);immutableEventRestore(device,authored);
    std::cout<<"fem_regional_material_check=pass anatomical_wall_qualified=false source_density_supplied=false active_cardiac_source_reproduced=false\n";return 0;
}catch(const std::exception& error){std::cerr<<"fem regional material check failed: "<<error.what()<<'\n';return 1;}}}
