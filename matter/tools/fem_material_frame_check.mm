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

// Synthetic conforming two-tet constitutive and transaction fixture. The
// Guccione law is sourced; this geometry, density, loading and other path
// controls are not a calibrated heart or an anatomical-wall admission.
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
WorldSource source(const MaterialProgram& material,bool framed=true,bool mixed=false){
    WorldSource s;s.environmentCount=2;s.gravity={0,0,0};s.frameTimestep=1e-4;s.materials={material};
    s.mixedSolver.relativeResidual=1e-7;s.mixedSolver.newtonIterations=12;s.mixedSolver.fgmresIterations=64;
    ObjectSource o;o.name="synthetic_shared_node_material_frames";o.representation=Representation::fem;o.mixedFEM=mixed;
    o.deformableContact=false;o.deformableSelfContact=false;o.characteristicLength=.01;
    o.femNodes={{{0,0,0}},{{.01,0,0}},{{0,.01,0}},{{0,0,.01}},{{0,0,-.01}}};
    o.tetrahedra={{{0,1,2,3}},{{0,2,1,4}}};o.femFixedNodes={0,1,2};o.femInitialVelocity={.025,.017,.013};
    if(framed){o.femMaterialFrameRotations={{{0,0,std::sin(.305),std::cos(.305)}},{{0,std::sin(-.245),0,std::cos(-.245)}}};
        // Opaque nonzero fixture identity. It represents synthetic frame
        // authoring, not a source anatomical mesh or measured microstructure.
        o.femMaterialFrameSourceIdentity={0xf731c96c934e625eull,0x836db2ec25021611ull,0x71a72c65dc7b2f5aull,0x36ab5d1b57e2bb96ull};}
    s.objects={o};return s;
}
CompiledWorld cook(const WorldSource& s){auto c=compileWorld(s,{.maximumRateExponent=0,.emitSpecializedMetal=false});need(c.succeeded(),"frame fixture compile: "+diagnostic(c.diagnostics));return std::move(c.world);}

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
    for(unsigned e=0;e<2*w.dispatch.tetrahedronCount;++e)for(unsigned i=0;i<w.stateInitials.size();++i)v.state[e*w.dispatch.materialStateStride+i]=w.stateInitials[i];
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
using Oracle=std::function<ref::Result(const Matrix&,const Matrix&,const Matrix&,const Matrix&)>;
double compare(const CompiledWorld& w,const Inputs& v,const std::vector<NMFEMElementVectorGPU>& actual,bool tangent,const Oracle& oracle){
    double error=0,scale=0;
    for(unsigned e=0;e<2;++e)for(unsigned ti=0;ti<w.dispatch.tetrahedronCount;++ti){const auto& t=w.fem.tetrahedra[ti];
        const Matrix q=(w.objects[t.identity.y].flags&NM_OBJECT_FEM_MATERIAL_FRAME)?frame(t.materialFrameRotation):ref::identity;
        const auto result=oracle(gradient(w,v,e,ti),gradient(w,v,e,ti,false,true),gradient(w,v,e,ti,true),q);
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
void learnedFrameChecks(Device& d){
    // Synthetic single-neuron ICNN: W=softplus_beta(w*(I4-1)+b)
    // +g*(J+1/J-2). This closed form permits an independent tensor
    // derivative, rather than comparing the production network to itself.
    auto parsed=parseMatter(R"(material synthetic_framed_icnn_control {
        parameter density : kg/m^3 = 1000; parameter scale : Pa = 1;
        model generic; energy = 0*scale; valid = J(); supports fem;
    })");need(parsed.succeeded(),diagnostic(parsed.diagnostics));
    auto material=parsed.material;material.hint=ConstitutiveHint::polyconvexICNN;
    material.mixed.fibreDirection={.6,.8,0};
    LearnedMaterialSource network;network.invariantCount=5;network.softplusBeta=2.5f;
    network.determinantFloor=.05f;network.growthCoefficient=.03125f;
    LearnedLayerSource layer;layer.inputWidth=5;layer.outputWidth=1;
    layer.inputWeights={0,0,0,0,.75f};layer.biases={.125f};network.layers={layer};material.learned=network;
    const auto authored=source(material);const auto world=cook(authored);const auto input=manufactured(world);
    const auto rawFibre=world.mixedMaterials[0].fibre;
    Vec a=xyz(rawFibre);const double length=std::sqrt(dot(a,a));for(double& value:a)value/=length;
    const Oracle oracle=[a](const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
        const Vec fibre=transform(q,a),u=transform(f,fibre),du=transform(h,fibre);
        const double j=ref::determinant(f),ell=ref::contract(ref::inverseTranspose(f),h),dj=j*ell;
        const double z=.75*(dot(u,u)-1)+.125,slope=1/(1+std::exp(-2.5*z));
        const double dz=1.5*dot(u,du),dslope=2.5*slope*(1-slope)*dz;
        const Matrix it=ref::inverseTranspose(f),cofactor=ref::scale(it,j);
        const Matrix dit=ref::scale(ref::multiply(ref::multiply(it,ref::transpose(h)),it),-1);
        const Matrix dcofactor=ref::add(ref::scale(it,dj),ref::scale(dit,j));
        ref::Result result;
        result.stress=ref::add(ref::scale(outer(u,fibre),1.5*slope),ref::scale(cofactor,.03125*(1-1/(j*j))));
        result.tangent=ref::add(ref::scale(ref::add(ref::scale(outer(u,fibre),dslope),ref::scale(outer(du,fibre),slope)),1.5),
            ref::scale(ref::add(ref::scale(cofactor,2*dj/(j*j*j)),ref::scale(dcofactor,1-1/(j*j))),.03125));return result;
    };
    const auto force=dispatch(d,world,input,false),tangent=dispatch(d,world,input,true);
    const double fe=compare(world,input,force,false,oracle),te=compare(world,input,tangent,true,oracle),fd=finiteDifference(d,world,input,tangent);
    need(fe<3e-4&&te<3e-4&&fd<3e-3,"framed ICNN independent oracle mismatch force="+number(fe)+" tangent="+number(te)+" fd="+number(fd));
    // Independent frame equivalence: the same physical fibre is authored
    // directly in world-reference coordinates, with all frame logic disabled.
    // Each element has its own frame, so make one unframed control per frame.
    auto equivalentForce=force,equivalentTangent=tangent;
    for(unsigned ti=0;ti<world.dispatch.tetrahedronCount;++ti){
        auto unframed=source(material,false);unframed.materials[0].mixed.fibreDirection=transform(frame(world.fem.tetrahedra[ti].materialFrameRotation),a);
        const auto control=cook(unframed);const auto cf=dispatch(d,control,input,false),ct=dispatch(d,control,input,true);
        for(unsigned env=0;env<2;++env){const auto index=env*world.dispatch.tetrahedronCount+ti;equivalentForce[index]=cf[index];equivalentTangent[index]=ct[index];}}
    const double frameForce=difference(force,equivalentForce),frameTangent=difference(tangent,equivalentTangent);
    need(frameForce<3e-4&&frameTangent<3e-4,"framed ICNN differs from independently authored physical fibre");
    const auto unrotated=cook(source(material,false));const double angular=difference(force,dispatch(d,unrotated,input,false));
    need(angular>1e-3,"ICNN fixture did not resolve material-frame angular response");
    std::cout<<"framed_icnn_force_error="<<fe<<" framed_icnn_tangent_error="<<te<<" framed_icnn_fd_error="<<fd
        <<" framed_icnn_world_fibre_force_error="<<frameForce<<" framed_icnn_world_fibre_tangent_error="<<frameTangent
        <<" framed_icnn_angular_consequence="<<angular<<" trained_material_qualified=false\n";
}
void directChecks(Device& d,const MaterialProgram& guccione){
    const auto s=source(guccione);const auto w=cook(s);const auto v=manufactured(w);
    const auto force=dispatch(d,w,v,false),tangent=dispatch(d,w,v,true);
    const Oracle passive=[](const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){return ref::framedGuccione(f,h,q);};
    const double forceError=compare(w,v,force,false,passive),tangentError=compare(w,v,tangent,true,passive),fd=finiteDifference(d,w,v,tangent);
    need(forceError<3e-4&&tangentError<3e-4,"source Guccione production frame oracle mismatch force="+number(forceError)+" tangent="+number(tangentError));
    need(fd<3e-3,"production frame tangent differs from force FD: "+number(fd));
    auto identitySource=s;for(auto& q:identitySource.objects[0].femMaterialFrameRotations)q={0,0,0,1};
    const auto wi=cook(identitySource),wl=cook(source(guccione,false));
    const auto identityForce=dispatch(d,wi,v,false),legacyForce=dispatch(d,wl,v,false);
    const double identityError=difference(identityForce,legacyForce),angular=difference(force,identityForce);
    need(identityError==0,"identity frame changed legacy force arithmetic");need(angular>1e-3,"anisotropic frame rotation had no resolvable force consequence");
    need(difference(dispatch(d,wi,v,true),dispatch(d,wl,v,true))==0,"identity frame changed legacy tangent arithmetic");
    paired(force,"source element forces");paired(tangent,"source element tangent");

    const auto parsed=parseMatter(R"(material synthetic_frame_state_rate_path {
        parameter density : kg/m^3 = 1000; parameter k : Pa = 1000; parameter eta : Pa*s = 20;
        state history : one = 0.23 transfer average;
        model generic; energy = 0.5*k*pow(F(0,0)-next(history),2);
        dissipation = 0.5*eta*pow(Fdot(0,2),2);
        update history = history + dt()*Fdot(0,1); valid = J(); supports fem;
    })");need(parsed.succeeded(),diagnostic(parsed.diagnostics));
    const auto wr=cook(source(parsed.material));const auto vr=manufactured(wr);const double dt=wr.dispatch.gravityAndTimestep.w;
    const Oracle rate=[dt](const Matrix& f,const Matrix& h,const Matrix& r,const Matrix& q){
        const Matrix fl=ref::multiply(f,q),hl=ref::multiply(h,q),rl=ref::multiply(r,q);ref::Result out;
        const double old=double(float(.23)),next=old+dt*rl[1];Matrix p{},dp{};p[0]=1000*(fl[0]-next);p[2]=20*rl[2];
        dp[0]=1000*(hl[0]-hl[1]);dp[2]=20*hl[2]/dt;out.stress=ref::multiply(p,ref::transpose(q));out.tangent=ref::multiply(dp,ref::transpose(q));return out;};
    const auto rf=dispatch(d,wr,vr,false),rt=dispatch(d,wr,vr,true);
    const double rfe=compare(wr,vr,rf,false,rate),rte=compare(wr,vr,rt,true,rate),rfd=finiteDifference(d,wr,vr,rt);
    need(rfe<3e-4&&rte<3e-3&&rfd<4e-3,"synthetic state/rate frame path mismatch force="+number(rfe)+" tangent="+number(rte)+" fd="+number(rfd));

    auto activeMaterial=guccione;activeMaterial.mixed.maximumActiveTension=12000;
    const auto wa=cook(source(activeMaterial,true,true));auto va=manufactured(wa);const unsigned count=2*wa.dispatch.femNodeCount;
    for(unsigned i=0;i<count;++i){va.fields[i].secondary.x=.3f;va.direction[count+i].w=.001f;}
    auto off=va;for(auto& f:off.fields)f.secondary.x=0;for(unsigned i=0;i<count;++i)off.direction[count+i].w=0;
    const auto af=dispatch(d,wa,va,false),at=dispatch(d,wa,va,true),of=dispatch(d,wa,off,false),ot=dispatch(d,wa,off,true);
    auto activeOnly=af,activeTangent=at;
    for(unsigned i=0;i<af.size();++i){auto* a=reinterpret_cast<float*>(&activeOnly[i]);auto* t=reinterpret_cast<float*>(&activeTangent[i]);const auto* b=reinterpret_cast<const float*>(&of[i]);const auto* u=reinterpret_cast<const float*>(&ot[i]);for(unsigned j=0;j<16;++j){a[j]-=b[j];t[j]-=u[j];}}
    const double activeDt=wa.dispatch.gravityAndTimestep.w;
    const Oracle active=[activeDt](const Matrix& f,const Matrix& h,const Matrix&,const Matrix& q){
        const Vec fibre{q[0],q[3],q[6]},u=transform(f,fibre),du=transform(h,fibre);const double l2=dot(u,u),j=ref::determinant(f),ell=ref::contract(ref::inverseTranspose(f),h);
        const double a=double(float(.3)),da=double(float(.001))/activeDt,prefactor=12000*j/l2;ref::Result out;
        out.stress=ref::scale(outer(u,fibre),prefactor*a);
        out.tangent=ref::scale(ref::add(ref::scale(outer(u,fibre),a*(ell-2*dot(u,du)/l2)+da),ref::scale(outer(du,fibre),a)),prefactor);return out;};
    const double afe=compare(wa,va,activeOnly,false,active),ate=compare(wa,va,activeTangent,true,active),afd=finiteDifference(d,wa,va,at);
    need(afe<3e-4&&ate<3e-4&&afd<4e-3,"mixed active frame control mismatch force="+number(afe)+" tangent="+number(ate)+" fd="+number(afd));
    std::cout<<"source_frame_force_error="<<forceError<<" source_frame_tangent_error="<<tangentError<<" source_frame_fd_error="<<fd<<" identity_error="<<identityError<<" anisotropic_angular_consequence="<<angular
        <<" synthetic_state_rate_force_error="<<rfe<<" synthetic_state_rate_tangent_error="<<rte<<" synthetic_state_rate_fd_error="<<rfd
        <<" mixed_active_force_error="<<afe<<" mixed_active_tangent_error="<<ate<<" mixed_active_fd_error="<<afd<<" mixed_source_equivalent=false\n";
}

struct Run {
    Device& device;CompiledWorld world;Runtime runtime;id<MTLBuffer> statuses;
    Run(Device& d,const WorldSource& s):device(d),world(cook(s)){
        const auto path=std::filesystem::temp_directory_path()/(std::string("numi-frame-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        CompileResult package;package.world=world;std::string error;need(writePackage(package,path,&error),error);CompiledWorld loaded;
        const bool ok=readPackage(path,loaded,nullptr,&error);std::error_code remove;std::filesystem::remove(path,remove);need(ok&&!remove&&loaded.fingerprint==world.fingerprint,"frame package roundtrip: "+error);world=std::move(loaded);
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
        request.phase=EncodePhase::postCommit;result=runtime.encode(request);need(result.encoded,result.message);[command commit];[command waitUntilCompleted];need(command.status==MTLCommandBufferStatusCompleted,"integrated frame command failed");
        const auto after=state();for(unsigned e=0;e<2;++e)if(int(e)!=reject)need(after.statuses[e].code==NM_STATUS_SUCCESS,"integrated frame step="+std::to_string(index)+" env="+std::to_string(e)+" status="+std::to_string(after.statuses[e].code)+" residual="+number(after.statuses[e].diagnostics.z));
        if(reject>=0)need(after.statuses[unsigned(reject)].code!=NM_STATUS_SUCCESS,"injected frame rejection lost");(void)resets;(void)failure;
    }
};
bool physicalSame(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b){return same(a.femNodes,b.femNodes)&&same(a.femFields,b.femFields)&&same(a.femMaterialState,b.femMaterialState)&&same(a.femTopologyTetrahedra,b.femTopologyTetrahedra);}
void integratedChecks(Device& d,const MaterialProgram& material){
    Run run(d,source(material));const auto initial=run.state();RuntimeStateSnapshot one;constexpr unsigned steps=16;
    double motion=0;for(unsigned i=0;i<steps;++i){run.step(i);const auto s=run.state();if(i==0)one=s;paired(s.femNodes,"accepted nodes");paired(s.femMaterialState,"accepted material state");
        for(unsigned n=0;n<s.femNodes.size();++n){need(std::bit_cast<unsigned>(s.femNodes[n].positionAndMass.w)==std::bit_cast<unsigned>(initial.femNodes[n].positionAndMass.w),"material frame changed mechanical nodal mass");
            const auto delta=sub(xyz(s.femNodes[n].positionAndMass),xyz(initial.femNodes[n].positionAndMass));motion=std::max(motion,std::sqrt(dot(delta,delta)));}}
    const auto final=run.state();need(motion>1e-7,"integrated frame trajectory did not move");run.restore(initial);for(unsigned i=0;i<steps;++i)run.step(i);need(physicalSame(final,run.state()),"material frame replay not bitwise");
    const auto before=run.state();run.step(steps,1);const auto rejected=run.state();envSame(before.femNodes,rejected.femNodes,1,"rejected nodes");envSame(before.femMaterialState,rejected.femMaterialState,1,"rejected material state");envSame(before.femTopologyTetrahedra,rejected.femTopologyTetrahedra,1,"rejected frames");
    auto corrupt=rejected;need(!corrupt.femTopologyTetrahedra.empty(),"missing frame snapshot topology");corrupt.femTopologyTetrahedra[0].materialFrameRotation={0,0,0,1};
    auto denied=run.runtime.restore(corrupt);need(!denied.encoded&&physicalSame(rejected,run.state()),"valid alternate quaternion bypassed snapshot frame identity");
    corrupt=rejected;corrupt.femTopologyTetrahedra[0].materialFrameRotation.x=std::numeric_limits<float>::quiet_NaN();denied=run.runtime.restore(corrupt);
    need(!denied.encoded&&physicalSame(rejected,run.state()),"nonfinite snapshot frame accepted or mutated state");
    // A frame belongs to its exact immutable cell. Checking only q would
    // allow the same quaternion to be rebound through nodes/rest/material.
    const std::vector<std::pair<std::string,std::function<void(NMTetrahedronGPU&)>>> rebindings{
        {"nodes",[](auto& t){std::swap(t.nodes.x,t.nodes.y);}},
        {"rest operator",[](auto& t){t.inverseRestRow0.x=std::nextafter(t.inverseRestRow0.x,std::numeric_limits<float>::infinity());}},
        {"material identity",[](auto& t){++t.identity.x;}},
        {"object identity",[](auto& t){++t.identity.y;}},
        {"active flag",[](auto& t){t.identity.w^=NM_OBJECT_ACTIVE;}}};
    for(const auto& [name,change]:rebindings){corrupt=rejected;change(corrupt.femTopologyTetrahedra[0]);denied=run.runtime.restore(corrupt);
        need(!denied.encoded&&physicalSame(rejected,run.state()),"snapshot frame "+name+" rebinding was accepted or mutated state");}
    corrupt=rejected;corrupt.adaptive[0].activeRepresentation=NM_REPRESENTATION_RIGID;denied=run.runtime.restore(corrupt);
    need(!denied.encoded&&physicalSame(rejected,run.state()),"snapshot changed framed FEM into rigid representation");
    corrupt=rejected;++corrupt.allocationGeneration;denied=run.runtime.restore(corrupt);
    need(!denied.encoded&&physicalSame(rejected,run.state()),"snapshot changed immutable framed allocation generation");
    run.step(steps+1,-1,1);const auto reset=run.state();envSame(one.femNodes,reset.femNodes,1,"reset nodes");envSame(one.femTopologyTetrahedra,reset.femTopologyTetrahedra,1,"reset frames");
    std::cout<<"accepted_steps="<<steps<<" environments=2 motion_m="<<motion<<" replay=bitwise rollback=isolated reset=bitwise frame_snapshot_tamper=denied immutable_cell_rebinding=denied mass=unchanged\n";
}
} // namespace

int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==2,"usage: fem-material-frame-check GUCCIONE.nmatter");auto parsed=parseMatterFile(argv[1]);need(parsed.succeeded(),diagnostic(parsed.diagnostics));set(parsed.material,"density",1000); // synthetic only
    Device device;std::cout<<std::setprecision(17)<<"frame_device="<<[device.device name].UTF8String<<" abi="<<NM_MATTER_ABI_VERSION<<'\n';
    directChecks(device,parsed.material);learnedFrameChecks(device);integratedChecks(device,parsed.material);
    std::cout<<"fem_material_frame_check=pass anatomical_wall_qualified=false source_density_supplied=false active_cardiac_source_reproduced=false\n";return 0;
}catch(const std::exception& error){std::cerr<<"fem material frame check failed: "<<error.what()<<'\n';return 1;}}}
