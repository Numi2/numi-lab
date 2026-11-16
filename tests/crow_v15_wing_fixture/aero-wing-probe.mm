#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <iomanip>

struct Surface { uint32_t body; float side,length,area; };
struct Wind { mr_float4 velocityDensity[3]; uint32_t environments,bodyStride,pad0,pad1; };
static_assert(sizeof(Surface)==16 && sizeof(Wind)==64);
struct Aero { __strong id<MTLComputePipelineState> pipeline; __strong id<MTLBuffer> surfaces; __strong id<MTLBuffer> captured; Wind wind; unsigned encoded=0; };
static bool encode(void* data,const metalrobo::MetalWorldDevicePhysicsPass& pass){
 auto& a=*static_cast<Aero*>(data);if(pass.phase!=metalrobo::MetalWorldDevicePhysicsPhase::preDynamics)return true;
 id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;id<MTLBuffer> states=(__bridge id<MTLBuffer>)pass.currentBodies;id<MTLBuffer> forces=(__bridge id<MTLBuffer>)pass.bodyWrenches;
 if(!cb||!states||!forces||pass.environmentCount!=3||pass.bodyStateStride!=41||pass.bodyWrenchStride!=41)return false;
 id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];if(!enc)return false;
 [enc setComputePipelineState:a.pipeline];[enc setBuffer:states offset:0 atIndex:0];[enc setBuffer:forces offset:0 atIndex:1];[enc setBuffer:a.surfaces offset:0 atIndex:2];[enc setBytes:&a.wind length:sizeof(a.wind) atIndex:3];[enc setBuffer:a.captured offset:0 atIndex:4];
 [enc dispatchThreads:MTLSizeMake(18,1,1) threadsPerThreadgroup:MTLSizeMake(18,1,1)];[enc endEncoding];++a.encoded;return true;
}
static void abortAero(void*,void*){}
struct V{double x,y,z;};static V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}static V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}static V operator*(V a,double s){return {a.x*s,a.y*s,a.z*s};}static double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}static V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}static V unit(V a){return a*(1/std::sqrt(dot(a,a)));}
static V rotate(const std::array<double,4>&q,V v){V u{q[0],q[1],q[2]};return v+cross(u,cross(u,v)+v*q[3])*2;}
struct Load {V f{},t{};};
static Load reference(const metalrobo::ArticulatedBodyKinematics& b,const Surface&s,const mr_float4&w){
 V chord=rotate(b.orientation,{1,0,0}),span=rotate(b.orientation,{0,s.side,0}),normal=unit(cross(chord,span));V velocity{b.linearVelocity[0],b.linearVelocity[1],b.linearVelocity[2]},omega{b.angularVelocity[0],b.angularVelocity[1],b.angularVelocity[2]},wind{w.x,w.y,w.z};Load out;
 for(int i=0;i<4;++i){double eta=(i+.5)/4-.5;V r=rotate(b.orientation,{0,s.side*s.length*eta,0});V incoming=wind-(velocity+cross(omega,r));incoming=incoming-span*dot(incoming,span);double speed2=dot(incoming,incoming);if(!(speed2>1e-12))continue;V flow=unit(incoming),travel=flow*(-1);double alpha=std::atan2(dot(travel,normal),dot(travel,chord));double cl=std::clamp(3*alpha,-1.2,1.2),cd=.06+.1*cl*cl;V f=(unit(cross(span,flow))*cl+flow*cd)*(.5*w.w*speed2*(s.area/4.0));out.f=out.f+f;out.t=out.t+cross(r,f);}return out;
}
static void vjson(V x){std::cout<<'['<<x.x<<','<<x.y<<','<<x.z<<']';}
int main(int argc,char**argv){@autoreleasepool{try{
 if(argc!=3)throw std::runtime_error("expected engine and experimental aero metallibs");
 auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
 metalrobo::CompiledWorld world;auto comp=metalrobo::compileMetalWorld(m,0,world);if(!comp.succeeded())throw std::runtime_error(comp.message);auto&a=m.articulations[0];
 auto dof=[&](const char*name){auto it=std::find(m.dofNames.begin(),m.dofNames.end(),name);if(it==m.dofNames.end())throw std::runtime_error("missing joint");return m.dofs.at(it-m.dofNames.begin());};
 auto le=dof("crow_left_elbow_flex"),lw=dof("crow_left_wrist_flex"),re=dof("crow_right_elbow_flex"),rw=dof("crow_right_wrist_flex");
 std::vector<float> qs,vs,efforts(3*a.nv,0);for(int i=0;i<3;++i){qs.insert(qs.end(),m.defaultQ.begin(),m.defaultQ.end());vs.insert(vs.end(),m.defaultV.begin(),m.defaultV.end());}
 qs[2*a.nq+le.qIndex]=.9f;qs[2*a.nq+lw.qIndex]=.4f;qs[2*a.nq+re.qIndex]=.9f;qs[2*a.nq+rw.qIndex]=.4f;
 vs[2*a.nv+le.vIndex]=1;vs[2*a.nv+lw.vIndex]=-.3f;vs[2*a.nv+re.vIndex]=1;vs[2*a.nv+rw.vIndex]=-.3f;
 const float lens[3]={.08355912f,.09530571f,.09089033f};float total=lens[0]+lens[1]+lens[2];Surface surfaces[6]{};const uint32_t indices[6]={5,35,37,6,38,40};for(int i=0;i<6;++i)surfaces[i]={indices[i],i<3?1.0f:-1.0f,lens[i%3],.075f*lens[i%3]/total};
 Wind wind{{{0,0,0,1.225f},{-5,0,-1,1.225f},{-5,0,-1,1.225f}},3,41,0,0};
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError* error=nil;NSString* filename=[NSString stringWithUTF8String:argv[2]];id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:filename] error:&error];if(!lib)throw std::runtime_error([[error description] UTF8String]);id<MTLFunction> fn=[lib newFunctionWithName:@"crow_articulated_wing_aero"];id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:fn error:&error];if(!pipeline)throw std::runtime_error([[error description] UTF8String]);
 Aero aero{};aero.pipeline=pipeline;aero.surfaces=[device newBufferWithBytes:surfaces length:sizeof(surfaces) options:MTLResourceStorageModeShared];aero.captured=[device newBufferWithLength:18*sizeof(MRABABodyWrenchGPU) options:MTLResourceStorageModeShared];aero.wind=wind;if(!aero.surfaces||!aero.captured)throw std::runtime_error("aero allocation failed");
 metalrobo::MetalWorldBatch b;b.environmentCount=3;b.controlStepCount=1;b.initialQ=qs;b.initialV=vs;b.efforts=efforts;
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.0001f;cfg.physicsSubsteps=1;cfg.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;cfg.applyBodyDamping=false;
 metalrobo::MetalWorldConfig settings;settings.metallibPath=argv[1];settings.preferParallelABA=false;metalrobo::MetalWorldContext context(settings);metalrobo::MetalWorldResult results[2];metalrobo::MetalWorldDiagnostics d[2];d[0]=context.run(world,b,cfg,results[0]);
 cfg.devicePhysicsProgram.context=&aero;cfg.devicePhysicsProgram.encode=encode;cfg.devicePhysicsProgram.abort=abortAero;cfg.devicePhysicsProgram.flags=metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches;cfg.devicePhysicsProgram.fingerprint=0x81dbe371a5419b65ULL;d[1]=context.run(world,b,cfg,results[1]);
 bool ok=true;for(auto&x:d)ok=ok&&x.succeeded()&&x.dispatched&&x.published&&x.successfulStepCount==3&&x.failedStepCount==0;ok=ok&&aero.encoded==1;
 const auto* observed=static_cast<const MRABABodyWrenchGPU*>(aero.captured.contents);std::array<Load,18> ref{};
 for(int lane=0;lane<3;++lane){std::vector<double>q(qs.begin()+lane*a.nq,qs.begin()+(lane+1)*a.nq),v(vs.begin()+lane*a.nv,vs.begin()+(lane+1)*a.nv);std::vector<metalrobo::ArticulatedBodyKinematics> kin(m.bodies.size());auto k=metalrobo::computeArticulatedBodyKinematics(m,0,q,v,kin);ok=ok&&k.succeeded();for(int i=0;i<6;++i)ref[lane*6+i]=reference(kin[surfaces[i].body],surfaces[i],wind.velocityDensity[lane]);}
 std::cout<<std::setprecision(12)<<"{\"world_fingerprint\":"<<comp.fingerprint<<",\"encodes\":"<<aero.encoded<<",\"initial_q\":[";for(size_t i=0;i<qs.size();++i){if(i)std::cout<<',';std::cout<<qs[i];}std::cout<<"],\"initial_v\":[";for(size_t i=0;i<vs.size();++i){if(i)std::cout<<',';std::cout<<vs[i];}std::cout<<"],\"runs\":[";
 for(int arm=0;arm<2;++arm){if(arm)std::cout<<',';std::cout<<"{\"ok\":"<<(d[arm].succeeded()?"true":"false")<<",\"steps\":"<<d[arm].successfulStepCount<<",\"failed\":"<<d[arm].failedStepCount<<",\"message\":"<<std::quoted(d[arm].message)<<",\"final_v\":[";for(size_t i=0;i<results[arm].finalV.size();++i){if(i)std::cout<<',';std::cout<<results[arm].finalV[i];}std::cout<<"]}";}
 std::cout<<"],\"loads\":[";for(int i=0;i<18;++i){if(i)std::cout<<',';auto&g=observed[i];V gf{g.force.x,g.force.y,g.force.z},gt{g.torque.x,g.torque.y,g.torque.z};double diffF=std::max({std::abs(gf.x-ref[i].f.x),std::abs(gf.y-ref[i].f.y),std::abs(gf.z-ref[i].f.z)}),diffT=std::max({std::abs(gt.x-ref[i].t.x),std::abs(gt.y-ref[i].t.y),std::abs(gt.z-ref[i].t.z)});ok=ok&&std::isfinite(diffF)&&std::isfinite(diffT)&&diffF<=2e-5&&diffT<=2e-6;if(i<6)ok=ok&&std::sqrt(dot(gf,gf))<1e-8&&std::sqrt(dot(gt,gt))<1e-8;
  std::cout<<"{\"lane\":"<<i/6<<",\"body\":"<<std::quoted(m.bodyNames[surfaces[i%6].body])<<",\"gpu_force\":";vjson(gf);std::cout<<",\"gpu_torque\":";vjson(gt);std::cout<<",\"cpu_force\":";vjson(ref[i].f);std::cout<<",\"cpu_torque\":";vjson(ref[i].t);std::cout<<'}';}
 for(int i=0;i<3;++i){auto l=observed[6+i].force,r=observed[9+i].force;ok=ok&&std::abs(l.x-r.x)<1e-5&&std::abs(l.y+r.y)<1e-5&&std::abs(l.z-r.z)<1e-5;}
 bool differs=false;for(int i:{1,2,4,5}){auto u=observed[6+i].force,v=observed[12+i].force;double delta=std::sqrt((u.x-v.x)*(u.x-v.x)+(u.y-v.y)*(u.y-v.y)+(u.z-v.z)*(u.z-v.z));differs=differs||delta>=.001;}ok=ok&&differs;
 if(results[0].finalV.size()==3*a.nv&&results[1].finalV.size()==3*a.nv)for(int lane=1;lane<3;++lane){double delta=0;for(unsigned i=0;i<a.nv;++i)delta=std::max(delta,std::abs(double(results[0].finalV[lane*a.nv+i]-results[1].finalV[lane*a.nv+i])));ok=ok&&delta>=1e-6;}else ok=false;
 std::cout<<"],\"criterion_pass\":"<<(ok?"true":"false")<<"}\n";return ok?0:2;
}catch(const std::exception&e){std::cerr<<"exception: "<<e.what()<<'\n';return 1;}}}
