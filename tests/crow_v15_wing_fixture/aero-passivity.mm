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

int main(int argc,char**argv){@autoreleasepool{try{
 if(argc!=3)throw std::runtime_error("expected engine and aero metallibs");
 auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;m.world.gravityAndTimestep.x=m.world.gravityAndTimestep.y=m.world.gravityAndTimestep.z=0;
 metalrobo::CompiledWorld world;auto comp=metalrobo::compileMetalWorld(m,0,world);if(!comp.succeeded())throw std::runtime_error(comp.message);auto&a=m.articulations[0];constexpr int steps=240;
 auto dof=[&](const char*name){auto it=std::find(m.dofNames.begin(),m.dofNames.end(),name);if(it==m.dofNames.end())throw std::runtime_error("missing joint");return m.dofs.at(it-m.dofNames.begin());};
 auto le=dof("crow_left_elbow_flex"),lw=dof("crow_left_wrist_flex"),re=dof("crow_right_elbow_flex"),rw=dof("crow_right_wrist_flex");
 std::vector<float> qs,vs,efforts(steps*3*a.nv,0);for(int i=0;i<3;++i){qs.insert(qs.end(),m.defaultQ.begin(),m.defaultQ.end());vs.insert(vs.end(),m.defaultV.begin(),m.defaultV.end());}
 vs[a.nv]=5;vs[2*a.nv]=5;qs[2*a.nq+le.qIndex]=.4f;qs[2*a.nq+re.qIndex]=.4f;qs[2*a.nq+lw.qIndex]=.6f;qs[2*a.nq+rw.qIndex]=.6f;
 vs[2*a.nv+le.vIndex]=.2f;vs[2*a.nv+re.vIndex]=.2f;vs[2*a.nv+lw.vIndex]=-.1f;vs[2*a.nv+rw.vIndex]=-.1f;
 const float lens[3]={.08355912f,.09530571f,.09089033f};float total=lens[0]+lens[1]+lens[2];Surface surfaces[6]{};const uint32_t indices[6]={5,35,37,6,38,40};for(int i=0;i<6;++i)surfaces[i]={indices[i],i<3?1.0f:-1.0f,lens[i%3],.075f*lens[i%3]/total};
 Wind wind{{{0,0,0,1.225f},{0,0,0,1.225f},{0,0,0,1.225f}},3,41,0,0};id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError* error=nil;id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]] error:&error];if(!lib)throw std::runtime_error([[error description] UTF8String]);id<MTLFunction> fn=[lib newFunctionWithName:@"crow_articulated_wing_aero"];id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:fn error:&error];if(!pipeline)throw std::runtime_error([[error description] UTF8String]);
 Aero aero{};aero.pipeline=pipeline;aero.surfaces=[device newBufferWithBytes:surfaces length:sizeof(surfaces) options:MTLResourceStorageModeShared];aero.captured=[device newBufferWithLength:18*sizeof(MRABABodyWrenchGPU) options:MTLResourceStorageModeShared];aero.wind=wind;if(!aero.surfaces||!aero.captured)throw std::runtime_error("aero allocation failed");
 metalrobo::MetalWorldBatch b;b.environmentCount=3;b.controlStepCount=steps;b.initialQ=qs;b.initialV=vs;b.efforts=efforts;
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=1.0f/240;cfg.physicsSubsteps=4;cfg.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;cfg.applyBodyDamping=false;
 metalrobo::MetalWorldConfig settings;settings.metallibPath=argv[1];settings.preferParallelABA=false;metalrobo::MetalWorldContext context(settings);metalrobo::MetalWorldResult results[2];metalrobo::MetalWorldDiagnostics d[2];d[0]=context.run(world,b,cfg,results[0]);
 cfg.devicePhysicsProgram.context=&aero;cfg.devicePhysicsProgram.encode=encode;cfg.devicePhysicsProgram.abort=abortAero;cfg.devicePhysicsProgram.flags=metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches;cfg.devicePhysicsProgram.fingerprint=0x81dbe371a5419b65ULL;d[1]=context.run(world,b,cfg,results[1]);
 bool ok=aero.encoded==steps*4;for(auto&x:d)ok=ok&&x.succeeded()&&x.dispatched&&x.published&&x.successfulStepCount==3*steps&&x.failedStepCount==0;
 metalrobo::ArticulatedDynamicsConfig ref;ref.gravity={0,0,0};ref.applyBodyDamping=false;std::vector<double> energies[2];
 for(int arm=0;arm<2;++arm){energies[arm].resize(3*(steps+1));if(results[arm].observations.size()!=steps*3*(a.nq+a.nv)){ok=false;continue;}
  for(int lane=0;lane<3;++lane){std::vector<double>q(qs.begin()+lane*a.nq,qs.begin()+(lane+1)*a.nq),v(vs.begin()+lane*a.nv,vs.begin()+(lane+1)*a.nv);metalrobo::ArticulatedInvariants inv;auto id=metalrobo::computeArticulatedInvariants(m,0,q,v,inv,ref);ok=ok&&id.succeeded();energies[arm][lane*(steps+1)]=inv.kineticEnergy;
   for(int s=0;s<steps;++s){auto begin=results[arm].observations.begin()+(s*3+lane)*(a.nq+a.nv);q.assign(begin,begin+a.nq);v.assign(begin+a.nq,begin+a.nq+a.nv);auto x=metalrobo::computeArticulatedInvariants(m,0,q,v,inv,ref);ok=ok&&x.succeeded()&&std::isfinite(inv.kineticEnergy);energies[arm][lane*(steps+1)+s+1]=inv.kineticEnergy;}
  }
 }
 for(int arm=0;arm<2;++arm)for(int lane=0;lane<3;++lane){auto base=lane*(steps+1);double start=energies[arm][base],end=energies[arm][base+steps];if(lane==0)ok=ok&&std::abs(start)<1e-12&&std::abs(end)<1e-12;else if(arm==0)ok=ok&&std::abs(end-start)<.005;else{ok=ok&&end<start-.001;for(int s=1;s<=steps;++s)ok=ok&&(energies[arm][base+s]-energies[arm][base+s-1])<=.0001;}}
 std::cout<<std::setprecision(12)<<"{\"fingerprint\":"<<comp.fingerprint<<",\"encodes\":"<<aero.encoded<<",\"initial_q\":[";for(size_t i=0;i<qs.size();++i){if(i)std::cout<<',';std::cout<<qs[i];}std::cout<<"],\"initial_v\":[";for(size_t i=0;i<vs.size();++i){if(i)std::cout<<',';std::cout<<vs[i];}std::cout<<"],\"arms\":[";
 for(int arm=0;arm<2;++arm){if(arm)std::cout<<',';std::cout<<"{\"ok\":"<<(d[arm].succeeded()?"true":"false")<<",\"successful_steps\":"<<d[arm].successfulStepCount<<",\"failed_steps\":"<<d[arm].failedStepCount<<",\"message\":"<<std::quoted(d[arm].message)<<",\"energies\":[";for(size_t i=0;i<energies[arm].size();++i){if(i)std::cout<<',';std::cout<<energies[arm][i];}std::cout<<"],\"trajectory\":[";for(size_t i=0;i<results[arm].observations.size();++i){if(i)std::cout<<',';std::cout<<results[arm].observations[i];}std::cout<<"]}";}
 std::cout<<"],\"criterion_pass\":"<<(ok?"true":"false")<<"}\n";return ok?0:2;
}catch(const std::exception&e){std::cerr<<"exception: "<<e.what()<<'\n';return 1;}}}
