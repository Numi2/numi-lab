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
 metalrobo::CompiledWorld world;auto comp=metalrobo::compileMetalWorld(m,0,world);if(!comp.succeeded())throw std::runtime_error(comp.message);auto&a=m.articulations[0];constexpr int steps=960,lanes=3;
 std::array<std::string,4> names{"crow_left_elbow_flex","crow_left_wrist_flex","crow_right_elbow_flex","crow_right_wrist_flex"};std::array<unsigned,4> qi{},vi{};
 for(unsigned i=0;i<4;++i){auto it=std::find(m.dofNames.begin(),m.dofNames.end(),names[i]);if(it==m.dofNames.end())throw std::runtime_error("missing joint");auto&d=m.dofs.at(it-m.dofNames.begin());qi[i]=d.qIndex;vi[i]=d.vIndex;}
 std::vector<float> qs,vs,commands(steps*lanes*a.nv,0);for(int lane=0;lane<lanes;++lane){qs.insert(qs.end(),m.defaultQ.begin(),m.defaultQ.end());vs.insert(vs.end(),m.defaultV.begin(),m.defaultV.end());}
 for(int s=0;s<steps;++s)for(int lane=0;lane<lanes;++lane){float* cmd=&commands[(s*lanes+lane)*a.nv];for(auto&d:m.dofs)if((d.flags&MR_DOF_FLAG_DRIVE)&&d.qIndex<m.defaultQ.size()&&d.vIndex<a.nv)cmd[d.vIndex]=m.defaultQ[d.qIndex];float x=std::min(1.0f,(s+1.0f)/360.0f),smooth=x*x*(3-2*x);for(int j=0;j<4;++j)cmd[vi[j]]=(j%2==0?1.8f:1.575f)*smooth;}
 const float lens[3]={.08355912f,.09530571f,.09089033f};float total=lens[0]+lens[1]+lens[2];Surface surfaces[6]{};const uint32_t indices[6]={5,35,37,6,38,40};for(int i=0;i<6;++i)surfaces[i]={indices[i],i<3?1.0f:-1.0f,lens[i%3],.075f*lens[i%3]/total};
 Wind wind{{{0,0,0,1.225f},{-5,0,0,1.225f},{-10,0,0,1.225f}},3,41,0,0};id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError* error=nil;id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]] error:&error];if(!lib)throw std::runtime_error([[error description] UTF8String]);id<MTLFunction> fn=[lib newFunctionWithName:@"crow_articulated_wing_aero"];id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:fn error:&error];if(!pipeline)throw std::runtime_error([[error description] UTF8String]);
 Aero aero{};aero.pipeline=pipeline;aero.surfaces=[device newBufferWithBytes:surfaces length:sizeof(surfaces) options:MTLResourceStorageModeShared];aero.captured=[device newBufferWithLength:18*sizeof(MRABABodyWrenchGPU) options:MTLResourceStorageModeShared];aero.wind=wind;
 metalrobo::MetalWorldBatch b;b.environmentCount=lanes;b.controlStepCount=steps;b.initialQ=qs;b.initialV=vs;b.efforts=commands;
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=1.0f/240;cfg.physicsSubsteps=32;cfg.solverMode=metalrobo::MetalWorldSolverMode::temporalCone;cfg.actuationMode=metalrobo::MetalWorldActuationMode::implicitPositionDrive;cfg.captureContactEvidence=true;
 cfg.devicePhysicsProgram.context=&aero;cfg.devicePhysicsProgram.encode=encode;cfg.devicePhysicsProgram.abort=abortAero;cfg.devicePhysicsProgram.flags=metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches;cfg.devicePhysicsProgram.fingerprint=0x81dbe371a5419b65ULL;
 metalrobo::MetalWorldConfig settings;settings.metallibPath=argv[1];settings.preferParallelABA=false;metalrobo::MetalWorldContext context(settings);metalrobo::MetalWorldResult out;auto d=context.run(world,b,cfg,out);
 bool finite=true;for(float x:out.observations)finite=finite&&std::isfinite(x);bool ok=d.succeeded()&&d.dispatched&&d.published&&d.successfulStepCount==lanes*steps&&d.failedStepCount==0&&aero.encoded==steps*32&&finite;
 if(out.finalQ.size()==lanes*a.nq){for(int lane=0;lane<lanes;++lane)for(int j=0;j<4;++j)ok=ok&&std::abs(out.finalQ[lane*a.nq+qi[j]]-(j%2==0?1.8f:1.575f))<.01f;}else ok=false;
 if(out.observations.size()==steps*lanes*(a.nq+a.nv)){for(int lane=0;lane<lanes;++lane)for(int j=0;j<4;++j){const size_t k=((steps-1)*lanes+lane)*(a.nq+a.nv)+a.nq+vi[j];ok=ok&&std::abs(out.observations[k])<.05f;}}else ok=false;
 std::cout<<std::setprecision(12)<<"{\"fingerprint\":"<<comp.fingerprint<<",\"encodes\":"<<aero.encoded<<",\"ok\":"<<(d.succeeded()?"true":"false")<<",\"published\":"<<(d.published?"true":"false")<<",\"steps\":"<<d.successfulStepCount<<",\"failed\":"<<d.failedStepCount<<",\"first_failure_step\":"<<d.firstFailingControlStep<<",\"status_code\":"<<d.firstGPUStatusCode<<",\"message\":"<<std::quoted(d.message)<<",\"nq\":"<<a.nq<<",\"nv\":"<<a.nv<<",\"qindices\":[";for(int j=0;j<4;++j){if(j)std::cout<<',';std::cout<<qi[j];}std::cout<<"],\"final_q\":[";for(size_t i=0;i<out.finalQ.size();++i){if(i)std::cout<<',';std::cout<<out.finalQ[i];}std::cout<<"],\"trajectory\":[";for(size_t i=0;i<out.observations.size();++i){if(i)std::cout<<',';std::cout<<out.observations[i];}std::cout<<"],\"contact_statuses\":[";for(size_t i=0;i<out.contactStatuses.size();++i){auto&s=out.contactStatuses[i];if(i)std::cout<<',';std::cout<<"{\"code\":"<<s.code<<",\"step\":"<<s.controlStep<<",\"substep\":"<<s.physicsSubstep<<",\"pairs\":"<<s.activePairs<<",\"contacts\":"<<s.activeContacts<<",\"first_failed_pair\":"<<s.firstFailingPair<<'}';}std::cout<<"],\"tracking_screen_pass\":"<<(ok?"true":"false")<<"}\n";return ok?0:2;
 }catch(const std::exception&e){std::cerr<<"exception: "<<e.what()<<'\n';return 1;}}}
