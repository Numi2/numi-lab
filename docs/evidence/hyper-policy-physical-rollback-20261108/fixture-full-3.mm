#include "metalrobo/RunProgram.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalHyperPolicy.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace metalrobo;
template<class T> void req(const T& d,const char* m){if(!d.succeeded())throw std::runtime_error(std::string(m)+": "+d.message);}
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
constexpr unsigned E=2,N=30;
struct Capture {
 MetalHyperPolicyRuntime* policy;
 MetalWorldDeviceActionProgram inner;
 __strong id<MTLBuffer> phases;
 __strong id<MTLBuffer> actions;
 __strong id<MTLBuffer> resets;
 __strong id<MTLBuffer> world;
 __strong id<MTLBuffer> contact;
 unsigned actionCount;
};
static bool captureState(Capture* c,void* command,unsigned step,unsigned stage){
#ifdef DIAG
 const size_t i=(step*3+stage)*E;
 return c->policy->diagnosticSnapshot(command,(__bridge void*)c->phases,(__bridge void*)c->actions,
  i*sizeof(MRHyperPolicyPhaseStateGPU),i*c->actionCount*sizeof(float),E);
#else
 (void)c;(void)command;(void)step;(void)stage;return true;
#endif
}
static bool actionEncode(void* x,const MetalWorldDeviceActionPass& p) noexcept {
 auto*c=static_cast<Capture*>(x);return c->inner.encode(c->inner.context,p) && captureState(c,p.commandBuffer,p.controlStep,1);
}
static bool actionAfter(void* x,const MetalWorldDeviceObservationStepResult& p) noexcept {
 auto*c=static_cast<Capture*>(x);return !c->inner.afterStep || c->inner.afterStep(c->inner.context,p);
}
static bool before(void* x,const MetalWorldDeviceObservationPass& p){
 auto*c=static_cast<Capture*>(x);id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
 auto b=[cb blitCommandEncoder];if(!b)return false;id<MTLBuffer> masks=(__bridge id<MTLBuffer>)p.resetMasks;
 [b copyFromBuffer:masks sourceOffset:p.controlStep*E*sizeof(unsigned) toBuffer:c->resets destinationOffset:p.controlStep*E*sizeof(unsigned) size:E*sizeof(unsigned)];[b endEncoding];
 return p.controlStep==0 || captureState(c,p.commandBuffer,p.controlStep,0);
}
static bool after(void* x,const MetalWorldDeviceObservationStepResult& p){
 auto*c=static_cast<Capture*>(x);id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
 auto b=[cb blitCommandEncoder];if(!b)return false;
 id<MTLBuffer> w=(__bridge id<MTLBuffer>)p.worldStatuses;id<MTLBuffer> contact=(__bridge id<MTLBuffer>)p.contactStatuses;
 [b copyFromBuffer:w sourceOffset:0 toBuffer:c->world destinationOffset:p.controlStep*E*sizeof(MRMetalWorldStatusGPU) size:E*sizeof(MRMetalWorldStatusGPU)];
 [b copyFromBuffer:contact sourceOffset:0 toBuffer:c->contact destinationOffset:p.controlStep*E*sizeof(MRMetalWorldContactStatusGPU) size:E*sizeof(MRMetalWorldContactStatusGPU)];[b endEncoding];
 return captureState(c,p.commandBuffer,p.controlStep,2);
}
int main(int argc,char**argv){try{require(argc==2,"shader path");
#include "setup.inc"
 MetalHyperPolicyRuntime policy;MetalHyperPolicyConfiguration pc;pc.metallibPath=argv[1];req(createMetalHyperPolicyRuntime(cp,t,pc,policy),"create policy");
 MetalWorldContext world;MetalWorldResidentState state;MetalWorldStepConfig cfg;cfg.timestepSeconds=.02f;cfg.physicsSubsteps=4;cfg.actuationMode=MetalWorldActuationMode::implicitPositionDrive;cfg.taskProgram=t;cfg.deviceActionProgram=policy.actionProgram();cfg.ccdMode=MetalWorldCCDMode::disabled;
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();auto alloc=[&](size_t size){return [dev newBufferWithLength:size options:MTLResourceStorageModeShared];};
 Capture c{&policy,cfg.deviceActionProgram,alloc(N*3*E*sizeof(MRHyperPolicyPhaseStateGPU)),alloc(N*3*E*l.actionCount*sizeof(float)),alloc(N*E*sizeof(unsigned)),alloc(N*E*sizeof(MRMetalWorldStatusGPU)),alloc(N*E*sizeof(MRMetalWorldContactStatusGPU)),l.actionCount};
 require(c.phases&&c.actions&&c.resets&&c.world&&c.contact,"capture allocations");
 cfg.deviceActionProgram.context=&c;cfg.deviceActionProgram.encode=&actionEncode;cfg.deviceActionProgram.afterStep=&actionAfter;
 cfg.deviceObservationProgram={.context=&c,.encode=&before,.begin=[](void*,void*)noexcept{return true;},.finish=[](void*,void*,bool)noexcept{},.afterStep=&after};
 std::vector<float> q(E*run.world().nq()),v(E*run.world().nv(),0);std::vector<MRBodyStateGPU> scene;
 for(unsigned e=0;e<E;++e){std::copy(run.model().defaultQ.begin(),run.model().defaultQ.end(),q.begin()+e*run.world().nq());scene.insert(scene.end(),run.defaultSceneBodies().begin(),run.defaultSceneBodies().end());}
 require(scene.size()==8,"scene size");scene[0].position.z+=.15f;scene[3].position.z+=2;scene[4].position.z+=2;scene[7].position.z+=2;
 std::vector<unsigned> reset(N*E,0);MetalWorldBatch batch{.environmentCount=E,.controlStepCount=N,.initialQ=q,.initialV=v,.policyRevision=1,.resetMasks=reset,.initialSceneBodies=scene};
 MetalWorldSubmission sub;auto d=world.initializeResidentState(run.world(),batch,cfg,state,sub);req(d,"submit");MetalWorldResult out;d=sub.wait(out);require(d.published,"expected published outcome");
 std::vector<float> phase,action,lat,log;require(policy.copyRolloutTrace(N,E,phase,action,lat,log),"trace");
 std::cout<<std::setprecision(10)<<"{\"published\":"<<d.published<<",\"success\":"<<d.successfulStepCount<<",\"failed\":"<<d.failedStepCount<<",\"host\":"<<static_cast<int>(d.status)<<",\"samples\":[";
 auto* masks=static_cast<unsigned*>(c.resets.contents);auto* ws=static_cast<MRMetalWorldStatusGPU*>(c.world.contents);auto* cs=static_cast<MRMetalWorldContactStatusGPU*>(c.contact.contents);auto* ps=static_cast<MRHyperPolicyPhaseStateGPU*>(c.phases.contents);auto* acts=static_cast<float*>(c.actions.contents);
 for(unsigned step=0;step<N;++step)for(unsigned e=0;e<E;++e){if(step||e)std::cout<<",";size_t i=step*E+e,b=(step*3)*E+e,k=(step*3+1)*E+e,a=(step*3+2)*E+e;std::cout<<"{\"step\":"<<step<<",\"env\":"<<e<<",\"reset\":"<<masks[i]<<",\"world\":"<<ws[i].code<<",\"contact\":"<<cs[i].code<<",\"trace\":"<<phase[i]<<",\"termination\":"<<(i<out.transitions.size()?out.transitions[i].termination.z:999);
#ifdef DIAG
 std::cout<<",\"pre_phase\":"<<ps[b].phase.x<<",\"post_phase\":"<<ps[a].phase.x<<",\"pre_action\":"<<acts[b*l.actionCount]<<",\"post_action\":"<<acts[a*l.actionCount];
 auto record=[](const MRHyperPolicyPhaseStateGPU &state) {auto *bytes=reinterpret_cast<const unsigned char*>(&state);for(size_t j=0;j<sizeof(state);++j)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(bytes[j]);std::cout<<std::dec<<std::setfill(' ');};
 std::cout<<",\"pre_record\":\"";record(ps[b]);std::cout<<"\",\"post_record\":\"";record(ps[a]);std::cout<<"\",\"candidate_record\":\"";record(ps[k]);std::cout<<"\",\"candidate_actions\":[";for(unsigned j=0;j<l.actionCount;++j){if(j)std::cout<<",";std::cout<<acts[k*l.actionCount+j];}std::cout<<"],\"pre_actions\":[";for(unsigned j=0;j<l.actionCount;++j){if(j)std::cout<<",";std::cout<<acts[b*l.actionCount+j];}std::cout<<"],\"post_actions\":[";for(unsigned j=0;j<l.actionCount;++j){if(j)std::cout<<",";std::cout<<acts[a*l.actionCount+j];}std::cout<<"]";
#endif
 std::cout<<"}";}
 auto floats=[](const char* name,const std::vector<float>&a){std::cout<<"],\""<<name<<"\":[";for(size_t i=0;i<a.size();++i){if(i)std::cout<<",";std::cout<<a[i];}};floats("q",out.finalQ);floats("v",out.finalV);floats("actor",out.actorObservations);floats("trace_phase",phase);floats("trace_action",action);
 std::cout<<"],\"scene_bytes\":\"";auto* bytes=reinterpret_cast<const unsigned char*>(out.finalSceneBodies.data());for(size_t i=0;i<out.finalSceneBodies.size()*sizeof(MRBodyStateGPU);++i)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(bytes[i]);std::cout<<std::dec<<"\"}\n";
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
