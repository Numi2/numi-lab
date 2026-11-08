#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
constexpr unsigned N=20;
template<class T> bool eq(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
struct Capture { __strong id<MTLBuffer> qs; __strong id<MTLBuffer> vs; __strong id<MTLBuffer> nodes; __strong id<MTLBuffer> contacts; __strong id<MTLBuffer> statuses; unsigned nq=0,nv=0,nrod=0,stride=0,calls=0; };
static bool encodeCapture(void* x,const metalrobo::MetalWorldDevicePhysicsPass& p) noexcept {
 if(p.phase!=metalrobo::MetalWorldDevicePhysicsPhase::postCommit)return true;
 auto*c=static_cast<Capture*>(x);if(p.environmentCount!=1||p.controlStep>=N||p.physicsSubsteps!=1||!p.rodNodes||!p.contactConstraints||!p.contactStatuses)return false;
 id<MTLCommandBuffer> cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
 if(!c->qs){id<MTLDevice>d=cb.device;c->nq=p.nq;c->nv=p.nv;c->nrod=p.rodNodeCount;c->stride=p.contactConstraintStride;
  c->qs=[d newBufferWithLength:N*p.nq*sizeof(float) options:MTLResourceStorageModeShared];c->vs=[d newBufferWithLength:N*p.nv*sizeof(float) options:MTLResourceStorageModeShared];
  c->nodes=[d newBufferWithLength:N*p.rodNodeCount*sizeof(MRRodNodeStateGPU) options:MTLResourceStorageModeShared];
  c->contacts=[d newBufferWithLength:N*p.contactConstraintStride*sizeof(MRContactConstraintGPU) options:MTLResourceStorageModeShared];
  c->statuses=[d newBufferWithLength:N*sizeof(MRMetalWorldContactStatusGPU) options:MTLResourceStorageModeShared];
  if(!c->qs||!c->vs||!c->nodes||!c->contacts||!c->statuses)return false;}
 if(c->nq!=p.nq||c->nv!=p.nv||c->nrod!=p.rodNodeCount||c->stride!=p.contactConstraintStride)return false;
 auto b=[cb blitCommandEncoder];if(!b)return false;
 auto copy=[&](void* source,id<MTLBuffer> dest,NSUInteger size){[b copyFromBuffer:(__bridge id<MTLBuffer>)source sourceOffset:0 toBuffer:dest destinationOffset:p.controlStep*size size:size];};
 copy(p.q,c->qs,p.nq*sizeof(float));copy(p.v,c->vs,p.nv*sizeof(float));copy(p.rodNodes,c->nodes,p.rodNodeCount*sizeof(MRRodNodeStateGPU));copy(p.contactConstraints,c->contacts,p.contactConstraintStride*sizeof(MRContactConstraintGPU));copy(p.contactStatuses,c->statuses,sizeof(MRMetalWorldContactStatusGPU));[b endEncoding];c->calls++;return true;
}
static void abortCapture(void*,void*) noexcept {}
int main(int argc,char**argv){@autoreleasepool {
 if(argc!=3)return 2;int sweeps=std::atoi(argv[2]);
 auto m=metalrobo::makeFreeSphereEngineModel();m.world.gravityAndTimestep={0,-9.81f,0,.001f};m.defaultQ[1]=.0099f;m.defaultV[0]=.05f;m.defaultV[1]=0.f;
 m.shapes[1].dimensions={.01f,0,0,0};m.shapes[1].contactRestAndBoundingRadius={.001f,0,.01f,0};for(auto&x:m.materials){x.friction={.5f,.5f,0,0};x.response={0,0,0,0};}
 MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;std::vector<MRBodyStateGPU> scene{ground};
 metalrobo::DiscreteRodMaterial mat;mat.radius=.003;mat.density=1000;mat.youngModulus=1.e5;metalrobo::HeterogeneousRodProgram rod;rod.instanceId="sustained_mixed_rod";rod.model=metalrobo::makeStraightSutureRod(3,.04,mat);rod.defaultState=metalrobo::makeDiscreteElasticRodDefaultState(rod.model);for(auto&p:rod.defaultState.positions){p[0]-=.02;p[1]=.0228;}rod.collision.enableCCD=false;rod.collision.contactOffset=.001;rod.stepConfig.gravity={0,-9.81f,0};rod.stepConfig.timestep=.001;
 const std::vector<metalrobo::HeterogeneousWorldComponent> components{{&m,"body",scene}};const std::vector<metalrobo::HeterogeneousRodProgram> rods{rod};metalrobo::HeterogeneousWorld hw;metalrobo::EngineModelComposeConfig compose;compose.gravityAndTimestep={0,-9.81f,0,.001f};auto hd=metalrobo::composeHeterogeneousWorld(components,rods,hw,compose);if(!hd.succeeded()){std::cerr<<hd.message;return 3;}metalrobo::CompiledWorld world;auto cd=metalrobo::compileMetalWorld(hw,world);if(!cd.succeeded()){std::cerr<<cd.message;return 4;}
 std::vector<float> effort(N*world.nv(),0);metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=N,.initialQ=hw.model.defaultQ,.initialV=hw.model.defaultV,.efforts=effort,.initialSceneBodies=hw.defaultSceneBodies};
 metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.001f;cfg.physicsSubsteps=1;cfg.velocityIterations=sweeps;cfg.finalVelocityIterations=0;cfg.ccdMode=metalrobo::MetalWorldCCDMode::disabled;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;
 metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext ctx(owner);metalrobo::MetalWorldResult plain,recorded,repeated;auto plainDiag=ctx.run(world,batch,cfg,plain);
 Capture cap,cap2;cfg.devicePhysicsProgram={.context=&cap,.encode=encodeCapture,.abort=abortCapture,.fingerprint=0x534e415031ULL,.flags=metalrobo::MetalWorldDevicePhysicsRequiresRigidContactEvidence};auto rd=ctx.run(world,batch,cfg,recorded);
 cfg.devicePhysicsProgram.context=&cap2;auto rr=ctx.run(world,batch,cfg,repeated);
 auto sameSpan=[](const auto& v,const auto& span){using T=typename std::decay_t<decltype(v)>::value_type;return v.size()==span.size()&&(v.empty()||std::memcmp(v.data(),span.data(),v.size()*sizeof(T))==0);};
 bool finalRestored=eq(recorded.finalQ,hw.model.defaultQ)&&eq(recorded.finalV,hw.model.defaultV)&&eq(recorded.finalSceneBodies,hw.defaultSceneBodies)&&sameSpan(recorded.finalRodNodes,world.defaultRodNodes())&&sameSpan(recorded.finalRodEdges,world.defaultRodEdges());
 bool same=eq(plain.finalQ,recorded.finalQ)&&eq(plain.finalV,recorded.finalV)&&eq(plain.finalSceneBodies,recorded.finalSceneBodies)&&eq(plain.finalRodNodes,recorded.finalRodNodes)&&eq(plain.finalRodEdges,recorded.finalRodEdges)&&eq(plain.statuses,recorded.statuses)&&eq(plain.contactStatuses,recorded.contactStatuses);
 bool replay=eq(recorded.finalQ,repeated.finalQ)&&eq(recorded.finalV,repeated.finalV)&&eq(recorded.finalSceneBodies,repeated.finalSceneBodies)&&eq(recorded.finalRodNodes,repeated.finalRodNodes)&&eq(recorded.finalRodEdges,repeated.finalRodEdges)&&eq(recorded.statuses,repeated.statuses)&&eq(recorded.contactStatuses,repeated.contactStatuses);
 if(!cap.qs||cap.calls!=N||!cap2.qs||cap2.calls!=N){std::cerr<<"capture calls "<<cap.calls<<" "<<cap2.calls<<" diag="<<rd.message;return 5;}
 auto q=(float*)cap.qs.contents;auto v=(float*)cap.vs.contents;auto nodes=(MRRodNodeStateGPU*)cap.nodes.contents;auto contacts=(MRContactConstraintGPU*)cap.contacts.contents;auto status=(MRMetalWorldContactStatusGPU*)cap.statuses.contents;
 bool exactTrace=std::memcmp(cap.qs.contents,cap2.qs.contents,N*cap.nq*sizeof(float))==0&&std::memcmp(cap.vs.contents,cap2.vs.contents,N*cap.nv*sizeof(float))==0&&std::memcmp(cap.nodes.contents,cap2.nodes.contents,N*cap.nrod*sizeof(MRRodNodeStateGPU))==0&&std::memcmp(cap.statuses.contents,cap2.statuses.contents,N*sizeof(MRMetalWorldContactStatusGPU))==0;
 for(unsigned i=0;i<N;i++){unsigned count=std::min(status[i].activeContacts,cap.stride);exactTrace=exactTrace&&std::memcmp(contacts+i*cap.stride,(MRContactConstraintGPU*)cap2.contacts.contents+i*cap.stride,count*sizeof(MRContactConstraintGPU))==0;}
 if(recorded.statuses.size()!=N){std::cerr<<"world status size "<<recorded.statuses.size();return 6;}
 std::cout<<std::setprecision(10)<<"{\"plain_published\":"<<(plainDiag.published?"true":"false")<<",\"published\":"<<(rd.published?"true":"false")<<",\"success\":"<<rd.successfulStepCount<<",\"failed\":"<<rd.failedStepCount<<",\"first_code\":"<<rd.firstGPUStatusCode<<",\"final_restored\":"<<(finalRestored?"true":"false")<<",\"capture_noninvasive\":"<<(same?"true":"false")<<",\"repeat\":"<<(replay?"true":"false")<<",\"exact_trace\":"<<(exactTrace?"true":"false")<<",\"rod_masses\":[";
 for(size_t k=0;k<rod.model.nodeMasses.size();k++){if(k)std::cout<<",";std::cout<<rod.model.nodeMasses[k];}std::cout<<"],\"initial_q\":[";for(size_t k=0;k<hw.model.defaultQ.size();k++){if(k)std::cout<<",";std::cout<<hw.model.defaultQ[k];}std::cout<<"],\"initial_v\":[";for(size_t k=0;k<hw.model.defaultV.size();k++){if(k)std::cout<<",";std::cout<<hw.model.defaultV[k];}std::cout<<"],\"initial_rod_v\":[";for(size_t k=0;k<world.defaultRodNodes().size();k++){if(k)std::cout<<",";auto&z=world.defaultRodNodes()[k].velocity;std::cout<<"["<<z.x<<","<<z.y<<","<<z.z<<"]";}std::cout<<"],\"steps\":[";
 for(unsigned i=0;i<N;i++){if(i)std::cout<<",";auto &s=status[i];const float* prevQ=i?q+(i-1)*cap.nq:hw.model.defaultQ.data();const float* prevV=i?v+(i-1)*cap.nv:hw.model.defaultV.data();const MRRodNodeStateGPU* prevNodes=i?nodes+(i-1)*cap.nrod:world.defaultRodNodes().data();bool unchangedFromPrevious=std::memcmp(q+i*cap.nq,prevQ,cap.nq*sizeof(float))==0&&std::memcmp(v+i*cap.nv,prevV,cap.nv*sizeof(float))==0&&std::memcmp(nodes+i*cap.nrod,prevNodes,cap.nrod*sizeof(MRRodNodeStateGPU))==0;bool stepRestored=std::memcmp(q+i*cap.nq,hw.model.defaultQ.data(),cap.nq*sizeof(float))==0&&std::memcmp(v+i*cap.nv,hw.model.defaultV.data(),cap.nv*sizeof(float))==0&&std::memcmp(nodes+i*cap.nrod,world.defaultRodNodes().data(),cap.nrod*sizeof(MRRodNodeStateGPU))==0;std::cout<<"{\"unchanged_from_previous\":"<<(unchangedFromPrevious?"true":"false")<<",\"failure_tangent_diagnostic\":"<<s.residuals.z<<",\"step_restored\":"<<(stepRestored?"true":"false")<<",\"world_code\":"<<recorded.statuses[i].code<<",\"code\":"<<s.code<<",\"contacts\":"<<s.activeContacts<<",\"required\":"<<s.requiredConstraints<<",\"sweeps\":"<<s.solverIterations<<",\"normal_residual\":"<<s.residuals.y<<",\"q\":[";
 for(unsigned k=0;k<cap.nq;k++){if(k)std::cout<<",";std::cout<<q[i*cap.nq+k];}std::cout<<"],\"v\":[";for(unsigned k=0;k<cap.nv;k++){if(k)std::cout<<",";std::cout<<v[i*cap.nv+k];}std::cout<<"],\"rod_v\":[";for(unsigned k=0;k<cap.nrod;k++){if(k)std::cout<<",";auto&z=nodes[i*cap.nrod+k].velocity;std::cout<<"["<<z.x<<","<<z.y<<","<<z.z<<"]";}std::cout<<"],\"rows\":[";
 for(unsigned k=0;k<std::min(s.activeContacts,cap.stride);k++){if(k)std::cout<<",";auto&c=contacts[i*cap.stride+k];std::cout<<"{\"flags\":"<<c.flags<<",\"body_a\":"<<c.bodyA<<",\"body_b\":"<<c.bodyB<<",\"point\":["<<c.pointAndSeparation.x<<","<<c.pointAndSeparation.y<<","<<c.pointAndSeparation.z<<"],\"normal\":["<<c.normal.x<<","<<c.normal.y<<","<<c.normal.z<<"],\"tangent\":["<<c.tangent.x<<","<<c.tangent.y<<","<<c.tangent.z<<"],\"impulse\":["<<c.impulses.x<<","<<c.impulses.y<<","<<c.impulses.z<<"]}";}std::cout<<"]}";}
 std::cout<<"]}\n";return 0;
}}
