// Exercise MetalWorld borrowed half-step kinematics on an actual slider owner.
// This is an intermediate owner contract, not a coupled Matter transaction.
#define main numi_slider_baseline_entry
#include "midpoint_slider_rigid_baseline.cpp"
#undef main
#include <limits>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct CandidateProbe {
 id<MTLBuffer> increment=nil;
 id<MTLBuffer> halfQ=nil;
 id<MTLBuffer> fullQ=nil;
 id<MTLBuffer> halfBodies=nil;
 id<MTLBuffer> fullBodies=nil;
 bool pre=false,post=false;
};
static bool candidateProbeEncode(void* opaque,const metalrobo::MetalWorldDevicePhysicsPass& pass){
 auto& p=*static_cast<CandidateProbe*>(opaque);
 if(pass.phase==metalrobo::MetalWorldDevicePhysicsPhase::postCommit){p.post=true;return true;}
 if(pass.environmentCount!=1 || pass.nv!=1 || pass.qStride!=1 || pass.bodyStateStride<2 ||
    pass.encodeCoupledCandidate==nullptr || pass.coupledCandidateContext==nullptr)return false;
 id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
 id<MTLDevice> device=command.commandQueue.device;
 p.increment=[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
 p.halfQ=[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
 p.fullQ=[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
 p.halfBodies=[device newBufferWithLength:pass.bodyStateStride*sizeof(MRBodyStateGPU) options:MTLResourceStorageModeShared];
 p.fullBodies=[device newBufferWithLength:pass.bodyStateStride*sizeof(MRBodyStateGPU) options:MTLResourceStorageModeShared];
 if(!p.increment||!p.halfQ||!p.fullQ||!p.halfBodies||!p.fullBodies)return false;
 *static_cast<float*>(p.increment.contents)=.007f;
 metalrobo::MetalWorldCoupledCandidateQuery q{};
 q.input=(__bridge void*)p.increment;q.candidateQ=(__bridge void*)p.halfQ;
 q.candidateBodies=(__bridge void*)p.halfBodies;q.generalizedVectorStride=1;
 q.candidateQStride=1;q.candidateBodyStride=pass.bodyStateStride;
 q.operation=metalrobo::MetalWorldCoupledCandidateOperation::candidateHalfStepKinematics;
 if(!pass.encodeCoupledCandidate(pass.coupledCandidateContext,pass,q))return false;
 auto badPass=pass;badPass.timestepSeconds=std::numeric_limits<float>::denorm_min();
 if(pass.encodeCoupledCandidate(pass.coupledCandidateContext,badPass,q))return false;
 q.candidateQ=(__bridge void*)p.fullQ;q.candidateBodies=(__bridge void*)p.fullBodies;
 q.operation=metalrobo::MetalWorldCoupledCandidateOperation::candidateKinematics;
 if(!pass.encodeCoupledCandidate(pass.coupledCandidateContext,pass,q))return false;
 p.pre=true;return true;
}
static void candidateProbeAbort(void*,void*){}
int main(int argc,char**argv){@autoreleasepool{try{
 require(argc==2,"usage: midpoint_slider_candidate_owner_probe METALLIB");
 constexpr float h=.001f,v0=.02f,g=-9.81f;
 metalrobo::EngineModel model;model.name="midpoint_slider_candidate_owner_probe";
 model.world.abiVersion=MR_ENGINE_ABI_VERSION;model.world.bodyCount=2;model.world.articulationCount=1;
 model.world.jointCount=1;model.world.nq=1;model.world.nv=1;model.world.pairCapacity=1;
 model.world.contactCapacity=1;model.world.constraintCapacity=1;model.world.islandCapacity=1;
 model.world.solverType=MR_SOLVER_REFERENCE_FP64;model.world.frictionConeType=MR_FRICTION_CONE_ELLIPTIC;
 model.world.gravityAndTimestep={0,0,g,h};model.world.solverScales={1e-8f,1e-9f,2,1e-4f};
 MRArticulationGPU a{};a.rootBody=0;a.rootType=MR_ROOT_FIXED;a.firstBody=0;a.bodyCount=2;a.firstJoint=0;
 a.jointCount=1;a.qOffset=0;a.nq=1;a.vOffset=0;a.nv=1;model.articulations.push_back(a);
 model.bodies.push_back(body(MR_INVALID_INDEX,MR_INVALID_INDEX,1));model.bodies.push_back(body(0,0,.01f));
 MRJointDescriptorGPU j{};j.parentBody=0;j.childBody=1;j.jointType=MR_JOINT_PRISMATIC;
 j.qOffset=0;j.nq=1;j.vOffset=0;j.nv=1;j.axis0={0,0,1,0};j.parentRotation={0,0,0,1};j.childRotation={0,0,0,1};model.joints.push_back(j);
 MRDofPropertiesGPU d{};d.articulationIndex=0;d.jointIndex=0;d.qIndex=0;d.vIndex=0;model.dofs.push_back(d);
 model.defaultQ={0};model.defaultV={v0};std::string reason;require(model.valid(&reason),"invalid slider: "+reason);
 metalrobo::CompiledWorld world;auto cooked=metalrobo::compileMetalWorld(model,0,world);require(cooked.succeeded(),"compile: "+cooked.message);
 std::vector<float> effort(1,0);metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=effort};
 CandidateProbe probe;metalrobo::MetalWorldStepConfig config{};config.timestepSeconds=h;config.physicsSubsteps=1;
 config.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;config.matrixFreeArticulatedContact=false;config.streamedArticulatedContactResponses=false;config.captureContactEvidence=false;
 config.devicePhysicsProgram.context=&probe;config.devicePhysicsProgram.encode=&candidateProbeEncode;
 config.devicePhysicsProgram.abort=&candidateProbeAbort;config.devicePhysicsProgram.fingerprint=0x74248f3c94b042e1ull;
 config.devicePhysicsProgram.flags=metalrobo::MetalWorldDevicePhysicsOwnsCoupledCandidate;
 config.devicePhysicsProgram.coupledCandidatePointCapacity=1;
 metalrobo::MetalWorldConfig ownerConfig{};ownerConfig.metallibPath=argv[1];metalrobo::MetalWorldContext context(ownerConfig);
 metalrobo::MetalWorldResult result;auto run=context.run(world,batch,config,result);
 require(run.succeeded(),"run: "+run.message);require(probe.pre&&probe.post,"device physics phases missing");
 require(result.finalQ.size()==1 && result.finalV.size()==1,"missing owner state");
 const float qHalf=*static_cast<const float*>(probe.halfQ.contents),qFull=*static_cast<const float*>(probe.fullQ.contents);
 const auto halfBody=static_cast<const MRBodyStateGPU*>(probe.halfBodies.contents)[1];
 const auto fullBody=static_cast<const MRBodyStateGPU*>(probe.fullBodies.contents)[1];
 const double stageExpected=.5*double(h)*(double(v0)+double(.007f));
 const double fullExpected=double(h)*(double(v0)+double(.007f));
 const double ownerExpected=double(h)*(double(v0)+double(h)*double(g));
 require(std::abs(qHalf-stageExpected)<1e-9,"incorrect half-step candidate");
 require(std::abs(qFull-fullExpected)<1e-9,"incorrect full-step candidate");
 require(std::abs(halfBody.position.z-stageExpected)<1e-9 &&
         std::abs(fullBody.position.z-fullExpected)<1e-9,"incorrect projected body position");
 require(std::abs(result.finalQ[0]-ownerExpected)<1e-8,"scratch query changed owner integration");
 std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.slider-owner-half-step-candidate.v1\",\"half_q\":"<<qHalf<<",\"full_q\":"<<qFull<<",\"half_body_z\":"<<halfBody.position.z<<",\"full_body_z\":"<<fullBody.position.z<<",\"underflow_rejected\":true"<<",\"owner_final_q\":"<<result.finalQ[0]<<",\"owner_final_v\":"<<result.finalV[0]<<",\"scope\":\"borrowed scratch kinematics; not a coupled midpoint transaction\"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
