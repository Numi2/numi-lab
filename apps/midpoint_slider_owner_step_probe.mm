// Test a real, narrowly admitted MetalWorld slider midpoint transaction.
// Matter contact coupling is not exercised by this owner-only fixture.
#define main numi_slider_baseline_entry
#include "midpoint_slider_rigid_baseline.cpp"
#undef main
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
struct OwnerProbe {float acceleration=0;float expectedMass=0;float expectedGravity=0;id<MTLBuffer> increment=nil;id<MTLBuffer> output=nil;id<MTLBuffer> stageQ=nil;id<MTLBuffer> stageBodies=nil;bool pre=false,post=false,skip=false,inject=false,legacyPublish=false,poison=false;std::uint32_t statusFlags=0;};
static bool encodeOwner(void* opaque,const metalrobo::MetalWorldDevicePhysicsPass& pass){
 auto& p=*static_cast<OwnerProbe*>(opaque);
 if(!pass.sliderMidpointMechanicalDataValid ||
    !std::isfinite(pass.sliderMidpointMassKg) || pass.sliderMidpointMassKg!=p.expectedMass ||
    pass.sliderMidpointGravityForceNewtons != pass.sliderMidpointMassKg * p.expectedGravity)return false;
 if(pass.phase==metalrobo::MetalWorldDevicePhysicsPhase::postCommit){
  p.post=true;
  if(p.inject){
   id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
   id<MTLBuffer> dest=(__bridge id<MTLBuffer>)pass.environmentStatuses;
   MRMetalWorldStatusGPU failure{};failure.code=MR_STEP_DID_NOT_CONVERGE;failure.environment=0;failure.flags=p.statusFlags;
   id<MTLBuffer> source=[command.commandQueue.device newBufferWithBytes:&failure length:sizeof(failure) options:MTLResourceStorageModeShared];
   if(!source||!dest||dest.length<sizeof(failure))return false;
   id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];if(!blit)return false;
   [blit copyFromBuffer:source sourceOffset:0 toBuffer:dest destinationOffset:0 size:sizeof(failure)];[blit endEncoding];
   [command addCompletedHandler:^(id<MTLCommandBuffer> done){(void)done;(void)source;}];
  }
  return true;
 }
 p.pre=true;if(p.skip)return true;
 id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
 id<MTLDevice> device=command.commandQueue.device;
 p.increment=[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
 p.output=[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
 if(!p.increment||!p.output)return false;
 *static_cast<float*>(p.increment.contents)=p.poison ? std::numeric_limits<float>::quiet_NaN() : .5f*pass.timestepSeconds*p.acceleration;
 p.stageQ=[device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
 p.stageBodies=[device newBufferWithLength:pass.bodyStateStride*sizeof(MRBodyStateGPU) options:MTLResourceStorageModeShared];
 if(!p.stageQ||!p.stageBodies)return false;
 metalrobo::MetalWorldCoupledCandidateQuery stage{};
 stage.input=(__bridge void*)p.increment;stage.candidateQ=(__bridge void*)p.stageQ;
 stage.candidateBodies=(__bridge void*)p.stageBodies;stage.generalizedVectorStride=1;
 stage.candidateQStride=1;stage.candidateBodyStride=pass.bodyStateStride;
 stage.operation=metalrobo::MetalWorldCoupledCandidateOperation::candidateHalfStepKinematics;
 if(!pass.encodeCoupledCandidate(pass.coupledCandidateContext,pass,stage))return false;
 metalrobo::MetalWorldCoupledCandidateQuery q{};q.input=(__bridge void*)p.increment;
 q.output=(__bridge void*)p.output;q.generalizedVectorStride=1;
 q.operation=p.legacyPublish ? metalrobo::MetalWorldCoupledCandidateOperation::publishCandidate : metalrobo::MetalWorldCoupledCandidateOperation::publishSliderMidpoint;
 return pass.encodeCoupledCandidate(pass.coupledCandidateContext,pass,q);
}
static void abortOwner(void*,void*){}
static metalrobo::EngineModel slider(float h,float v,bool rotate=false,float mass=.01f,float gravity=-9.81f){
 metalrobo::EngineModel model;model.name="midpoint_slider_owner_step";
 model.world.abiVersion=MR_ENGINE_ABI_VERSION;model.world.bodyCount=2;model.world.articulationCount=1;
 model.world.jointCount=1;model.world.nq=1;model.world.nv=1;model.world.pairCapacity=1;
 model.world.contactCapacity=1;model.world.constraintCapacity=1;model.world.islandCapacity=1;
 model.world.solverType=MR_SOLVER_REFERENCE_FP64;model.world.frictionConeType=MR_FRICTION_CONE_ELLIPTIC;
 model.world.gravityAndTimestep={0,0,gravity,h};model.world.solverScales={1e-8f,1e-9f,2,1e-4f};
 MRArticulationGPU a{};a.rootBody=0;a.rootType=MR_ROOT_FIXED;a.firstBody=0;a.bodyCount=2;a.firstJoint=0;
 a.jointCount=1;a.qOffset=0;a.nq=1;a.vOffset=0;a.nv=1;model.articulations.push_back(a);
 model.bodies.push_back(body(MR_INVALID_INDEX,MR_INVALID_INDEX,1));model.bodies.push_back(body(0,0,mass));
 model.bodies[1].dampingAndSpeedLimits={0,0,0,0};
 MRJointDescriptorGPU j{};j.parentBody=0;j.childBody=1;j.jointType=rotate?MR_JOINT_REVOLUTE:MR_JOINT_PRISMATIC;
 j.qOffset=0;j.nq=1;j.vOffset=0;j.nv=1;j.axis0={0,0,1,0};j.parentRotation={0,0,0,1};j.childRotation={0,0,0,1};model.joints.push_back(j);
 MRDofPropertiesGPU d{};d.articulationIndex=0;d.jointIndex=0;d.qIndex=0;d.vIndex=0;model.dofs.push_back(d);
 model.defaultQ={0};model.defaultV={v};return model;
}
static metalrobo::MetalWorldDiagnostics run(const std::string& shader,metalrobo::EngineModel model,float h,float force,OwnerProbe& probe,metalrobo::MetalWorldResult& result,float effort=0,metalrobo::MetalWorldContext* existing=nullptr){
 std::string reason;require(model.valid(&reason),"invalid slider: "+reason);metalrobo::CompiledWorld world;
 auto cooked=metalrobo::compileMetalWorld(model,0,world);require(cooked.succeeded(),"compile: "+cooked.message);
 std::vector<float> efforts(1,effort);metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=efforts};
 metalrobo::MetalWorldStepConfig config{};config.timestepSeconds=h;config.physicsSubsteps=1;
 config.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;config.ccdMode=metalrobo::MetalWorldCCDMode::disabled;
 config.matrixFreeArticulatedContact=false;config.streamedArticulatedContactResponses=false;config.captureContactEvidence=false;
 probe.expectedMass=model.bodies[1].massAndInverseMass.x;
 probe.expectedGravity=model.world.gravityAndTimestep.z;
 probe.acceleration=probe.expectedGravity+force/probe.expectedMass;config.devicePhysicsProgram.context=&probe;
 config.devicePhysicsProgram.encode=&encodeOwner;config.devicePhysicsProgram.abort=&abortOwner;
 config.devicePhysicsProgram.fingerprint=0x5ee124f0a9bdfc10ull;
 config.devicePhysicsProgram.flags=metalrobo::MetalWorldDevicePhysicsOwnsCoupledCandidate|metalrobo::MetalWorldDevicePhysicsOwnsSliderMidpoint;
 config.devicePhysicsProgram.coupledCandidatePointCapacity=1;
 metalrobo::MetalWorldConfig c{};c.metallibPath=shader;metalrobo::MetalWorldContext context(c);
 return (existing ? *existing : context).run(world,batch,config,result);
}
int main(int argc,char**argv){@autoreleasepool{try{
 require(argc==2,"usage: midpoint_slider_owner_step_probe METALLIB");
 double maxQ=0,maxV=0,maxStageV=0;for(float gravity:{-9.81f,-3.7f,0.f})for(float mass:{.01f,.025f})for(float h:{.001f,.002f})for(float v:{.02f,-.03f})for(float force:{0.f,.015f}){
  OwnerProbe p;metalrobo::MetalWorldResult result;auto d=run(argv[1],slider(h,v,false,mass,gravity),h,force,p,result);
  require(d.succeeded(),"midpoint owner step: "+d.message);require(p.pre&&p.post&&result.finalQ.size()==1&&result.finalV.size()==1,"missing midpoint output");
  require(result.layout.devicePhysicsFingerprint!=0x5ee124f0a9bdfc10ull,"midpoint execution fingerprint aliases original");
  const double acceleration=double(gravity)+double(force)/double(mass);
  const double expectedQ=double(h)*double(v)+.5*double(h)*double(h)*acceleration;
  const double expectedV=double(v)+double(h)*acceleration;
  const double expectedStageV=double(v)+.5*double(h)*acceleration;
  const auto stageBody=static_cast<const MRBodyStateGPU*>(p.stageBodies.contents)[1];
  if(std::abs(stageBody.linearVelocityAndInverseMass.z-expectedStageV)>=1e-6)
   std::cerr<<"candidate stage velocity actual="<<stageBody.linearVelocityAndInverseMass.z<<" expected="<<expectedStageV<<"\n";
  require(std::abs(stageBody.linearVelocityAndInverseMass.z-expectedStageV)<1e-6,"candidate stage body velocity missing");
  maxStageV=std::max(maxStageV,std::abs(stageBody.linearVelocityAndInverseMass.z-expectedStageV));
  maxQ=std::max(maxQ,std::abs(result.finalQ[0]-expectedQ));maxV=std::max(maxV,std::abs(result.finalV[0]-expectedV));
 }
 require(maxQ<1e-8 && maxV<1e-6,"midpoint full-step equations failed");
 OwnerProbe bad;metalrobo::MetalWorldResult ignored;auto rotating=run(argv[1],slider(.001f,.02f,true),.001f,0,bad,ignored);
 require(!rotating.succeeded()&&!bad.pre,"rotating joint not rejected before callback");
 OwnerProbe loaded;auto nonzero=run(argv[1],slider(.001f,.02f),.001f,0,loaded,ignored,.1f);
 require(!nonzero.succeeded()&&!loaded.pre,"extra effort not rejected before callback");
 OwnerProbe missing;missing.skip=true;auto noCandidate=run(argv[1],slider(.001f,.02f),.001f,0,missing,ignored);
 require(!noCandidate.succeeded(),"missing stage candidate was accepted");
 OwnerProbe poisoned;poisoned.poison=true;metalrobo::MetalWorldResult poisonedResult;
 auto poisonedStatus=run(argv[1],slider(.001f,.02f),.001f,0,poisoned,poisonedResult);
 std::cerr<<"nonfinite status="<<poisonedStatus.firstGPUStatusCode<<" succeeded="<<poisonedStatus.succeeded()<<" qsize="<<poisonedResult.finalQ.size()<<" vsize="<<poisonedResult.finalV.size()<<" q="<<(poisonedResult.finalQ.empty()?999:poisonedResult.finalQ[0])<<" v="<<(poisonedResult.finalV.empty()?999:poisonedResult.finalV[0])<<" message="<<poisonedStatus.message<<"\n";
 require(!poisonedStatus.succeeded()&&poisonedStatus.firstGPUStatusCode==MR_STEP_NONFINITE_RESULT&&
         poisonedResult.finalQ==std::vector<float>{0}&&poisonedResult.finalV==std::vector<float>{.02f},
         "nonfinite midpoint stage was not rejected and restored");
 OwnerProbe legacy;legacy.legacyPublish=true;auto oldPublish=run(argv[1],slider(.001f,.02f),.001f,0,legacy,ignored);
 require(!oldPublish.succeeded(),"legacy publisher accepted in midpoint owner");
 metalrobo::MetalWorldConfig reusableConfig{};reusableConfig.metallibPath=argv[1];metalrobo::MetalWorldContext reusable(reusableConfig);
 auto initial=slider(.001f,.02f);initial.defaultQ[0]=.017f;OwnerProbe goodProbe;metalrobo::MetalWorldResult good;
 auto goodStatus=run(argv[1],initial,.001f,.015f,goodProbe,good,0,&reusable);require(goodStatus.succeeded(),"retry baseline failed");
 OwnerProbe fault;fault.inject=true;fault.statusFlags=good.layout.dispatch.flags;metalrobo::MetalWorldResult rejected;
 auto failed=run(argv[1],initial,.001f,.015f,fault,rejected,0,&reusable);
 std::cerr<<"fault pre="<<fault.pre<<" post="<<fault.post<<" succeeded="<<failed.succeeded()<<" steps="<<failed.failedStepCount<<" code="<<failed.firstGPUStatusCode<<" message="<<failed.message<<"\n";
 require(fault.pre&&fault.post&&!failed.succeeded()&&failed.failedStepCount==1u&&failed.firstGPUStatusCode==MR_STEP_DID_NOT_CONVERGE,"late failure category not retained");
 auto sameBytes=[](const auto& a,const auto& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);};
 require(sameBytes(rejected.finalQ,initial.defaultQ)&&sameBytes(rejected.finalV,initial.defaultV),"late failure did not restore slider state");
 auto restored=initial;restored.defaultQ=rejected.finalQ;restored.defaultV=rejected.finalV;
 OwnerProbe retryProbe;metalrobo::MetalWorldResult retry;auto retried=run(argv[1],restored,.001f,.015f,retryProbe,retry,0,&reusable);
 require(retried.succeeded()&&sameBytes(retry.finalQ,good.finalQ)&&sameBytes(retry.finalV,good.finalV),"restored slider retry differs from accepted state");
 std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.slider-owner-midpoint-step.v1\",\"accepted_steps\":48,\"owner_mass_and_gravity_metadata_checked\":true,\"max_position_error_m\":"<<maxQ<<",\"max_velocity_error_mps\":"<<maxV<<",\"max_stage_velocity_error_mps\":"<<maxStageV<<",\"rotating_joint_rejected\":true,\"extra_effort_rejected\":true,\"missing_candidate_rejected\":true,\"late_rejection_and_retry_exact\":true,\"legacy_publish_rejected\":true,\"nonfinite_stage_rejected_restored\":true,\"scope\":\"MetalWorld only; no Matter contact\"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
