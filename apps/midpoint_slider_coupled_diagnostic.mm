// Diagnostic-only native FEM / genuinely reacting MetalWorld slider.
// Full accuracy, conservation and rollback gates are assessed separately.
#define main numi_plane_contact_diagnostic_main
#include "plane_contact_probe.mm"
#undef main
#include "numi/matter/metal_world.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "midpoint_slider_snapshot.hpp"

struct SliderObservation {
 metalrobo::MetalWorldDevicePhysicsProgram inner{};
 id<MTLBuffer> wrench=nil, support=nil, candidateQ=nil, candidateV=nil;
 bool injectLateFailure=false;
 std::uint32_t failureFlags=0u;
};
static bool observeSlider(void* opaque,const metalrobo::MetalWorldDevicePhysicsPass& pass){
 auto& o=*static_cast<SliderObservation*>(opaque);
 if(std::getenv("NM_SLIDER_ENCODE_DIAGNOSTICS")) std::cerr << "slider callback phase=" << static_cast<unsigned>(pass.phase) << " begin\n";
 if(pass.phase==metalrobo::MetalWorldDevicePhysicsPhase::postCommit){
  // Read the published rigid candidate before Matter can reject it and the
  // enclosing owner restores q/v. These copies do not affect acceptance.
  id<MTLCommandBuffer> cmd=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
  id<MTLBuffer> sourceQ=(__bridge id<MTLBuffer>)pass.q,sourceV=(__bridge id<MTLBuffer>)pass.v;
  o.candidateQ=[cmd.commandQueue.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
  o.candidateV=[cmd.commandQueue.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
  if(!sourceQ||!sourceV||!o.candidateQ||!o.candidateV||sourceQ.length<sizeof(float)||sourceV.length<sizeof(float))return false;
  id<MTLBlitCommandEncoder> capture=[cmd blitCommandEncoder];if(!capture)return false;
  [capture copyFromBuffer:sourceQ sourceOffset:0 toBuffer:o.candidateQ destinationOffset:0 size:sizeof(float)];
  [capture copyFromBuffer:sourceV sourceOffset:0 toBuffer:o.candidateV destinationOffset:0 size:sizeof(float)];
  [capture endEncoding];
 }
 if(o.injectLateFailure && pass.phase==metalrobo::MetalWorldDevicePhysicsPhase::postCommit){
  id<MTLCommandBuffer> cmd=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
  id<MTLBuffer> dst=(__bridge id<MTLBuffer>)pass.environmentStatuses;
  MRMetalWorldStatusGPU failure{};failure.code=MR_STEP_DID_NOT_CONVERGE;failure.flags=o.failureFlags;
  id<MTLBuffer> src=[cmd.commandQueue.device newBufferWithBytes:&failure length:sizeof(failure) options:MTLResourceStorageModeShared];
  if(!src||!dst||dst.length<sizeof(failure))return false;
  id<MTLBlitCommandEncoder> inject=[cmd blitCommandEncoder];if(!inject)return false;
  [inject copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:sizeof(failure)];[inject endEncoding];
  [cmd addCompletedHandler:^(id<MTLCommandBuffer> done){(void)done;(void)src;}];
 }
 if(!o.inner.encode(o.inner.context,pass)){std::cerr << "slider inner encode failed phase=" << static_cast<unsigned>(pass.phase) << "\n";return false;}
 if(std::getenv("NM_SLIDER_ENCODE_DIAGNOSTICS")) std::cerr << "slider inner encode succeeded phase=" << static_cast<unsigned>(pass.phase) << "\n";
 if(pass.phase!=metalrobo::MetalWorldDevicePhysicsPhase::postCommit)return true;
 id<MTLCommandBuffer> command=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
 id<MTLBuffer> wrench=(__bridge id<MTLBuffer>)pass.bodyWrenches;
 id<MTLBuffer> support=(__bridge id<MTLBuffer>)pass.sliderMidpointSupportWrench;
 id<MTLDevice> device=command.commandQueue.device;
 o.wrench=[device newBufferWithLength:sizeof(MRABABodyWrenchGPU) options:MTLResourceStorageModeShared];
 o.support=[device newBufferWithLength:sizeof(MRABABodyWrenchGPU) options:MTLResourceStorageModeShared];
 if(!wrench||!support||!o.wrench||!o.support||wrench.length<2*sizeof(MRABABodyWrenchGPU))return false;
 id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];if(!blit)return false;
 [blit copyFromBuffer:wrench sourceOffset:sizeof(MRABABodyWrenchGPU) toBuffer:o.wrench destinationOffset:0 size:sizeof(MRABABodyWrenchGPU)];
 [blit copyFromBuffer:support sourceOffset:0 toBuffer:o.support destinationOffset:0 size:sizeof(MRABABodyWrenchGPU)];
 [blit endEncoding];return true;
}
static void abortSlider(void* opaque,void* command){auto& o=*static_cast<SliderObservation*>(opaque);o.inner.abort(o.inner.context,command);}
static metalrobo::EngineModel makeSlider(float h){
 metalrobo::EngineModel m;m.name="midpoint_fem_reacting_slider";
 m.world.abiVersion=MR_ENGINE_ABI_VERSION;m.world.bodyCount=2;m.world.articulationCount=1;
 m.world.jointCount=1;m.world.nq=1;m.world.nv=1;m.world.pairCapacity=1;m.world.contactCapacity=1;
 m.world.constraintCapacity=1;m.world.islandCapacity=1;m.world.solverType=MR_SOLVER_REFERENCE_FP64;
 m.world.frictionConeType=MR_FRICTION_CONE_ELLIPTIC;m.world.gravityAndTimestep={0,0,-9.81f,h};
 m.world.solverScales={1e-8f,1e-9f,2,1e-4f};
 MRArticulationGPU a{};a.rootBody=0;a.rootType=MR_ROOT_FIXED;a.firstBody=0;a.bodyCount=2;
 a.firstJoint=0;a.jointCount=1;a.nq=1;a.nv=1;m.articulations.push_back(a);
 for(unsigned i=0;i<2;i++){MRBodyPropertiesGPU b{};b.articulationIndex=0;b.parentBody=i?0:MR_INVALID_INDEX;
  b.inboundJoint=i?0:MR_INVALID_INDEX;b.motionType=MR_MOTION_DYNAMIC;
  const float mass=i?.01f:1.f;b.massAndInverseMass={mass,1.f/mass,0,0};
  b.inertiaRow0={.001f,0,0,0};b.inertiaRow1={0,.001f,0,0};b.inertiaRow2={0,0,.001f,0};
  b.inverseInertiaRow0={1000,0,0,0};b.inverseInertiaRow1={0,1000,0,0};b.inverseInertiaRow2={0,0,1000,0};
  m.bodies.push_back(b);}
 MRJointDescriptorGPU j{};j.parentBody=0;j.childBody=1;j.jointType=MR_JOINT_PRISMATIC;
 j.nq=1;j.nv=1;j.axis0={0,0,1,0};j.parentRotation={0,0,0,1};j.childRotation={0,0,0,1};m.joints.push_back(j);
 MRDofPropertiesGPU d{};d.articulationIndex=0;d.jointIndex=0;m.dofs.push_back(d);
 m.defaultQ={0};m.defaultV={.02f};return m;
}
int main(int argc,char**argv){@autoreleasepool{try{
 require(argc==6 || argc==7,"usage: midpoint_slider_coupled_diagnostic MATTER_METALLIB WORLD_METALLIB h initial_gap steps [rollback|rollback_precontact|reset_guard]");
 const bool precontact=argc==7 && std::string(argv[6])=="rollback_precontact";
 const bool rollback=precontact || (argc==7 && std::string(argv[6])=="rollback");
 const bool resetGuard=argc==7 && std::string(argv[6])=="reset_guard";
 require(argc==6 || rollback || resetGuard,"unknown diagnostic mode");
 const double h=std::stod(argv[3]),gap=std::stod(argv[4]);const int steps=std::stoi(argv[5]);
 require(h>0&&gap>0&&(steps>0 || (precontact && steps==0)),"invalid fixture arguments");
 for(const char* key:{"NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION","NM_MATTER_MIDPOINT_STAGE_COMPENSATION"}) {
  const char* value=std::getenv(key);
  require(value && std::strcmp(value,"1")==0,std::string("required diagnostic mode missing: ")+key);
 }
 const auto femWorld=model(2,h,gap,true,false,true,0.,1.,true);
 numi::matter::RuntimeConfiguration mc;mc.metallib=argv[1];mc.environmentCount=1;mc.captureDiagnostics=true;
 mc.adaptiveTransfer=false;mc.experimentalFEMPlaneAreaIPC=true;mc.experimentalFEMPlaneMidpoint=true;
 mc.experimentalFEMPlaneSliderMidpoint=true;numi::matter::Runtime matter;
 const auto init=matter.initialize(femWorld,mc);require(init.encoded&&matter.valid(),"Matter initialize: "+init.message);
 const auto rigid=makeSlider(float(h));
 const double physicalH=double(float(h));
 const double sliderMass=double(rigid.bodies[1].massAndInverseMass.x);
 const double sliderGravity=double(rigid.world.gravityAndTimestep.z);
 metalrobo::CompiledWorld world;
 const auto cooked=metalrobo::compileMetalWorld(rigid,0,world);require(cooked.succeeded(),"rigid compile: "+cooked.message);
 metalrobo::MetalWorldConfig ownerConfig{};ownerConfig.metallibPath=argv[2];metalrobo::MetalWorldContext owner(ownerConfig);
 SliderObservation obs;obs.inner=numi::matter::makeMetalWorldDevicePhysicsProgram(matter);require(obs.inner.valid(),"invalid Matter owner program");
 auto program=obs.inner;program.context=&obs;program.encode=&observeSlider;program.abort=&abortSlider;
 metalrobo::MetalWorldStepConfig config{};config.timestepSeconds=float(h);config.physicsSubsteps=1;
 config.solverMode=metalrobo::MetalWorldSolverMode::freeMotionABA;config.ccdMode=metalrobo::MetalWorldCCDMode::disabled;
 config.matrixFreeArticulatedContact=false;config.streamedArticulatedContactResponses=false;config.captureContactEvidence=false;
 config.devicePhysicsProgram=program;std::vector<float> q=rigid.defaultQ,v=rigid.defaultV,effort(1,0.f);
 auto previous=matter.snapshot();require(previous.available,"initial snapshot unavailable");
 if(const char* path=std::getenv("NM_SLIDER_RESTORE_SNAPSHOT")){
  auto saved=previous;slider_checkpoint::load(path,saved,q,v);
  require(saved.sourcePhysicsFingerprint==previous.sourcePhysicsFingerprint &&
      saved.deviceProgramFingerprint==previous.deviceProgramFingerprint,"slider snapshot fingerprint mismatch");
  const auto restored=matter.restore(saved);require(restored.encoded,"slider snapshot restore: "+restored.message);
  previous=matter.snapshot();require(previous.available && physicalEqual(saved,previous) &&
      saved.controlStep==previous.controlStep && saved.physicsSubstep==previous.physicsSubstep,
      "slider snapshot restore mismatch");
 }
 if(const char* path=std::getenv("NM_SLIDER_SAVE_INITIAL_SNAPSHOT")){
  slider_checkpoint::save(path,previous,q,v);
  auto check=previous;std::vector<float> checkQ,checkV;slider_checkpoint::load(path,check,checkQ,checkV);
  require(physicalEqual(previous,check) && previous.controlStep==check.controlStep &&
      previous.physicsSubstep==check.physicsSubstep && bytesEqual(q,checkQ) && bytesEqual(v,checkV),
      "initial slider checkpoint round-trip mismatch");
 }
 std::vector<float> area;std::string areaError;
 require(numi::matter::detail::femFixedPlaneReferenceAreas(femWorld,area,areaError),"reference areas: "+areaError);
 double totalArea=0;for(float value:area)totalArea+=double(value);
 if(resetGuard){
  const std::vector<std::uint32_t> resetMask{1u};const std::vector<float> resetQ{0.f},resetV{.02f};
  const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,
    .initialQ=q,.initialV=v,.efforts=effort,.resetMasks=resetMask,.resetQ=resetQ,.resetV=resetV};
  metalrobo::MetalWorldResult result;const auto ran=owner.run(world,batch,config,result);
  const auto after=matter.snapshot();
  require(ran.status==metalrobo::MetalWorldHostStatus::unsupportedTopology && after.available && physicalEqual(previous,after) &&
      previous.controlStep==after.controlStep && previous.physicsSubstep==after.physicsSubstep,
      "live reset stream did not reject with Matter state preserved");
  std::cout<<"{\"live_reset_rejected\":true,\"matter_restored\":true,\"host_status\":"<<int(ran.status)<<"}\n";return 0;
 }
 if(rollback){
  // Warm up through real accepted contact. The selected checkpoint, candidate,
  // forced late rejection and subsequent retry all use the same two owners.
  auto advance=[&](metalrobo::MetalWorldResult& result){
   const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,
       .initialQ=q,.initialV=v,.efforts=effort};
   return owner.run(world,batch,config,result);
  };
  bool contactSeen=false;
  for(int i=0;i<steps;++i){metalrobo::MetalWorldResult r;const auto d=advance(r);
   require(d.succeeded() && r.finalQ.size()==1 && r.finalV.size()==1,"warmup failed: "+d.message);
   q=r.finalQ;v=r.finalV;previous=matter.snapshot();require(previous.available,"warmup snapshot unavailable");
   for(const auto& sample:previous.contactSamples)
    contactSeen|=(sample.identity.w&NM_CONTACT_VALID) && sample.impulseAndNormal.w>0.f;
  }
  require(precontact ? !contactSeen : contactSeen,
      precontact ? "precontact rollback checkpoint has contact" : "rollback checkpoint must follow nonzero contact");
  const auto checkpoint=previous;const auto checkpointQ=q,checkpointV=v;
  metalrobo::MetalWorldResult good;const auto goodStatus=advance(good);
  require(goodStatus.succeeded(),"baseline continuation failed: "+goodStatus.message);
  const auto goodMatter=matter.snapshot();require(goodMatter.available,"baseline snapshot unavailable");
  obs.failureFlags=good.layout.dispatch.flags;
  const auto restored=matter.restore(checkpoint);require(restored.encoded,"restore checkpoint failed: "+restored.message);
  obs.injectLateFailure=true;metalrobo::MetalWorldResult rejected;const auto failure=advance(rejected);obs.injectLateFailure=false;
  const auto failedMatter=matter.snapshot();require(failedMatter.available,"rejected snapshot unavailable");
  const bool rigidRestored=bytesEqual(rejected.finalQ,checkpointQ)&&bytesEqual(rejected.finalV,checkpointV);
  const bool matterRestored=physicalEqual(checkpoint,failedMatter)&&checkpoint.controlStep==failedMatter.controlStep &&
      checkpoint.physicsSubstep==failedMatter.physicsSubstep;
  require(!failure.succeeded()&&failure.firstGPUStatusCode==MR_STEP_DID_NOT_CONVERGE && rigidRestored && matterRestored,
      "late failure category or whole-environment restoration failed: "+failure.message);
  metalrobo::MetalWorldResult retry;const auto retryStatus=advance(retry);const auto retryMatter=matter.snapshot();
  const bool exact=retryStatus.succeeded() && retryMatter.available &&
      bytesEqual(good.finalQ,retry.finalQ)&&bytesEqual(good.finalV,retry.finalV)&&
      physicalEqual(goodMatter,retryMatter)&&goodMatter.controlStep==retryMatter.controlStep &&
      goodMatter.physicsSubstep==retryMatter.physicsSubstep;
  require(exact,"accepted retry differs from uninterrupted continuation");
  std::cout<<"{\"after_nonzero_contact\":"<<(contactSeen?"true":"false")<<",\"failure_category\":"<<failure.firstGPUStatusCode
      <<",\"rigid_restored\":true,\"matter_restored\":true,\"retry_exact\":true}\n";
  return 0;
 }
 bool anyRejected=false;
 std::cout<<std::setprecision(17)<<"{\"schema\":\"numi.slider-coupled-diagnostic.v1\",\"h\":"<<physicalH
   <<",\"initial_gap\":"<<gap<<",\"matter_fingerprint\":"<<matter.deviceProgramFingerprint()
   <<",\"source_fingerprint\":"<<femWorld.physicsFingerprint
   <<",\"slider_mass_kg\":"<<sliderMass<<",\"initial_slider_velocity_mps\":"<<double(rigid.defaultV[0])
   <<",\"reference_area_m2\":"<<totalArea
   <<",\"activation_width_m\":"<<femWorld.dispatch.numericalLimits.x
   <<",\"contact_coefficient\":"<<mc.experimentalFEMPlaneAreaIPCK
   <<",\"reference_area_weights_m2\":[";
 for(std::size_t i=0;i<area.size();++i)std::cout<<(i?",":"")<<area[i];
 std::cout<<"],\"physical_residual_limit_Ns\":"<<2.0*(double(femWorld.mixedSolver.contactAcceptance.y)+
       double(femWorld.mixedSolver.contactAcceptance.z)*femWorld.mixedSolver.contactAcceptance.w)
   <<",\"steps\":[";
 for(int step=0;step<steps;step++){
  const auto startQ=q,startV=v;const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,
    .initialQ=q,.initialV=v,.efforts=effort};metalrobo::MetalWorldResult result;
  auto ran=owner.run(world,batch,config,result);auto now=matter.snapshot();
  require(now.available && now.statuses.size()==1,"snapshot or status unavailable");
  const bool accepted=ran.succeeded() && now.statuses[0].code==NM_STATUS_SUCCESS;
  if(step)std::cout<<",";std::cout<<"{\"step\":"<<step<<",\"accepted\":"<<(accepted?"true":"false")
   <<",\"world_status\":"<<ran.firstGPUStatusCode<<",\"matter_status\":"<<now.statuses[0].code;
  if(!accepted){
   anyRejected=true;
   if(const char* path=std::getenv("NM_SLIDER_FAILURE_SNAPSHOT")){
    slider_checkpoint::save(path,previous,startQ,startV);
    auto roundtrip=previous;std::vector<float> checkQ,checkV;
    slider_checkpoint::load(path,roundtrip,checkQ,checkV);
    require(physicalEqual(previous,roundtrip) && previous.controlStep==roundtrip.controlStep &&
        previous.physicsSubstep==roundtrip.physicsSubstep && bytesEqual(startQ,checkQ) && bytesEqual(startV,checkV),
        "slider failure snapshot round-trip mismatch");
    std::cerr<<"saved verified accepted-state slider checkpoint: "<<path<<"\n";
   }
   std::cerr<<"step "<<step<<" rejected: host="<<int(ran.status)<<" message="<<ran.message
       <<" gpu="<<ran.firstGPUStatusCode<<" matter="<<now.statuses[0].code
       <<" index="<<now.statuses[0].failingIndex<<"\n";
   const auto& failed=now.statuses[0];
   if(obs.candidateQ && obs.candidateV){
    std::cout<<",\"candidate_q\":"<<*static_cast<const float*>(obs.candidateQ.contents)
             <<",\"candidate_v\":"<<*static_cast<const float*>(obs.candidateV.contents);
   }
   if(obs.wrench){const auto& w=*static_cast<const MRABABodyWrenchGPU*>(obs.wrench.contents);
    std::cout<<",\"candidate_body_force_z\":"<<w.force.z;}
   std::cout<<",\"host_status\":"<<int(ran.status)<<",\"failure_index\":"<<failed.failingIndex
       <<",\"failure_object\":"<<failed.objectIndex<<",\"fgmres_iterations\":"<<failed.fgmresIterations
       <<",\"diagnostics\":["<<failed.diagnostics.x<<","<<failed.diagnostics.y<<","<<failed.diagnostics.z<<","<<failed.diagnostics.w<<"]}";break;
  }
  require(result.finalQ.size()==1&&result.finalV.size()==1&&obs.wrench&&obs.support &&
      now.solverCertificates.size()==1,"missing owner data or certificate");
  q=result.finalQ;v=result.finalV;const auto body=*static_cast<const MRABABodyWrenchGPU*>(obs.wrench.contents);
  const auto support=*static_cast<const MRABABodyWrenchGPU*>(obs.support.contents);
  double impulse=0,torqueX=0,torqueY=0,work=0,low=1e100,midLow=1e100;
  double oldContactPotential=0,newContactPotential=0,femLawWork=0,femWorkGateRatio=0;
  std::vector<double> nodalContactImpulse(now.femNodes.size(),0.0);
  std::vector<std::array<double,4>> femWorkNodes;
  for(const auto& s:now.contactSamples)if(s.identity.w&NM_CONTACT_VALID){
   require(s.identity.x>=femWorld.dispatch.gridNodeCount,"contact is not a FEM node");
   const auto node=s.identity.x-femWorld.dispatch.gridNodeCount;require(node<now.femNodes.size(),"contact node out of range");
   const double jz=-double(s.impulseAndNormal.z);impulse+=jz;nodalContactImpulse[node]+=jz;
   const double x=s.pointAndSeparation.x,y=s.pointAndSeparation.y;torqueX+=y*jz;torqueY-=x*jz;
   const double vm=.5*(physicalVelocity(previous,node,2)+physicalVelocity(now,node,2));work+=jz*vm;
  }
  for(std::size_t i=0;i<now.femNodes.size();++i){
   const auto& n=now.femNodes[i];const auto& old=previous.femNodes[i];
   const double z=double(n.restAndFixed.z)+n.referenceDisplacementAndMode.z+n.referenceDisplacementLow.z-double(q[0]);
   const double oldZ=double(old.restAndFixed.z)+old.referenceDisplacementAndMode.z+old.referenceDisplacementLow.z-double(startQ[0]);
   low=std::min(low,z);midLow=std::min(midLow,.5*(oldZ+z));
   const double d=femWorld.dispatch.numericalLimits.x,k=mc.experimentalFEMPlaneAreaIPCK;
   auto potential=[&](double g){if(area[i]<=0 || g>=d)return 0.;require(g>0,"nonpositive clearance in potential");const double v=g*g-d*d;return -double(area[i])*k*v*v*std::log(g*g/(d*d));};
   oldContactPotential+=potential(oldZ);newContactPotential+=potential(z);
   // Independent FP64 law reconstruction uses the actual nodal midpoint and
   // the fixed physical reference area. Compare work per node, so opposing
   // local errors cannot disappear in the total contact work.
   if(area[i]>0) {
    const double midpointGap=.5*(oldZ+z);
    require(midpointGap>0,"nonpositive midpoint clearance in contact-work reconstruction");
    double pressure=0;
    if(midpointGap<d) {
     const double square=midpointGap*midpointGap,shift=square-d*d;
     pressure=2*midpointGap*k*(2*shift*std::log(square/(d*d))+shift*shift/square);
    }
    const double expectedImpulse=physicalH*double(area[i])*pressure;
    const double vm=.5*(physicalVelocity(previous,i,2)+physicalVelocity(now,i,2));
    const double bound=std::max(1e-12,1e-5*std::abs(expectedImpulse))*std::abs(vm)+1e-24;
    femWorkNodes.push_back({double(i),nodalContactImpulse[i],expectedImpulse,vm});
    femLawWork+=expectedImpulse*vm;
    femWorkGateRatio=std::max(femWorkGateRatio,std::abs((nodalContactImpulse[i]-expectedImpulse)*vm)/bound);
   }
  }
  const double ownerImpulse=sliderMass*(double(v[0])-double(startV[0]))-physicalH*sliderMass*sliderGravity;
  const double ownerWork=ownerImpulse*.5*(double(startV[0])+double(v[0]));
  const auto independent=fixtureResidual(femWorld,previous,now,physicalH,true);
  const double ownerResidual=ownerImpulse-double(body.force.z)*physicalH;
  const double jointResidual=std::hypot(independent.l2,ownerResidual);
  const double sliderKinematic=double(q[0])-double(startQ[0])-
      physicalH*.5*(double(startV[0])+double(v[0]));
  std::cout<<",\"gap\":"<<low<<",\"midpoint_gap\":"<<midLow<<",\"q\":"<<q[0]<<",\"v\":"<<v[0]<<",\"fem_impulse\":"<<impulse
   <<",\"owner_momentum_impulse\":"<<ownerImpulse<<",\"owner_wrench_impulse\":"<<double(body.force.z)*physicalH
   <<",\"fem_torque_impulse_x\":"<<torqueX<<",\"fem_torque_impulse_y\":"<<torqueY
   <<",\"owner_moment_impulse_x\":"<<double(body.torque.x)*physicalH<<",\"owner_moment_impulse_y\":"<<double(body.torque.y)*physicalH
   <<",\"support_moment_impulse_x\":"<<double(support.torque.x)*physicalH<<",\"support_moment_impulse_y\":"<<double(support.torque.y)*physicalH
   <<",\"fem_contact_work\":"<<work<<",\"fem_law_work\":"<<femLawWork
   <<",\"fem_work_reconstruction_gate_ratio\":"<<femWorkGateRatio<<",\"owner_contact_work\":"<<ownerWork
   <<",\"fem_work_nodes\":[";
  for(std::size_t i=0;i<femWorkNodes.size();++i){
   if(i)std::cout<<",";const auto& row=femWorkNodes[i];
   std::cout<<"["<<row[0]<<","<<row[1]<<","<<row[2]<<","<<row[3]<<"]";
  }
  std::cout<<"],\"owner_wrench_work\":"<<double(body.force.z)*physicalH*.5*(double(startV[0])+double(v[0]))
   <<",\"contact_potential_before_J\":"<<oldContactPotential<<",\"contact_potential_after_J\":"<<newContactPotential
   <<",\"independent_fem_residual_Ns\":"<<independent.l2
   <<",\"independent_joint_residual_Ns\":"<<jointResidual
   <<",\"fem_kinematic_defect_m\":"<<independent.kinematic
   <<",\"slider_kinematic_defect_m\":"<<sliderKinematic
   <<",\"solver_contact_potential_J\":"<<now.solverCertificates[0].contact.w<<"}";
  previous=std::move(now);
 }
 std::cout<<"]}\n";return anyRejected?2:0;
 }catch(const std::exception&e){std::cerr<<"coupled diagnostic: "<<e.what()<<"\n";return 1;}}}
