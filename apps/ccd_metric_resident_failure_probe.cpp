#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

template<class T> bool equal(const std::vector<T>& a,const std::vector<T>& b) {
 return a.size()==b.size() && (a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
int main(int argc,char** argv) {
 if(argc!=4||!std::filesystem::is_regular_file(argv[1])||!std::filesystem::is_regular_file(argv[2])) return 2;
 const std::uint32_t count=std::uint32_t(std::stoul(argv[3]));
 if(count!=33 && count!=257) return 2;
 auto model=metalrobo::makeFreeSphereEngineModel();
 model.world.gravityAndTimestep={0,0,0,.01f};
 for(auto& m:model.materials){m.friction.x=12345.f;m.friction.y=12345.f;m.response.x=0;m.response.y=0;}
 model.bodies[0].motionType=MR_MOTION_KINEMATIC;
 auto& b=model.bodies[1];
 b.inertiaRow0={1.e3f,0,0,0};b.inertiaRow1={0,1.e3f,0,0};b.inertiaRow2={0,0,1.e3f,0};
 b.inverseInertiaRow0={1.e-3f,0,0,0};b.inverseInertiaRow1={0,1.e-3f,0,0};b.inverseInertiaRow2={0,0,1.e-3f,0};
 const MRShapeGPU original=model.shapes.back(); model.shapes.clear();
 for(std::uint32_t body=0;body<2;++body)for(std::uint32_t i=0;i<count;++i) {
  auto s=original;s.bodyIndex=body;s.slotGeneration=body*count+i+1;
  s.localPosition.x=.05f*(float(i)-.5f*float(count-1));s.dimensions.x=.01f;
  s.contactRestAndBoundingRadius={.001f,0,.01f,0};model.shapes.push_back(s);
 }
 model.world.shapeCount=std::uint32_t(model.shapes.size());model.defaultQ[1]=.0199f;
 const auto tiles=(count+MR_WAVE32_CONTACTS_PER_TILE-1u)/MR_WAVE32_CONTACTS_PER_TILE;
 const auto spill=3u*(count-MR_WAVE32_CONTACTS_PER_TILE);
 metalrobo::CompiledWorld world;
 auto c=metalrobo::compileMetalWorld(model,0,world,metalrobo::MetalWorldCapacityProfile{
  .candidatePairs=count,.rawContacts=count,.manifolds=count,.constraintBlocks=count,
  .constraintRows=3u*count,.islands=2u,.solverTiles=tiles,.spillRows=spill});
 if(!c.succeeded()){std::cerr<<"compile failed\n";return 3;}
 MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;
 ground.flagsAndIndices[0]=MR_MOTION_KINEMATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
 const std::vector<MRBodyStateGPU> scene{ground};
 const std::vector<float> zero(world.nv(),0.f);
 MRBodyStateGPU driven=ground;driven.linearVelocityAndInverseMass.x=-1.f;driven.linearVelocityAndInverseMass.y=1.f;
 const std::vector<MRBodyStateGPU> targets{driven};
 const metalrobo::MetalWorldBatch initial{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,
  .initialV=model.defaultV,.efforts=zero,.initialSceneBodies=scene};
 const metalrobo::MetalWorldBatch rejected{.environmentCount=1,.controlStepCount=1,.efforts=zero,.kinematicTargets=targets};
 const metalrobo::MetalWorldBatch neutral{.environmentCount=1,.controlStepCount=1,.efforts=zero};
 metalrobo::MetalWorldStepConfig config;config.timestepSeconds=.01f;config.physicsSubsteps=1;
 config.velocityIterations=2;config.finalVelocityIterations=0;config.ccdMode=metalrobo::MetalWorldCCDMode::disabled;
 config.applyBodyDamping=false;config.captureContactEvidence=true;
 auto failing=config;
 metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];
 metalrobo::MetalWorldContext ctx(owner),control(owner);
 metalrobo::MetalWorldResidentState resident,reference;
 metalrobo::MetalWorldSubmission ticket,referenceTicket;
 metalrobo::MetalWorldResult before,failed,after,referenceBefore,referenceAfter;
 auto a=ctx.initializeResidentState(world,initial,config,resident,ticket);
 auto b1=control.initializeResidentState(world,initial,config,reference,referenceTicket);
 if(!a.succeeded()||!b1.succeeded()){std::cerr<<"resident initialize submit failed\n";return 4;}
 auto ad=ticket.wait(before),bd=referenceTicket.wait(referenceBefore);
 auto countCache=[](const auto& r){std::uint64_t n=0;for(auto c:r.contactEvidence.manifoldCounts)n+=c;return n;};
 auto q=ctx.submitResident(world,rejected,failing,resident,ticket);
 if(!q.succeeded()){std::cerr<<"failure submit rejected on host status="<<int(q.status)<<"\n";return 5;}
 auto qd=ticket.wait(failed);
 bool restored=equal(before.finalQ,failed.finalQ)&&equal(before.finalV,failed.finalV)&&
  equal(before.finalSceneBodies,failed.finalSceneBodies)&&
  equal(before.contactEvidence.manifoldCounts,failed.contactEvidence.manifoldCounts)&&
  equal(before.contactEvidence.manifoldHeaders,failed.contactEvidence.manifoldHeaders)&&
  equal(before.contactEvidence.manifoldPoints,failed.contactEvidence.manifoldPoints);
 auto s=ctx.submitResident(world,neutral,config,resident,ticket);
 auto t=control.submitResident(world,neutral,config,reference,referenceTicket);
 if(!s.succeeded()||!t.succeeded()){std::cerr<<"continuation submit failed\n";return 6;}
 auto sd=ticket.wait(after),td=referenceTicket.wait(referenceAfter);
 bool same=equal(after.finalQ,referenceAfter.finalQ)&&equal(after.finalV,referenceAfter.finalV)&&
  equal(after.finalSceneBodies,referenceAfter.finalSceneBodies)&&
  equal(after.contactEvidence.manifoldCounts,referenceAfter.contactEvidence.manifoldCounts)&&
  equal(after.contactEvidence.manifoldHeaders,referenceAfter.contactEvidence.manifoldHeaders)&&
  equal(after.contactEvidence.manifoldPoints,referenceAfter.contactEvidence.manifoldPoints);
 metalrobo::MetalWorldConfig acceptedOwner=owner;acceptedOwner.metallibPath=argv[2];
 metalrobo::MetalWorldContext acceptedCtx(acceptedOwner);
 metalrobo::MetalWorldResidentState acceptedResident;
 metalrobo::MetalWorldSubmission acceptedTicket;
 metalrobo::MetalWorldResult acceptedInitial,acceptedTarget;
 auto as=acceptedCtx.initializeResidentState(world,initial,config,acceptedResident,acceptedTicket);
 if(!as.succeeded()){std::cerr<<"accepted-control initialize failed\n";return 7;}
 auto ai=acceptedTicket.wait(acceptedInitial);
 auto ats=acceptedCtx.submitResident(world,rejected,config,acceptedResident,acceptedTicket);
 if(!ats.succeeded()){std::cerr<<"accepted-control submit failed\n";return 8;}
 auto at=acceptedTicket.wait(acceptedTarget);
 const bool acceptedTargetApplied=ai.succeeded()&&at.succeeded()&&!acceptedTarget.finalSceneBodies.empty()&&
  acceptedTarget.finalSceneBodies[0].linearVelocityAndInverseMass.x==-1.f &&
  acceptedTarget.finalSceneBodies[0].linearVelocityAndInverseMass.y==1.f;
 std::cout<<"{\"accepted_target_status\":"<<at.firstGPUStatusCode
  <<",\"accepted_target_applied\":"<<(acceptedTargetApplied?"true":"false")
  <<",\"initial_ok\":"<<(ad.succeeded()&&bd.succeeded()?"true":"false")
  <<",\"initial_cache_count\":"<<countCache(before)
  <<",\"initial_reference_identical\":"<<(equal(before.finalQ,referenceBefore.finalQ)&&equal(before.finalV,referenceBefore.finalV)?"true":"false")
  <<",\"failure_status\":"<<qd.firstGPUStatusCode<<",\"failure_succeeded\":"<<(qd.succeeded()?"true":"false")
  <<",\"failed_vx\":"<<(failed.finalV.empty()?0:failed.finalV[0])
  <<",\"failed_vy\":"<<(failed.finalV.size()<2?0:failed.finalV[1])
  <<",\"normal_residual\":"<<(failed.contactStatuses.empty()?0:failed.contactStatuses[0].residuals.y)
  <<",\"active_contacts\":"<<(failed.contactStatuses.empty()?0:failed.contactStatuses[0].activeContacts)
  <<",\"failed_cache_count\":"<<countCache(failed)
  <<",\"failed_state_restored\":"<<(restored?"true":"false")
  <<",\"restore_q\":"<<(equal(before.finalQ,failed.finalQ)?"true":"false")
  <<",\"restore_v\":"<<(equal(before.finalV,failed.finalV)?"true":"false")
  <<",\"restore_scene\":"<<(equal(before.finalSceneBodies,failed.finalSceneBodies)?"true":"false")
  <<",\"before_scene_vx\":"<<(before.finalSceneBodies.empty()?0:before.finalSceneBodies[0].linearVelocityAndInverseMass.x)
  <<",\"before_scene_vy\":"<<(before.finalSceneBodies.empty()?0:before.finalSceneBodies[0].linearVelocityAndInverseMass.y)
  <<",\"failed_scene_vx\":"<<(failed.finalSceneBodies.empty()?0:failed.finalSceneBodies[0].linearVelocityAndInverseMass.x)
  <<",\"failed_scene_vy\":"<<(failed.finalSceneBodies.empty()?0:failed.finalSceneBodies[0].linearVelocityAndInverseMass.y)
  <<",\"before_scene_x\":"<<(before.finalSceneBodies.empty()?0:before.finalSceneBodies[0].position.x)
  <<",\"before_scene_y\":"<<(before.finalSceneBodies.empty()?0:before.finalSceneBodies[0].position.y)
  <<",\"failed_scene_x\":"<<(failed.finalSceneBodies.empty()?0:failed.finalSceneBodies[0].position.x)
  <<",\"failed_scene_y\":"<<(failed.finalSceneBodies.empty()?0:failed.finalSceneBodies[0].position.y)
  <<",\"failed_scene_equals_target\":"<<(equal(failed.finalSceneBodies,targets)?"true":"false")


  <<",\"restore_counts\":"<<(equal(before.contactEvidence.manifoldCounts,failed.contactEvidence.manifoldCounts)?"true":"false")
  <<",\"restore_headers\":"<<(equal(before.contactEvidence.manifoldHeaders,failed.contactEvidence.manifoldHeaders)?"true":"false")
  <<",\"restore_points\":"<<(equal(before.contactEvidence.manifoldPoints,failed.contactEvidence.manifoldPoints)?"true":"false")
  <<",\"continue_q\":"<<(equal(after.finalQ,referenceAfter.finalQ)?"true":"false")
  <<",\"continue_v\":"<<(equal(after.finalV,referenceAfter.finalV)?"true":"false")
  <<",\"continue_scene\":"<<(equal(after.finalSceneBodies,referenceAfter.finalSceneBodies)?"true":"false")
  <<",\"continue_counts\":"<<(equal(after.contactEvidence.manifoldCounts,referenceAfter.contactEvidence.manifoldCounts)?"true":"false")
  <<",\"continue_headers\":"<<(equal(after.contactEvidence.manifoldHeaders,referenceAfter.contactEvidence.manifoldHeaders)?"true":"false")
  <<",\"continue_points\":"<<(equal(after.contactEvidence.manifoldPoints,referenceAfter.contactEvidence.manifoldPoints)?"true":"false")

  <<",\"continuation_status\":"<<sd.firstGPUStatusCode<<",\"control_status\":"<<td.firstGPUStatusCode
  <<",\"continuation_identical\":"<<(same?"true":"false")<<"}\n";
}
