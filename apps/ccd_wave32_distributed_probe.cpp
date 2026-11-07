#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc,char** argv) {
 if(argc!=2||!std::filesystem::is_regular_file(argv[1]))return 2;
 std::cout<<std::setprecision(10)<<"{\"cases\":[";
 bool first=true;
 for(std::uint32_t count:{33u,257u})for(float initialVelocity:{-1.f,0.f,1.f})for(std::uint32_t iterations:{1u,2u,3u,16u}) {
  auto model=metalrobo::makeFreeSphereEngineModel();
  model.world.gravityAndTimestep={0,0,0,.01f};
  for(auto& m:model.materials){m.friction.x=0;m.friction.y=0;m.response.x=0;m.response.y=0;}
  auto& b=model.bodies[1];
  b.inertiaRow0={1.e3f,0,0,0};b.inertiaRow1={0,1.e3f,0,0};b.inertiaRow2={0,0,1.e3f,0};
  b.inverseInertiaRow0={1.e-3f,0,0,0};b.inverseInertiaRow1={0,1.e-3f,0,0};b.inverseInertiaRow2={0,0,1.e-3f,0};
  const MRShapeGPU original=model.shapes.back();
  model.shapes.clear();
  for(std::uint32_t body=0;body<2;++body)for(std::uint32_t i=0;i<count;++i) {
   MRShapeGPU s=original;s.bodyIndex=body;s.slotGeneration=body*count+i+1;
   s.localPosition.x=.05f*(float(i)-.5f*float(count-1));
   s.dimensions.x=.01f;s.contactRestAndBoundingRadius={.001f,0,.01f,0};
   model.shapes.push_back(s);
  }
  model.world.shapeCount=std::uint32_t(model.shapes.size());
  model.defaultQ[1]=.0199f;model.defaultV[1]=initialVelocity;
  const auto tiles=(count+MR_WAVE32_CONTACTS_PER_TILE-1u)/MR_WAVE32_CONTACTS_PER_TILE;
  const auto spill=count>MR_WAVE32_CONTACTS_PER_TILE?3u*(count-MR_WAVE32_CONTACTS_PER_TILE):0u;
  metalrobo::CompiledWorld world;
  auto compiled=metalrobo::compileMetalWorld(model,0,world,metalrobo::MetalWorldCapacityProfile{
    .candidatePairs=count,.rawContacts=count,.manifolds=count,.constraintBlocks=count,.constraintRows=3u*count,
    .islands=2u,.solverTiles=tiles,.spillRows=spill});
  if(!compiled.succeeded()){std::cerr<<"compile failed count="<<count<<"\n";return 3;}
  std::vector<float> effort(world.nv(),0.f);
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;
  ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  const std::vector<MRBodyStateGPU> scene{ground};
  const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,
    .initialV=model.defaultV,.efforts=effort,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig config;
  config.timestepSeconds=.01f;config.physicsSubsteps=1;config.velocityIterations=iterations;config.finalVelocityIterations=0;
  config.ccdMode=metalrobo::MetalWorldCCDMode::disabled;config.applyBodyDamping=false;config.captureContactEvidence=true;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];
  metalrobo::MetalWorldContext context(owner);
  metalrobo::MetalWorldResult result;auto d=context.run(world,batch,config,result);
  metalrobo::MetalWorldResult repeat;auto dr=context.run(world,batch,config,repeat);
  const auto equal=[](const auto& a,const auto& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0);};
  const bool replay=equal(result.finalQ,repeat.finalQ)&&equal(result.finalV,repeat.finalV)&&equal(result.finalSceneBodies,repeat.finalSceneBodies)&&d.firstGPUStatusCode==dr.firstGPUStatusCode;
  const bool restored=d.succeeded()||(equal(result.finalQ,model.defaultQ)&&equal(result.finalV,model.defaultV)&&equal(result.finalSceneBodies,scene));
  const bool empty=d.succeeded()||std::all_of(result.contactEvidence.manifoldCounts.begin(),result.contactEvidence.manifoldCounts.end(),[](auto n){return n==0;});
  if(result.contactStatuses.empty()||result.finalV.size()<2){std::cerr<<"missing output\n";return 4;}
  const auto& status=result.contactStatuses[0];
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"contacts\":"<<count<<",\"iterations\":"<<iterations
   <<",\"accepted\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<d.firstGPUStatusCode
   <<",\"active_contacts\":"<<status.activeContacts<<",\"queue_flags\":"<<status.queueFlags
   <<",\"distributed\":"<<((status.queueFlags&MR_ISLAND_WORK_DISTRIBUTED)?"true":"false")
   <<",\"initial_vn\":"<<initialVelocity<<",\"final_vn\":"<<result.finalV[1]
   <<",\"normal_residual\":"<<status.residuals.y
   <<",\"replay_identical\":"<<(replay?"true":"false")
   <<",\"state_restored\":"<<(restored?"true":"false")
   <<",\"rejected_contact_cache_empty\":"<<(empty?"true":"false")<<"}";
 }
 std::cout<<"]}\n";
}
