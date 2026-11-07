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
 for(std::uint32_t count:{33u,257u})for(float friction:{0.f})for(int mixed:{0})for(float initialVelocity:{-.08175f})for(std::uint32_t iterations:{1u,2u,3u,8u,16u,32u,64u}) {
  if(mixed && friction>0)continue;
  auto model=metalrobo::makeFreeSphereEngineModel();
  model.world.gravityAndTimestep={0,0,0,.01f};
  for(auto& m:model.materials){m.friction.x=friction;m.friction.y=friction;m.response.x=0;m.response.y=0;}
  auto& b=model.bodies[1];
  b.inertiaRow0={.1f,0,0,0};b.inertiaRow1={0,.1f,0,0};b.inertiaRow2={0,0,.1f,0};
  b.inverseInertiaRow0={10.f,0,0,0};b.inverseInertiaRow1={0,10.f,0,0};b.inverseInertiaRow2={0,0,10.f,0};
  const MRShapeGPU original=model.shapes.back();
  model.shapes.clear();
  for(std::uint32_t body=0;body<2;++body)for(std::uint32_t i=0;i<count;++i) {
   MRShapeGPU s=original;s.bodyIndex=body;s.slotGeneration=body*count+i+1;
   s.localPosition.x=.05f*float(i);
   if(body==1u && mixed && i%3u!=0u) {
       s.localPosition.y=-.0199f;
       if(i%3u==1u)s.localPosition.x+=.0199f;
       else s.localPosition.z+=.0199f;
   }
   s.dimensions.x=.01f;s.contactRestAndBoundingRadius={.001f,0,.01f,0};
   model.shapes.push_back(s);
  }
  model.world.shapeCount=std::uint32_t(model.shapes.size());
  const float incoming=mixed?initialVelocity*.01f:initialVelocity;
  model.defaultQ[1]=.0199f;model.defaultV[1]=incoming;
  model.defaultV[0]=mixed?incoming:(friction>0?1.f:0.f);
  model.defaultV[2]=mixed?incoming:0.f;
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
  double totalNormalImpulse=0,totalTangentialMagnitude=0,impulseMin=1.e100,impulseMax=0;
  double vectorImpulse[3]={0,0,0};
  for(std::size_t i=0;i<std::min<std::size_t>(status.activeContacts,result.contactEvidence.contacts.size());++i) {
      const auto& c=result.contactEvidence.contacts[i];const auto& j=c.impulses;
      if(count==257u) { const auto& row=result.contactEvidence.evaluatedRows[3*i]; std::cerr << std::setprecision(10) << "row " << iterations << " " << i << " " << c.pointAndSeparation.x << " " << c.pointAndSeparation.y << " " << c.pointAndSeparation.z << " " << c.pointAndSeparation.w << " " << c.normal.x << " " << c.normal.y << " " << c.normal.z << " " << row.targetVelocity << " " << row.preSolveVelocity << " " << row.regularization << " " << j.x << " " << j.y << " " << j.z << '\n'; }

      totalNormalImpulse+=j.x;totalTangentialMagnitude+=std::hypot(j.y,j.z);
      impulseMin=std::min(impulseMin,double(j.x));impulseMax=std::max(impulseMax,double(j.x));
      const double n[3]={c.normal.x,c.normal.y,c.normal.z},t[3]={c.tangent.x,c.tangent.y,c.tangent.z};
      const double b[3]={n[1]*t[2]-n[2]*t[1],n[2]*t[0]-n[0]*t[2],n[0]*t[1]-n[1]*t[0]};
      for(unsigned k=0;k<3;++k)vectorImpulse[k]+=n[k]*j.x+t[k]*j.y+b[k]*j.z;
  }
  double momentumError=0,work=0;
  for(unsigned i=0;i<3;++i) {
      const double dv=result.finalV[i]-model.defaultV[i];
      momentumError=std::max(momentumError,std::abs(dv-vectorImpulse[i]));
      work+=.5*(double(result.finalV[i])*result.finalV[i]-double(model.defaultV[i])*model.defaultV[i]);
      work+=.05*(double(result.finalV[i+3])*result.finalV[i+3]-double(model.defaultV[i+3])*model.defaultV[i+3]);
  }
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"contacts\":"<<count<<",\"iterations\":"<<iterations
   <<",\"accepted\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<d.firstGPUStatusCode
   <<",\"active_contacts\":"<<status.activeContacts<<",\"queue_flags\":"<<status.queueFlags
   <<",\"distributed\":"<<((status.queueFlags&MR_ISLAND_WORK_DISTRIBUTED)?"true":"false")
   <<",\"mixed_normals\":"<<(mixed?"true":"false")<<",\"friction\":"<<friction<<",\"initial_vt\":"<<model.defaultV[0]
   <<",\"final_vt\":"<<result.finalV[0]
   <<",\"initial_vn\":"<<incoming<<",\"final_vn\":"<<result.finalV[1]
   <<",\"final_vz\":"<<result.finalV[2]
   <<",\"normal_residual\":"<<status.residuals.y
   <<",\"candidate_normal_impulse_ns\":"<<totalNormalImpulse
   <<",\"candidate_normal_impulse_min_ns\":"<<(impulseMin==1.e100?0:impulseMin)
   <<",\"candidate_normal_impulse_max_ns\":"<<impulseMax;
  if(d.succeeded())std::cout<<",\"normal_impulse_ns\":"<<totalNormalImpulse
   <<",\"momentum_impulse_error_ns\":"<<momentumError
   <<",\"contact_kinetic_work_j\":"<<work
   <<",\"tangent_impulse_magnitude_ns\":"<<totalTangentialMagnitude;
  else std::cout<<",\"normal_impulse_ns\":null,\"momentum_impulse_error_ns\":null,\"contact_kinetic_work_j\":null,\"tangent_impulse_magnitude_ns\":null";
  std::cout
   <<",\"replay_identical\":"<<(replay?"true":"false")
   <<",\"state_restored\":"<<(restored?"true":"false")
   <<",\"rejected_contact_cache_empty\":"<<(empty?"true":"false")<<"}";
 }
 std::cout<<"]}\n";
}
