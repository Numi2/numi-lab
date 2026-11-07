#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
int main(int argc,char** argv) {
 if(argc<2)return 2;
 std::cout<<std::setprecision(12)<<"{\"schema\":\"numi.ccd.complete-steps.v1\",\"runs\":[";bool first=true;
 for(int limit:{16,128})for(int mode=0;mode<3;++mode)for(int fixture=0;fixture<3;++fixture) {
  auto model=metalrobo::makeFreeSphereEngineModel();
  model.world.gravityAndTimestep={0,0,0,.01f};
  auto& shape=model.shapes[1];shape.shapeType=MR_SHAPE_BOX;shape.flags|=MR_SHAPE_FLAG_ENABLE_CCD;
  shape.dimensions={.2f,.002f,.03f,0};shape.contactRestAndBoundingRadius={.001f,0,std::sqrt(.2f*.2f+.002f*.002f+.03f*.03f),0};
  auto& body=model.bodies[1];const float ix=(.002f*.002f+.03f*.03f)/3.f,iy=(.2f*.2f+.03f*.03f)/3.f,iz=(.2f*.2f+.002f*.002f)/3.f;
  body.inertiaRow0={ix,0,0,0};body.inertiaRow1={0,iy,0,0};body.inertiaRow2={0,0,iz,0};
  body.inverseInertiaRow0={1/ix,0,0,0};body.inverseInertiaRow1={0,1/iy,0,0};body.inverseInertiaRow2={0,0,1/iz,0};
  model.defaultQ[1]=fixture==0?.12f:.1998f;
  model.defaultV[1]=fixture==0?-22.f:0.f;
  model.defaultV[5]=fixture==0?0.f:(fixture==1?250.f:177.f);
  metalrobo::CompiledWorld world;auto compiled=metalrobo::compileMetalWorld(model,0,world);
  if(!compiled.succeeded()){std::cerr<<"world compile failed\n";return 1;}
  std::vector<float> efforts(world.nv(),0.f);
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  const std::vector<MRBodyStateGPU> scene{ground};
  const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=efforts,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig config;config.timestepSeconds=.01f;config.physicsSubsteps=1;
  config.velocityIterations=2;config.finalVelocityIterations=1;config.ccdMode=metalrobo::MetalWorldCCDMode::hybrid;
  config.maxConservativeAdvancementIterations=limit;config.directionalCCDAdvancement=mode>0;config.intervalLocalCCDAdvancement=mode==2;
  config.captureContactEvidence=true;config.applyBodyDamping=false;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext context(owner);
  metalrobo::MetalWorldResult result;auto start=std::chrono::steady_clock::now();auto d=context.run(world,batch,config,result);
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  metalrobo::MetalWorldResult repeated;auto d2=context.run(world,batch,config,repeated);
  const auto bytes=[](const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);};
  const bool replay=bytes(result.finalQ,repeated.finalQ)&&bytes(result.finalV,repeated.finalV)&&d.firstGPUStatusCode==d2.firstGPUStatusCode;
  const bool rollback=d.succeeded()||(bytes(result.finalQ,model.defaultQ)&&bytes(result.finalV,model.defaultV));
  const double impulse=result.finalV.size()>1?result.finalV[1]-model.defaultV[1]:0.;
  const unsigned contacts=result.contactStatuses.empty()?0:result.contactStatuses[0].activeContacts;

  const unsigned unresolved=result.environmentStatuses.empty()?0:result.environmentStatuses[0].unresolvedCCDCount;
  // Both torque-free rotation fixtures pass pi/2 during the step. The analytic
  // minimum plane gap of their unforced sweep is independent of endpoint pose.
  const double freeSweepGap=fixture==0?-.102:.1998-std::hypot(.2,.002);
  const bool unresponded=d.succeeded()&&fixture>0&&impulse==0.&&freeSweepGap<0.;
  double gap=0.;if(result.finalQ.size()>=7){const auto& q=result.finalQ;
   const double nx=2*(q[3]*q[4]+q[5]*q[6]),ny=1-2*(q[3]*q[3]+q[5]*q[5]),nz=2*(q[4]*q[5]-q[3]*q[6]);
   gap=q[1]-(.2*std::abs(nx)+.002*std::abs(ny)+.03*std::abs(nz));}
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"limit\":"<<limit<<",\"mode\":"<<mode<<",\"fixture\":"<<fixture<<",\"accepted\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<d.firstGPUStatusCode<<",\"host_status\":"<<int(d.status)<<",\"accepted_steps\":"<<d.successfulStepCount<<",\"gap\":"<<gap<<",\"normal_impulse_Ns\":"<<impulse<<",\"unresolved_ccd\":"<<unresolved<<",\"free_sweep_min_gap\":"<<freeSweepGap<<",\"accepted_unresponded_swept_hit\":"<<(unresponded?"true":"false")<<",\"active_contacts\":"<<contacts<<",\"rejected_state_restored\":"<<(rollback?"true":"false")<<",\"repeat_identical\":"<<(replay?"true":"false")<<",\"elapsed_with_initialization\":"<<elapsed<<"}";
 }
 std::cout<<"]}\n";
}
